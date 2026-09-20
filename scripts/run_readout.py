"""
Decoder-readout case study: where does TabPFN v3 put its attention mass under
random (C1), stratified (C2 at the 10 % plateau ratio) and retrieved (C4, 10 %)
contexts?

Companion to the grid, NOT part of it. Runs in the separate readout environment
(``.venv-readout``: tabpfn>=8.4 + tabpfn-extensions) because the decoder readout
does not exist under the grid's tabpfn 8.0.8. Outputs go to ``results/readout/``
and must never be mixed into ``results/runs``. See ``results/readout/README.md``.

Design (plan_run.md, 2026-09-20)
--------------------------------
* datasets eu_cc and banksim, seed 42, model tabpfn_3, n_estimators = 4,
  grid context limit (50k), same scored test set as the grid (eu_cc full,
  banksim negative cap 30,000);
* contexts are built with the SAME code and RNG streams as the studies they
  mirror, so each cell's context is identical to an existing grid cell:
    C1 -> runner._build_context_c1 with rng(seed)           (results/runs C1)
    C2 -> run_c2_grid sampler at r=0.10, 50k               (results/runs_c2grid)
    C4 -> RetrievalContextBuilder, silhouette grouping,
          per-group rng([seed, g]), fraud_ratio 0.10        (results/runs C4)
* every scored test row is predicted (so PR-AUC etc. can be checked against
  the grid for library-version drift), the readout is taken on a subset: all
  scored test frauds plus 2,000 seeded random scored legits.

Outputs, one set per (dataset, condition):
  results/readout/{dataset}_{condition}.parquet         one row per readout test row
  results/readout/{dataset}_{condition}_top10.parquet   long form: 10 rows per test row
  results/readout/{dataset}_{condition}_metrics.parquet metrics over the full scored set
Idempotent by output path.

Usage:
    TABPFN_TOKEN=... .venv-readout/bin/python scripts/run_readout.py
    .venv-readout/bin/python scripts/run_readout.py --dataset ai_banking   # smoke
"""
from __future__ import annotations

import argparse
import sys
import time
from importlib.metadata import version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.eval.metrics import compute_metrics
from src.experiment.runner import _PREDICT_CHUNK, _build_context_c1, _predict_proba_batched
from src.experiment.split_util import load_and_subsample_test
from src.models.ftm import CONTEXT_LIMITS, build_ftm
from src.rap.context import RetrievalContextBuilder

OUT = Path("results/readout")
MODEL = "tabpfn_3"
SEED = 42
N_ESTIMATORS = 4
C2_RATIO = 0.10          # plateau ratio (C2 grid); NOT the formal grid's 50/50 C2
C4_RATIO = 0.10
C4_METRIC = "cosine"
N_LEGIT_READOUT = 2000   # readout subset: all scored frauds + this many scored legits
READOUT_SEED_SALT = 0x2EAD
READOUT_CHUNK = 512      # rows per readout call: (heads x chunk x 50k) fp32 on GPU
TOP_K = 10
DATASETS = ["eu_cc", "banksim"]
CONDITIONS = ["C1", "C2", "C4"]


def build_c2_plateau(X_train, y_train, ctx_req, limit, seed, ratio):
    """Byte-for-byte the run_c2_grid.py sampler (same RNG stream) at one cell."""
    fidx = np.where(y_train == 1)[0]
    lidx = np.where(y_train == 0)[0]
    ctx = min(ctx_req, len(y_train), limit)
    rng = np.random.default_rng([seed, int(round(ratio * 1000)), ctx_req])
    nf = int(round(ratio * ctx))
    nl = ctx - nf
    nfu = min(nf, len(fidx))
    cf = rng.choice(fidx, nfu, replace=False)
    if nf > nfu and nfu > 0:
        cf = np.concatenate([cf, rng.choice(cf, nf - nfu, replace=True)])
    cl = rng.choice(lidx, min(nl, len(lidx)), replace=False)
    idx = np.concatenate([cf, cl]).astype(np.int64)
    rng.shuffle(idx)
    return idx


def readout_rows(model, X_rows, y_ctx, ctx_train_idx):
    """Readout for a block of test rows against the currently fitted context.

    Returns per-row dicts: fraud/legit attention mass and the top-K context rows
    (as GLOBAL train indices, labels and weights).
    """
    out = []
    fraud_mask = y_ctx == 1
    for start in range(0, len(X_rows), READOUT_CHUNK):
        w, cidx = model.readout(X_rows[start:start + READOUT_CHUNK])
        assert w.shape[1] == len(y_ctx) and np.array_equal(cidx, np.arange(len(y_ctx)))
        fraud_mass = w[:, fraud_mask].sum(axis=1)
        top = np.argsort(-w, axis=1)[:, :TOP_K]
        for i in range(len(w)):
            t = top[i]
            out.append({
                "attn_fraud_mass": float(fraud_mass[i]),
                "attn_legit_mass": float(1.0 - fraud_mass[i]),
                "top10_ctx_train_idx": ctx_train_idx[t].astype(np.int64),
                "top10_ctx_label": y_ctx[t].astype(np.int8),
                "top10_ctx_weight": w[i, t].astype(np.float32),
                "top10_mass": float(w[i, t].sum()),
            })
    return out


def run_cell(dataset, condition, split, model, lib):
    main_path = OUT / f"{dataset}_{condition}.parquet"
    if main_path.exists():
        print(f"[readout] SKIP {dataset} {condition} (exists)", flush=True)
        return
    X_train, y_train, X_test, y_test, info, prevalence, subsampled, negcap = split
    limit = CONTEXT_LIMITS[MODEL]
    context_size = min(len(y_train), limit)
    n_test = len(y_test)

    # Readout subset: all scored frauds + N seeded scored legits (row ids index
    # the SCORED arrays, same convention as results/preds row_idx).
    rng_sub = np.random.default_rng([SEED, READOUT_SEED_SALT])
    fraud_rows = np.where(y_test == 1)[0]
    legit_rows = np.where(y_test == 0)[0]
    legit_pick = rng_sub.choice(legit_rows, size=min(N_LEGIT_READOUT, len(legit_rows)), replace=False)
    sub_rows = np.sort(np.concatenate([fraud_rows, legit_pick]))
    print(f"[readout] {dataset} {condition}: scored={n_test} readout_subset={len(sub_rows)} "
          f"(fraud={len(fraud_rows)}, legit={len(legit_pick)})", flush=True)

    t_cell = time.perf_counter()
    probs = np.full(n_test, np.nan)
    group_of = np.full(n_test, -1, dtype=np.int64)
    rows = {}                       # test_row_id -> readout dict
    ctx_stats = []                  # per group
    n_groups, sil = None, None
    chunk = _PREDICT_CHUNK[MODEL]

    if condition in ("C1", "C2"):
        if condition == "C1":
            rng = np.random.default_rng(SEED)   # runner: rng(seed), C1 is its first consumer
            n = len(y_train)
            idx = rng.choice(n, size=context_size, replace=False) if n > context_size else np.arange(n)
        else:
            idx = build_c2_plateau(X_train, y_train, limit, limit, SEED, C2_RATIO)
        X_ctx, y_ctx = X_train[idx], y_train[idx]
        groups = [(0, np.arange(n_test), idx, X_ctx, y_ctx,
                   {"fraud_n": int(y_ctx.sum()), "fraud_unique": int(len(np.unique(idx[y_ctx == 1])))})]
    else:
        builder = RetrievalContextBuilder(X_train, y_train, metric=C4_METRIC, seed=SEED)
        labels, centres, n_groups, sil = builder.group_test(builder.scale(X_test))
        print(f"[readout] C4 grouping G={n_groups} silhouette={sil:.4f}", flush=True)
        groups = []
        for g in np.unique(labels):
            rng_g = np.random.default_rng([SEED, int(g)])
            # builder.build returns unscaled rows; we also need the global indices,
            # so call the sampler through the builder's own components.
            from src.rap.sampler import sample_c4
            idx, stats = sample_c4(builder.retr_fraud, builder.retr_legit,
                                   builder.fraud_idx, builder.legit_idx,
                                   centres[g], context_size, C4_RATIO, rng_g)
            groups.append((int(g), np.where(labels == g)[0], idx, X_train[idx], y_train[idx], stats))

    for g, members, idx, X_ctx, y_ctx, stats in groups:
        t0 = time.perf_counter()
        model.fit(X_ctx, y_ctx)
        probs[members] = _predict_proba_batched(model, X_test[members], batch_size=chunk)
        group_of[members] = g
        sub_in_g = members[np.isin(members, sub_rows)]
        ro = readout_rows(model, X_test[sub_in_g], y_ctx, idx)
        for r_id, r in zip(sub_in_g, ro):
            rows[int(r_id)] = r
        ctx_stats.append({"group_id": g, "n_rows": len(members), "n_readout": len(sub_in_g),
                          "n_ctx": len(idx), "n_ctx_fraud": stats["fraud_n"],
                          "n_ctx_fraud_unique": stats["fraud_unique"]})
        print(f"  [group {g}] rows={len(members)} readout={len(sub_in_g)} ctx_fraud={stats['fraud_n']} "
              f"(unique={stats['fraud_unique']}) {time.perf_counter() - t0:.0f}s", flush=True)

    assert np.all(np.isfinite(probs))
    wall_s = time.perf_counter() - t_cell
    mm = compute_metrics(y_test, probs, prevalence=prevalence if subsampled else None)
    print(f"[readout] {dataset} {condition}: pr_auc={mm['pr_auc']:.4f} "
          f"r@1={mm['recall_at_1fpr']:.4f} r@5={mm['recall_at_5fpr']:.4f} wall={wall_s:.0f}s", flush=True)

    stat_by_g = {s["group_id"]: s for s in ctx_stats}
    main = pd.DataFrame({
        "dataset": dataset, "condition": condition, "seed": SEED, "model": MODEL,
        "tabpfn_lib": lib, "fraud_ratio": {"C1": None, "C2": C2_RATIO, "C4": C4_RATIO}[condition],
        "test_row_id": sub_rows.astype(np.int64),
        "y_true": y_test[sub_rows].astype(np.int8),
        "y_score": probs[sub_rows],
        "group_id": group_of[sub_rows] if condition == "C4" else pd.array([pd.NA] * len(sub_rows), dtype="Int64"),
        "attn_fraud_mass": [rows[i]["attn_fraud_mass"] for i in sub_rows],
        "attn_legit_mass": [rows[i]["attn_legit_mass"] for i in sub_rows],
        "top10_mass": [rows[i]["top10_mass"] for i in sub_rows],
        "n_ctx_fraud": [stat_by_g[group_of[i]]["n_ctx_fraud"] for i in sub_rows],
        "n_ctx_fraud_unique": [stat_by_g[group_of[i]]["n_ctx_fraud_unique"] for i in sub_rows],
        "n_ctx_legit": [stat_by_g[group_of[i]]["n_ctx"] - stat_by_g[group_of[i]]["n_ctx_fraud"] for i in sub_rows],
        "top10_ctx_train_idx": [rows[i]["top10_ctx_train_idx"] for i in sub_rows],
        "top10_ctx_label": [rows[i]["top10_ctx_label"] for i in sub_rows],
        "top10_ctx_weight": [rows[i]["top10_ctx_weight"] for i in sub_rows],
    })
    long = pd.DataFrame({
        "dataset": dataset, "condition": condition,
        "test_row_id": np.repeat(sub_rows.astype(np.int64), TOP_K),
        "rank": np.tile(np.arange(1, TOP_K + 1), len(sub_rows)),
        "ctx_train_idx": np.concatenate([rows[i]["top10_ctx_train_idx"] for i in sub_rows]),
        "ctx_label": np.concatenate([rows[i]["top10_ctx_label"] for i in sub_rows]),
        "weight": np.concatenate([rows[i]["top10_ctx_weight"] for i in sub_rows]),
    })
    metrics = pd.DataFrame([{
        "dataset": dataset, "condition": condition, "seed": SEED, "model": MODEL,
        "tabpfn_lib": lib, "n_estimators": N_ESTIMATORS,
        "fraud_ratio": {"C1": None, "C2": C2_RATIO, "C4": C4_RATIO}[condition],
        "context_size": context_size, "n_groups": n_groups, "group_silhouette": sil,
        "n_test_scored": n_test, "n_fraud_test_scored": int(y_test.sum()),
        "n_readout_rows": len(sub_rows), **mm, "wall_s": wall_s,
    }]).astype({"n_groups": "Int64", "group_silhouette": "float64"})

    OUT.mkdir(parents=True, exist_ok=True)
    long.to_parquet(OUT / f"{dataset}_{condition}_top10.parquet", index=False)
    metrics.to_parquet(OUT / f"{dataset}_{condition}_metrics.parquet", index=False)
    main.to_parquet(main_path, index=False)   # written last: presence == cell complete
    print(f"[readout] wrote {main_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", nargs="*", default=DATASETS)
    ap.add_argument("--condition", nargs="*", default=CONDITIONS)
    args = ap.parse_args()
    lib = version("tabpfn")
    print(f"[readout] tabpfn {lib}, tabpfn-extensions {version('tabpfn-extensions')}", flush=True)
    model = build_ftm(MODEL, device="cuda", seed=SEED, n_estimators=N_ESTIMATORS)
    model.ensure_loaded()
    for ds in args.dataset:
        pending = [c for c in args.condition if not (OUT / f"{ds}_{c}.parquet").exists()]
        if not pending:
            print(f"[readout] {ds}: all cells present", flush=True)
            continue
        t0 = time.perf_counter()
        split = load_and_subsample_test(ds, SEED)
        print(f"[readout] loaded {ds} seed {SEED} in {time.perf_counter() - t0:.0f}s", flush=True)
        for c in pending:
            run_cell(ds, c, split, model, lib)


if __name__ == "__main__":
    main()
