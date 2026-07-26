"""
Time-split ablation driver: forced temporal 80/20 split at the C2 plateau recipe.

Bounds the temporal-leakage threat of the main grid's random splits (time
columns deliberately kept as features; see sections/5_methods.qmd and appendix
@sec-app-second): each dataset is split at a time-value boundary (train = first
~80% of rows by the native time column, test = the rest; no timestamp straddles
the split), and the stratified-RANDOM C2 context recipe is rerun unchanged on
the temporal train side. Controls are the matched runs_c2grid cells (r=0.10,
50k context, n_estimators=4, same models, same seeds): if the temporal
seed-mean lies inside the control cell's seed mean +/- sd, the random-split
figures show no measurable temporal inflation; otherwise the delta bounds the
combined leakage + shift effect. (Decided 2026-07-26: this band check is the
whole statistical comparison — no bootstrap machinery.)

Design (18 cells): {eu_cc, banksim, paysim} x {tabiclv2, tabpfn_3} x seeds
{42, 123, 7} at fraud_ratio=0.10, context 50k (clamped to the model budget, so
48k on tabiclv2), n_estimators=4, with-replacement fraud top-up (same sampling
code path as run_c2_grid.py). The split is deterministic, so seeds vary only
the context subsample and the test negative-cap subsample. baf is excluded:
fifar already evaluates the same rows under their published temporal protocol.

Protocol consequences (stated in the appendix prose):
- banksim's grouped-by-customer split is replaced; customers straddle the
  boundary (the deployment-realistic case).
- paysim's test period is fraud-heavy by simulator construction (prevalence
  ~0.34% vs 0.13% overall; the simulator stops generating legitimate traffic
  near the end), and its temporal train side holds only ~3,963 unique frauds,
  so the 5,000-row fraud quota duplicates mildly (~1.26x).
- Time features are KEPT (identical feature set to the main grid), so
  test-side time values lie outside the training range by construction.

Idempotent (config hash -> parquet) -> results/runs_timesplit/.

Usage:
    TABPFN_TOKEN=... uv run python scripts/run_timesplit_ablation.py           # run
    uv run python scripts/run_timesplit_ablation.py --dry-run                  # enumerate
    TABPFN_TOKEN=... uv run python scripts/run_timesplit_ablation.py --smoke   # 1 cell
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.data.loader import load_dataset
from src.data.schema import DATASETS
from src.data.splitter import split_info, split_temporal, subsample_test
from src.eval.metrics import compute_metrics
from src.experiment.runner import _predict_proba_batched
from src.models.ftm import CONTEXT_LIMITS, build_ftm

try:
    import torch
except Exception:
    torch = None

TS_VERSION = 1
# native time column per dataset (loader lowercases column names)
TIME_COLS = {"eu_cc": "time", "banksim": "step", "paysim": "step"}
SEEDS = [42, 123, 7]
MODELS = ["tabiclv2", "tabpfn_3"]
RATIO = 0.10
CONTEXT_REQ = 50_000
N_ESTIMATORS = 4
PREDICT_CHUNK = 16384
OUT = Path("results/runs_timesplit")


def cell_hash(dataset: str, seed: int, model: str) -> str:
    canonical = json.dumps({
        "study": "timesplit", "ts_version": TS_VERSION,
        "dataset": dataset, "seed": int(seed), "model": model,
        "fraud_ratio": RATIO, "context_size": CONTEXT_REQ,
        "n_estimators": N_ESTIMATORS,
    }, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def done_set():
    return {f.stem for f in OUT.glob("*.parquet")} | {f.stem for f in OUT.glob("*.failed")}


def load_temporal(dataset_id: str):
    """Deterministic temporal 80/20 split (seed-independent)."""
    X, y, feature_names = load_dataset(dataset_id)
    t = X[:, feature_names.index(TIME_COLS[dataset_id])]
    X_train, X_test, y_train, y_test, boundary = split_temporal(X, y, t)
    info = split_info(y_train, y_test)
    prevalence = info["n_fraud_test"] / info["n_test"]
    negcap = DATASETS[dataset_id].get("test_neg_cap")
    return X_train, y_train, X_test, y_test, info, prevalence, negcap, boundary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="run only the first pending cell (banksim is cheapest to load)")
    ap.add_argument("--datasets", nargs="*", default=None)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    datasets = [d for d in ("banksim", "eu_cc", "paysim")
                if args.datasets is None or d in args.datasets]
    all_cells = [(d, s, m) for d in datasets for s in SEEDS for m in MODELS]
    done = done_set()
    pending = [c for c in all_cells if cell_hash(*c) not in done]
    print(f"Total: {len(all_cells)}  pending: {len(pending)}  done: {len(all_cells) - len(pending)}")
    if args.dry_run:
        return
    if args.smoke:
        pending = pending[:1]

    from importlib.metadata import PackageNotFoundError, version
    libs = {}
    for pkg in ("tabpfn", "tabicl"):
        try: libs[pkg] = version(pkg)
        except PackageNotFoundError: libs[pkg] = None

    n_run = 0
    for dataset in datasets:
        dcells = [(s, m) for s in SEEDS for m in MODELS
                  if cell_hash(dataset, s, m) not in done and (dataset, s, m) in pending]
        if not dcells:
            continue
        print(f"\n### loading {dataset} (temporal split) ###", flush=True)
        Xtr, ytr, Xte_full, yte_full, info, prevalence, negcap, boundary = load_temporal(dataset)
        print(f"  boundary t={boundary:.0f}  train {info['n_train']}/{info['n_fraud_train']} fraud"
              f"  test {info['n_test']}/{info['n_fraud_test']} fraud"
              f" (prev {100 * prevalence:.3f}%)", flush=True)
        fidx = np.where(ytr == 1)[0]
        lidx = np.where(ytr == 0)[0]

        for seed, model in dcells:
            rid = cell_hash(dataset, seed, model)
            out_path = OUT / f"{rid}.parquet"
            if out_path.exists():
                continue
            Xtes, ytes = subsample_test(Xte_full, yte_full, seed=seed, max_negatives=negcap)
            subsampled = len(ytes) < info["n_test"]

            ctx = min(CONTEXT_REQ, len(ytr), CONTEXT_LIMITS[model])
            # same RNG stream construction and top-up policy as run_c2_grid.py
            rng = np.random.default_rng([seed, int(round(RATIO * 1000)), CONTEXT_REQ])
            nf = int(round(RATIO * ctx))
            nl = ctx - nf
            nfu = min(nf, len(fidx))
            cf = rng.choice(fidx, nfu, replace=False)
            if nf > nfu and nfu > 0:
                cf = np.concatenate([cf, rng.choice(cf, nf - nfu, replace=True)])
            cl = rng.choice(lidx, min(nl, len(lidx)), replace=False)
            idx = np.concatenate([cf, cl]).astype(np.int64)
            rng.shuffle(idx)

            mdl = build_ftm(model, device="cuda", seed=seed, n_estimators=N_ESTIMATORS)
            mdl.ensure_loaded()
            t0 = time.perf_counter(); mdl.fit(Xtr[idx], ytr[idx]); fit_s = time.perf_counter() - t0
            t0 = time.perf_counter()
            probs = _predict_proba_batched(mdl, Xtes, batch_size=PREDICT_CHUNK)
            pred_s = time.perf_counter() - t0
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
            if not np.all(np.isfinite(probs)):
                (OUT / f"{rid}.failed").write_text("non_finite\n")
                print(f"  !! non-finite {rid}", flush=True); continue
            mm = compute_metrics(ytes, probs, prevalence=prevalence if subsampled else None)
            row = {
                "run_id": rid, "study": "timesplit", "ts_version": TS_VERSION,
                "dataset": dataset, "seed": int(seed), "model": model,
                "split_rule": "temporal_80_20", "time_col": TIME_COLS[dataset],
                "boundary_t": float(boundary),
                "fraud_ratio": RATIO, "context_size_req": CONTEXT_REQ,
                "context_size_eff": int(ctx), "n_estimators": N_ESTIMATORS,
                "pr_auc": mm["pr_auc"], "recall_at_1fpr": mm["recall_at_1fpr"],
                "recall_at_5fpr": mm["recall_at_5fpr"], "roc_auc": mm["roc_auc"], "f1": mm["f1"],
                "realized_fraud_ratio": float(int(np.sum(ytr[idx])) / len(idx)),
                "context_fraud_n": int(nf), "context_fraud_unique": int(nfu),
                "dup_factor": float(nf / max(nfu, 1)),
                "n_train": info["n_train"], "n_test": info["n_test"],
                "n_fraud_train": info["n_fraud_train"], "n_fraud_test": info["n_fraud_test"],
                "n_test_scored": int(len(ytes)), "n_fraud_test_scored": int(ytes.sum()),
                "test_subsample_rule": f"neg_cap_{negcap}" if subsampled else "full",
                "prevalence": prevalence, "fit_s": fit_s, "predict_s": pred_s,
                "tabpfn_lib": libs["tabpfn"], "tabicl_lib": libs["tabicl"],
            }
            pd.DataFrame([row]).to_parquet(out_path, index=False)
            n_run += 1
            print(f"  [{n_run}/{len(pending)}] {dataset} s{seed} {model}"
                  f" ctx{ctx // 1000}k: PR-AUC={mm['pr_auc']:.4f}"
                  f" R@5FPR={mm['recall_at_5fpr']:.4f} dup={nf / max(nfu, 1):.1f}x"
                  f" ({fit_s + pred_s:.0f}s)", flush=True)
    print("\nAll done.")


if __name__ == "__main__":
    main()
