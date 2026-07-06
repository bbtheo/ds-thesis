"""
RAP design-space grid driver (4 grids).

Supersedes the silhouette-auto cosine C3/C4 grid. Tests whether ANY retrieval
design beats the random/stratified baselines, across:

  grid 1: metric=cosine    fraud=local(kNN)   legit=local(kNN)   [ratio sweep]
  grid 2: metric=euclidean fraud=local(kNN)   legit=local(kNN)   [ratio sweep]
  grid 3: metric=cosine    fraud=global(all)  legit=local(kNN)   [no ratio]
  grid 4: metric=euclidean fraud=global(all)  legit=local(kNN)   [no ratio]

Forced granularity G in {8,16,32,64} (NOT silhouette — that gave the fatal G=2).
FTMs: tabiclv2 + tabpfn_3.

Datasets — DELIBERATELY RESTRICTED to the 3 separable datasets (eu_cc, banksim,
paysim), NOT the full formal grid. baf and fifar sit at a separability floor
(see CLAUDE.md / scratchpad notes) where this design-space sweep is not
informative, and a prior full-formal-grid run of this script left partial baf
output in results/diag after burning a large amount of GPU time on cells that
were never going to be used. FORMAL_UNITS below is intentionally the 3-dataset
subset, not the 5-dataset formal grid used by run_pending_c1c2.py/run_pending_c3c4.py
— do not add baf/fifar back without re-deciding this scope; use --datasets to
run an even smaller subset, not to add datasets outside the 3.

Per (dataset,seed): load split once, fit StandardScaler, build cosine+euclidean
retrievers, KMeans groupings per G, and cache nearest-neighbour orders per
(metric,G) so the 5 ratios reuse one retrieval. Per model: build/ensure_loaded
once, reuse across its 40 cells. Each cell writes one idempotent parquet
(config hash -> filename), so the multi-day run is fully restartable. VRAM is
released between per-group fits (high-G otherwise hard-faults).

Counts: with STRATEGIES=["local"] (see below — 'global' was dropped as
decisively worst), the real enumeration is 2 metrics × 4 G × 5 ratios × 2 FTM
× 13 units = 1040 (the docstring previously said 1248, stale from when
STRATEGIES still included 'global'). Of those 1040, only the 720 cells for the
3-dataset scope above (2×4×5×2×9 units) exist in results/runs_rapds; re-running
this script with the full 13-unit formal grid would additionally launch ~320
baf/fifar cells, which is why the dataset list is gated below.

Usage:
    TABPFN_TOKEN=... uv run python scripts/run_rap_designspace.py            # run
    uv run python scripts/run_rap_designspace.py --dry-run                   # enumerate
    TABPFN_TOKEN=... uv run python scripts/run_rap_designspace.py --datasets eu_cc  # subset
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
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.eval.metrics import compute_metrics
from src.experiment.runner import _PREDICT_CHUNK, _predict_proba_batched
from src.experiment.split_util import load_and_subsample_test
from src.models.ftm import CONTEXT_LIMITS, build_ftm
from src.rap.retriever import KNNRetriever

try:
    import torch
except Exception:
    torch = None

DS_VERSION = 1
# GATED SCOPE (2026-07-06 review): only the 3 separable datasets, NOT the full
# 5-dataset formal grid. baf/fifar sit at a separability floor where this sweep
# is not informative, and a prior run left partial baf output in results/diag
# after burning GPU time on cells that were never going to be used. Do NOT add
# "fifar"/"baf" back here without re-deciding the scope (see module docstring);
# --datasets only narrows this set further, it cannot widen it.
FORMAL_UNITS = {
    "banksim": [42, 123, 7],
    "eu_cc":   [42, 123, 7],
    "paysim":  [42, 123, 7],
}
MODELS = ["tabiclv2", "tabpfn_3"]
METRICS = ["cosine", "euclidean"]
GRANULARITIES = [8, 16, 32, 64]
RATIOS = [0.05, 0.10, 0.20, 0.30, 0.50]
# Strategies to enumerate. 'global' (all-fraud + local-legit) was decisively the
# worst design on baf (g64_allF=0.075 vs C1=0.162), so the trimmed confirmation
# runs 'local' only; keep the global code path for generality.
STRATEGIES = ["local"]
N_ESTIMATORS = 4
PREDICT_CHUNK = 16384            # per-group predicts are tiny; this only helps coarse G
OUT = Path("results/runs_rapds")


def done_set() -> set[str]:
    """Completed cells = written parquets + .failed sentinels (so deterministically
    non-finite cells are not retried on every restart of the multi-day run)."""
    return ({f.stem for f in OUT.glob("*.parquet")}
            | {f.stem for f in OUT.glob("*.failed")})


def cell_configs(dataset: str, seed: int, model: str) -> list[dict]:
    cfgs = []
    for metric in METRICS:
        for G in GRANULARITIES:
            if "local" in STRATEGIES:  # grids 1/2: local fraud, ratio sweep
                for ratio in RATIOS:
                    cfgs.append(dict(dataset=dataset, seed=seed, model=model, metric=metric,
                                     granularity=G, fraud_strategy="local", fraud_ratio=ratio))
            if "global" in STRATEGIES:  # grids 3/4: global fraud (all train frauds), no ratio
                cfgs.append(dict(dataset=dataset, seed=seed, model=model, metric=metric,
                                 granularity=G, fraud_strategy="global", fraud_ratio=None))
    return cfgs


def cell_hash(cfg: dict) -> str:
    canonical = json.dumps({
        "study": "rap_designspace", "ds_version": DS_VERSION,
        "dataset": cfg["dataset"], "seed": int(cfg["seed"]), "model": cfg["model"],
        "metric": cfg["metric"], "granularity": int(cfg["granularity"]),
        "fraud_strategy": cfg["fraud_strategy"],
        "fraud_ratio": cfg["fraud_ratio"], "n_estimators": N_ESTIMATORS,
    }, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--datasets", nargs="*", default=None, help="subset of datasets")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    units = [(d, s) for d, seeds in FORMAL_UNITS.items()
             for s in seeds if (args.datasets is None or d in args.datasets)]
    all_cells = [(d, s, m, c) for (d, s) in units for m in MODELS
                 for c in cell_configs(d, s, m)]
    done = done_set()
    pending = [(d, s, m, c) for (d, s, m, c) in all_cells if cell_hash(c) not in done]
    print(f"Total cells: {len(all_cells)}  |  pending: {len(pending)}  |  done: {len(all_cells)-len(pending)}")
    if args.dry_run:
        from collections import Counter
        by_grid = Counter((c["metric"], c["fraud_strategy"]) for _, _, _, c in pending)
        for k, v in sorted(by_grid.items()):
            print(f"  pending {k}: {v}")
        return

    lib_versions = {}
    from importlib.metadata import PackageNotFoundError, version
    for pkg in ("tabpfn", "tabicl"):
        try:
            lib_versions[pkg] = version(pkg)
        except PackageNotFoundError:
            lib_versions[pkg] = None

    MAXCTX = max(CONTEXT_LIMITS[m] for m in MODELS)
    n_run = 0
    for (dataset, seed) in units:
        unit_cells = [(m, c) for m in MODELS for c in cell_configs(dataset, seed, m)]
        done = done_set()
        if all(cell_hash(c) in done for _, c in unit_cells):
            continue
        print(f"\n### loading {dataset} seed={seed} ###", flush=True)
        Xtr, ytr, Xtes, ytes, info, prevalence, subsampled, negcap = \
            load_and_subsample_test(dataset, seed)
        sub_rule = f"neg_cap_{negcap}" if subsampled else "full"

        scaler = StandardScaler().fit(Xtr)
        Xs = scaler.transform(Xtr)
        Xtes_s = scaler.transform(Xtes)
        fidx = np.where(ytr == 1)[0]
        lidx = np.where(ytr == 0)[0]
        retrievers = {m: (KNNRetriever(m).fit(Xs[fidx]), KNNRetriever(m).fit(Xs[lidx]))
                      for m in METRICS}
        groupings: dict[int, tuple] = {}
        for G in GRANULARITIES:
            raw = KMeans(n_clusters=G, random_state=seed, n_init=10).fit_predict(Xtes_s)
            # Canonicalise like context.py._canonicalise: order groups by smallest member
            # index (stable across KMeans's multithreaded label permutation), and recompute
            # centres as exact means (kills cluster_centers_ float noise). Keeps the per-group
            # RNG keying reproducible across reruns.
            order = sorted(np.unique(raw), key=lambda g: int(np.flatnonzero(raw == g).min()))
            remap = {old: new for new, old in enumerate(order)}
            lab = np.array([remap[g] for g in raw])
            centres = np.vstack([Xtes_s[lab == i].mean(0) for i in range(len(order))])
            groupings[G] = (lab, centres)
        # nearest-neighbour orders per (metric, G): retrieve once at MAXCTX depth, reuse across ratios
        orders: dict[tuple, tuple] = {}

        def get_orders(metric, G):
            key = (metric, G)
            if key not in orders:
                lab, centres = groupings[G]
                rf, rl = retrievers[metric]
                fo = [fidx[rf.query(centres[g], min(MAXCTX, len(fidx)))] for g in range(len(centres))]
                lo = [lidx[rl.query(centres[g], min(MAXCTX, len(lidx)))] for g in range(len(centres))]
                orders[key] = (lab, centres, fo, lo)
            return orders[key]

        for model in MODELS:
            mcells = [c for c in cell_configs(dataset, seed, model) if cell_hash(c) not in done]
            if not mcells:
                continue
            ctx = min(len(ytr), CONTEXT_LIMITS[model])
            print(f"  -- {model}: {len(mcells)} pending cells (ctx={ctx}) --", flush=True)
            mdl = build_ftm(model, device="cuda", seed=seed, n_estimators=N_ESTIMATORS)
            mdl.ensure_loaded()
            for cfg in mcells:
                rid = cell_hash(cfg)
                out_path = OUT / f"{rid}.parquet"
                if out_path.exists():
                    continue
                metric, G = cfg["metric"], cfg["granularity"]
                strat, ratio = cfg["fraud_strategy"], cfg["fraud_ratio"]
                lab, centres, fo, lo = get_orders(metric, G)
                # build per-group contexts
                if strat == "global":
                    nf = min(len(fidx), ctx)  # clamp: all frauds, but never exceed context
                else:
                    nf = int(round(ratio * ctx))
                nl = ctx - nf
                idx_pg, funiq = [], min(nf, len(fidx))
                for g in range(len(centres)):
                    if strat == "global":
                        cf = fidx
                    else:
                        cf = fo[g][:funiq]
                        if nf > funiq and funiq > 0:
                            rng = np.random.default_rng([seed, g, 7])
                            cf = np.concatenate([cf, rng.choice(cf, nf - funiq, replace=True)])
                    cl = lo[g][:min(nl, len(lidx))]
                    idx = np.concatenate([cf, cl]).astype(np.int64)
                    np.random.default_rng([seed, g]).shuffle(idx)
                    idx_pg.append(idx)

                probs = np.empty(len(ytes), dtype=np.float64)
                t_fit = t_pred = 0.0
                for g in range(len(idx_pg)):
                    idx = idx_pg[g]
                    t0 = time.perf_counter(); mdl.fit(Xtr[idx], ytr[idx]); t_fit += time.perf_counter() - t0
                    m = lab == g
                    if m.any():
                        t0 = time.perf_counter()
                        probs[m] = _predict_proba_batched(mdl, Xtes[m], batch_size=PREDICT_CHUNK)
                        t_pred += time.perf_counter() - t0
                    if torch is not None and torch.cuda.is_available():
                        torch.cuda.empty_cache()

                if not np.all(np.isfinite(probs)):
                    # Mark done with a sentinel so a deterministically-bad cell is not
                    # retried (burning a full fit+predict) on every restart.
                    (OUT / f"{rid}.failed").write_text("non_finite_probs\n")
                    print(f"  !! non-finite probs in {rid} -> wrote .failed sentinel", flush=True)
                    continue
                mm = compute_metrics(ytes, probs, prevalence=prevalence if subsampled else None)
                yt = ytes.astype(bool)
                row = {
                    "run_id": rid, "study": "rap_designspace", "ds_version": DS_VERSION,
                    "dataset": dataset, "seed": int(seed), "model": model,
                    "metric": metric, "granularity": int(G),
                    "fraud_strategy": strat, "fraud_ratio": ratio,
                    "n_estimators": N_ESTIMATORS,
                    "pr_auc": mm["pr_auc"], "recall_at_1fpr": mm["recall_at_1fpr"],
                    "recall_at_5fpr": mm["recall_at_5fpr"], "roc_auc": mm["roc_auc"], "f1": mm["f1"],
                    "legit_score": float(probs[~yt].mean()),
                    "n_groups": int(len(idx_pg)), "context_size": int(ctx),
                    "context_fraud_n": int(nf), "context_fraud_unique": int(funiq),
                    "n_train": info["n_train"], "n_test": info["n_test"],
                    "n_fraud_train": info["n_fraud_train"], "n_fraud_test": info["n_fraud_test"],
                    "n_test_scored": int(len(ytes)), "n_fraud_test_scored": int(ytes.sum()),
                    "test_subsample_rule": sub_rule, "prevalence": prevalence,
                    "fit_s": t_fit, "predict_s": t_pred,
                    "tabpfn_lib": lib_versions["tabpfn"], "tabicl_lib": lib_versions["tabicl"],
                }
                pd.DataFrame([row]).astype({"fraud_ratio": "float64"}).to_parquet(out_path, index=False)
                n_run += 1
                print(f"  [{n_run}/{len(pending)}] {dataset} s{seed} {model} {metric} G{G} "
                      f"{strat}{f' r{ratio}' if ratio else ''}: PR-AUC={mm['pr_auc']:.4f} "
                      f"(C1 bar varies) fit={t_fit:.0f}s pred={t_pred:.0f}s", flush=True)

    print("\nAll done.")


if __name__ == "__main__":
    main()
