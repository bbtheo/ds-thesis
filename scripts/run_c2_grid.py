"""
C2 cross-tab driver: stratified-RANDOM context over fraud_ratio × context_size.

Strengthens the "ratio control, not relevance" argument by mapping the RANDOM
baseline (C2) over the same axes the RAP sweep explored, so RAP-vs-C2 is matched.
Fraud is oversampled WITH REPLACEMENT to hit the target ratio (same policy as the
C4 sampler), so (a) a 40% ratio is actually realised on low-fraud datasets and
(b) the duplication confound is isolated: if forcing a high ratio via duplication
hurts random-C2 the same as retrieved-C4, it's duplication, not retrieval.

Grid: fraud_ratio {0.02,0.04,0.05,0.06,0.08,0.10,0.20,0.30,0.40} (9 ratios; low
ratios added 2026-06-20) × context_size {10k,20k,30k,40k,50k} (5) × FTM
{tabiclv2, tabpfn_3} (2) × dataset {eu_cc,banksim,paysim} (3) × 3 seeds
= 9×5×2×3×3 = 810 cells (see RATIOS/CONTEXT_SIZES/MODELS/UNITS below — the
original docstring said 360, written before the low-ratio points were added).
context_size is capped at min(len(train), model context limit); both requested and
effective are recorded. One model build per (dataset,seed,model), refit per cell.
Idempotent (config hash -> parquet) -> results/runs_c2grid/.

Usage:
    TABPFN_TOKEN=... uv run python scripts/run_c2_grid.py            # run
    uv run python scripts/run_c2_grid.py --dry-run                   # enumerate
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

from src.eval.metrics import compute_metrics
from src.experiment.runner import _PREDICT_CHUNK, _predict_proba_batched
from src.experiment.split_util import load_and_subsample_test
from src.models.ftm import CONTEXT_LIMITS, build_ftm

try:
    import torch
except Exception:
    torch = None

C2_VERSION = 1
UNITS = {"eu_cc": [42, 123, 7], "banksim": [42, 123, 7], "paysim": [42, 123, 7]}
MODELS = ["tabiclv2", "tabpfn_3"]
RATIOS = [0.02, 0.04, 0.05, 0.06, 0.08, 0.10, 0.20, 0.30, 0.40]  # low ratios added 2026-06-20
CONTEXT_SIZES = [10000, 20000, 30000, 40000, 50000]
N_ESTIMATORS = 4
PREDICT_CHUNK = 16384
OUT = Path("results/runs_c2grid")


def cell_configs(dataset, seed, model):
    return [dict(dataset=dataset, seed=seed, model=model, fraud_ratio=r, context_size=c)
            for r in RATIOS for c in CONTEXT_SIZES]


def cell_hash(cfg):
    canonical = json.dumps({
        "study": "c2_grid", "c2_version": C2_VERSION,
        "dataset": cfg["dataset"], "seed": int(cfg["seed"]), "model": cfg["model"],
        "fraud_ratio": cfg["fraud_ratio"], "context_size": int(cfg["context_size"]),
        "n_estimators": N_ESTIMATORS,
    }, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def done_set():
    return {f.stem for f in OUT.glob("*.parquet")} | {f.stem for f in OUT.glob("*.failed")}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--datasets", nargs="*", default=None)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    units = [(d, s) for d, seeds in UNITS.items() for s in seeds
             if (args.datasets is None or d in args.datasets)]
    all_cells = [(d, s, m, c) for (d, s) in units for m in MODELS for c in cell_configs(d, s, m)]
    done = done_set()
    pending = [x for x in all_cells if cell_hash(x[3]) not in done]
    print(f"Total: {len(all_cells)}  pending: {len(pending)}  done: {len(all_cells)-len(pending)}")
    if args.dry_run:
        return

    from importlib.metadata import PackageNotFoundError, version
    libs = {}
    for pkg in ("tabpfn", "tabicl"):
        try: libs[pkg] = version(pkg)
        except PackageNotFoundError: libs[pkg] = None

    n_run = 0
    for (dataset, seed) in units:
        ucells = [(m, c) for m in MODELS for c in cell_configs(dataset, seed, m)]
        done = done_set()
        if all(cell_hash(c) in done for _, c in ucells):
            continue
        print(f"\n### loading {dataset} seed={seed} ###", flush=True)
        Xtr, ytr, Xtes, ytes, info, prevalence, subsampled, _negcap = \
            load_and_subsample_test(dataset, seed)
        fidx = np.where(ytr == 1)[0]
        lidx = np.where(ytr == 0)[0]

        for model in MODELS:
            mcells = [c for c in cell_configs(dataset, seed, model) if cell_hash(c) not in done]
            if not mcells:
                continue
            limit = CONTEXT_LIMITS[model]
            print(f"  -- {model}: {len(mcells)} cells (limit={limit}) --", flush=True)
            mdl = build_ftm(model, device="cuda", seed=seed, n_estimators=N_ESTIMATORS)
            mdl.ensure_loaded()
            for cfg in mcells:
                rid = cell_hash(cfg)
                out_path = OUT / f"{rid}.parquet"
                if out_path.exists():
                    continue
                ratio = cfg["fraud_ratio"]
                ctx_req = cfg["context_size"]
                ctx = min(ctx_req, len(ytr), limit)  # effective context size
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
                    "run_id": rid, "study": "c2_grid", "c2_version": C2_VERSION,
                    "dataset": dataset, "seed": int(seed), "model": model,
                    "fraud_ratio": ratio, "context_size_req": int(ctx_req),
                    "context_size_eff": int(ctx), "n_estimators": N_ESTIMATORS,
                    "pr_auc": mm["pr_auc"], "recall_at_1fpr": mm["recall_at_1fpr"],
                    "recall_at_5fpr": mm["recall_at_5fpr"], "roc_auc": mm["roc_auc"], "f1": mm["f1"],
                    "realized_fraud_ratio": float(int(np.sum(ytr[idx])) / len(idx)),
                    "context_fraud_n": int(nf), "context_fraud_unique": int(nfu),
                    "dup_factor": float(nf / max(nfu, 1)),
                    "n_test_scored": int(len(ytes)), "n_fraud_test_scored": int(ytes.sum()),
                    "prevalence": prevalence, "fit_s": fit_s, "predict_s": pred_s,
                    "tabpfn_lib": libs["tabpfn"], "tabicl_lib": libs["tabicl"],
                }
                pd.DataFrame([row]).to_parquet(out_path, index=False)
                n_run += 1
                print(f"  [{n_run}/{len(pending)}] {dataset} s{seed} {model} r{ratio} "
                      f"ctx{ctx_req//1000}k(eff{ctx//1000}k): PR-AUC={mm['pr_auc']:.4f} "
                      f"dup={nf/max(nfu,1):.1f}x ({fit_s+pred_s:.0f}s)", flush=True)
    print("\nAll done.")


if __name__ == "__main__":
    main()
