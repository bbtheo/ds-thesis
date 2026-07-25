"""
eu_cc SOTA-comparison driver: stratified 5-fold CV, full training fold as context.

Best-effort protocol for the external-SOTA comparison chapter (NOT part of the
formal C1-C4 grid; n_estimators=12 here vs the grid's settled 4, so these rows
must never be mixed into grid pivots). Design decided 2026-07-24 from the
runs_c2grid evidence:

- Stratified 5-fold CV over the FULL 284,807-row dataset (fold split seed
  CV_SEED=42), each test fold scored in full (eu_cc has test_neg_cap=None).
- Context = the ENTIRE ~228k-row training fold (no subsampling). This inherits
  C2's real eu_cc advantage (all train frauds guaranteed present) at the top of
  the context-size curve (50k->200k plateau; 200k >= 50k on 2/3 seeds for both
  models at ratio 0.02).
- Two arms: natural ratio (no oversampling; ~0.17% fraud) and a 2% light-
  duplication hedge (all negatives kept, fraud topped up with replacement to a
  2% fraction — 0.17% at full context is the only point below the c2grid
  sweep's tested floor). Higher ratios are excluded: the sweep shows they hurt
  (H3 duplication degeneracy).
- Models: tabpfn_3 + tabiclv2 only (v3 checkpoint accepts 1M rows; tabiclv2 has
  no hard cap but 48k is its pre-training bound — full-fold contexts are
  extrapolation and must be flagged as such, same caveat as the c2grid
  100k/200k cells).
- Per-fold predictions are saved to results/runs_sota_cv/preds/ so pooled
  metrics (every one of the 492 frauds scored exactly once across folds) can be
  computed at analysis time alongside per-fold mean +/- sd.

Idempotent (config hash -> parquet) -> results/runs_sota_cv/.

Usage:
    TABPFN_TOKEN=... uv run python scripts/run_eucc_sota_cv.py            # run
    uv run python scripts/run_eucc_sota_cv.py --dry-run                   # enumerate
    TABPFN_TOKEN=... uv run python scripts/run_eucc_sota_cv.py --smoke    # 1 cell
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
from sklearn.model_selection import StratifiedKFold

from src.data.loader import load_dataset
from src.eval.metrics import compute_metrics
from src.experiment.runner import _predict_proba_batched
from src.models.ftm import build_ftm

try:
    import torch
except Exception:
    torch = None

SOTA_CV_VERSION = 1
DATASET = "eu_cc"
CV_SEED = 42
N_SPLITS = 5
MODELS = ["tabpfn_3", "tabiclv2"]  # tabpfn_3 first: headline model
ARMS = [None, 0.02]  # fraud-ratio arms; None = natural (no oversampling)
N_ESTIMATORS = 12
CONTEXT_LIMIT_OVERRIDE = 300_000  # full fold ~228k (+2% top-up ~232k) < v3's 1M cap
# Predict chunking is result-neutral; 16384 is the proven run_c2_grid value.
# TabICL's forward OOMs at the 228k context on 12GB VRAM (~23GB context-encoder
# output, chunk-size invariant: 24.3GB at chunk 16384 vs 22.9GB at 2048), so
# tabiclv2 runs with the library's disk offload instead of a smaller chunk.
PREDICT_CHUNK = {"tabpfn_3": 16384, "tabiclv2": 16384}
TABICL_OFFLOAD_DIR = Path.home() / ".cache" / "tabicl_offload"  # NOT /tmp (tmpfs)
OUT = Path("results/runs_sota_cv")
PREDS = OUT / "preds"


def cell_configs():
    return [dict(model=m, fraud_ratio=r, fold=f)
            for m in MODELS for r in ARMS for f in range(N_SPLITS)]


def cell_hash(cfg):
    canonical = json.dumps({
        "study": "sota_cv", "version": SOTA_CV_VERSION, "dataset": DATASET,
        "cv_seed": CV_SEED, "n_splits": N_SPLITS, "model": cfg["model"],
        "fraud_ratio": cfg["fraud_ratio"], "fold": int(cfg["fold"]),
        "n_estimators": N_ESTIMATORS,
    }, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def build_context(y_train_idx_fraud, y_train_idx_legit, ratio, rng):
    """Return context row indices (into the training fold) for one arm.

    Natural arm: the whole fold. Ratio arm: all negatives + all unique frauds,
    topped up with replacement until the fraud fraction reaches ``ratio``.
    """
    if ratio is None:
        idx = np.concatenate([y_train_idx_fraud, y_train_idx_legit])
        nf = nfu = len(y_train_idx_fraud)
    else:
        n_legit = len(y_train_idx_legit)
        nf = int(np.ceil(ratio / (1.0 - ratio) * n_legit))
        nfu = len(y_train_idx_fraud)
        cf = y_train_idx_fraud
        if nf > nfu:
            cf = np.concatenate([cf, rng.choice(cf, nf - nfu, replace=True)])
        idx = np.concatenate([cf, y_train_idx_legit])
    idx = idx.astype(np.int64)
    rng.shuffle(idx)
    return idx, nf, nfu


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="run first pending cell, then exit")
    ap.add_argument("--models", nargs="*", default=None)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    PREDS.mkdir(parents=True, exist_ok=True)

    cells = [c for c in cell_configs()
             if args.models is None or c["model"] in args.models]
    done = {f.stem for f in OUT.glob("*.parquet")} | {f.stem for f in OUT.glob("*.failed")}
    pending = [c for c in cells if cell_hash(c) not in done]
    print(f"Total: {len(cells)}  pending: {len(pending)}  done: {len(cells) - len(pending)}")
    if args.dry_run or not pending:
        return

    from importlib.metadata import PackageNotFoundError, version
    libs = {}
    for pkg in ("tabpfn", "tabicl"):
        try: libs[pkg] = version(pkg)
        except PackageNotFoundError: libs[pkg] = None

    X, y, _feature_names = load_dataset(DATASET)
    y = y.astype(np.int64)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=CV_SEED)
    folds = list(skf.split(X, y))
    print(f"{DATASET}: {len(y)} rows, {int(y.sum())} frauds, "
          f"{N_SPLITS}-fold stratified CV (cv_seed={CV_SEED})")

    n_run = 0
    for model in MODELS:
        mcells = [c for c in pending if c["model"] == model]
        if not mcells:
            continue
        print(f"\n### {model} n_est={N_ESTIMATORS}: {len(mcells)} cells ###", flush=True)
        offload = None
        if model == "tabiclv2":
            TABICL_OFFLOAD_DIR.mkdir(parents=True, exist_ok=True)
            offload = str(TABICL_OFFLOAD_DIR)
        mdl = build_ftm(model, device="cuda", seed=CV_SEED, n_estimators=N_ESTIMATORS,
                        context_limit_override=CONTEXT_LIMIT_OVERRIDE,
                        disk_offload_dir=offload)
        mdl.ensure_loaded()
        for cfg in mcells:
            rid = cell_hash(cfg)
            out_path = OUT / f"{rid}.parquet"
            if out_path.exists():
                continue
            fold = cfg["fold"]
            ratio = cfg["fraud_ratio"]
            tr_idx, te_idx = folds[fold]
            ytr = y[tr_idx]
            fidx = tr_idx[ytr == 1]
            lidx = tr_idx[ytr == 0]
            ratio_key = 0 if ratio is None else int(round(ratio * 1000))
            rng = np.random.default_rng([CV_SEED, fold, ratio_key])
            idx, nf, nfu = build_context(fidx, lidx, ratio, rng)

            t0 = time.perf_counter(); mdl.fit(X[idx], y[idx]); fit_s = time.perf_counter() - t0
            t0 = time.perf_counter()
            probs = _predict_proba_batched(mdl, X[te_idx], batch_size=PREDICT_CHUNK[model])
            pred_s = time.perf_counter() - t0
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
            if not np.all(np.isfinite(probs)):
                (OUT / f"{rid}.failed").write_text("non_finite\n")
                print(f"  !! non-finite {rid}", flush=True)
                continue
            yte = y[te_idx]
            mm = compute_metrics(yte, probs, prevalence=None)  # full fold scored, no correction

            pd.DataFrame({
                "run_id": rid, "fold": fold,
                "row_idx": te_idx.astype(np.int64),
                "y_true": yte.astype(np.int8), "prob": probs.astype(np.float32),
            }).to_parquet(PREDS / f"{rid}.parquet", index=False)
            row = {
                "run_id": rid, "study": "sota_cv", "sota_cv_version": SOTA_CV_VERSION,
                "dataset": DATASET, "cv_seed": CV_SEED, "n_splits": N_SPLITS,
                "fold": fold, "model": model,
                "fraud_ratio": np.nan if ratio is None else float(ratio),
                "arm": "natural" if ratio is None else f"ratio_{ratio}",
                "n_estimators": N_ESTIMATORS,
                "context_size": int(len(idx)), "context_fraud_n": int(nf),
                "context_fraud_unique": int(nfu), "dup_factor": float(nf / max(nfu, 1)),
                "realized_fraud_ratio": float(nf / len(idx)),
                "n_train_fold": int(len(tr_idx)), "n_test_fold": int(len(te_idx)),
                "n_fraud_test_fold": int(yte.sum()),
                "pr_auc": mm["pr_auc"], "recall_at_1fpr": mm["recall_at_1fpr"],
                "recall_at_5fpr": mm["recall_at_5fpr"], "roc_auc": mm["roc_auc"],
                "f1": mm["f1"], "fit_s": fit_s, "predict_s": pred_s,
                "tabpfn_lib": libs["tabpfn"], "tabicl_lib": libs["tabicl"],
            }
            pd.DataFrame([row]).to_parquet(out_path, index=False)
            n_run += 1
            print(f"  [{n_run}/{len(pending)}] {model} fold{fold} "
                  f"{'natural' if ratio is None else f'r{ratio}'} "
                  f"ctx={len(idx)} dup={nf / max(nfu, 1):.1f}x: "
                  f"PR-AUC={mm['pr_auc']:.4f} R@1={mm['recall_at_1fpr']:.4f} "
                  f"({fit_s + pred_s:.0f}s)", flush=True)
            if args.smoke:
                print("Smoke cell complete.")
                return
    print("\nAll done.")


if __name__ == "__main__":
    main()
