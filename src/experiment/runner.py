"""
Single-run experiment orchestrator.

Each run is fully specified by a config dict:
    dataset     : str   — dataset key from schema.py
    model       : str   — model key ("xgboost", "catboost", "tabpfn_v2", "tabpfn_25", "tabpfn_26", "tabiclv2")
    condition   : str   — "C1", "C2", "C3", "C4"  (C3/C4 = RAP retrieval, Phase 3)
    seed        : int   — random seed (42, 123, or 7)
    fraud_ratio : float | None — target fraud ratio in context (C4 only)
    metric      : str | None   — retrieval metric "cosine"/"euclidean" (C3/C4 only;
                                 defaults to "cosine")
    batch_size  : int | None   — IGNORED for C3/C4 (the per-group design auto-picks
                                 the number of groups; normalised to None) and for
                                 C1/C2 (fixed context). Not a result-affecting factor.
    k           : int | None   — IGNORED (C3/C4 retrieval depth is derived from
                                 context_size; normalised to None, out of the hash)

The run_id is the first 12 hex chars of the SHA-256 hash of the canonically
serialised config (includes a pipeline schema version).
Output goes to results/runs/{run_id}.parquet.
Runner is idempotent: skips if output file already exists.

Usage
-----
    from src.experiment.runner import run
    run({"dataset": "ai_banking", "model": "xgboost", "condition": "C1",
         "seed": 42, "batch_size": 128, "k": None, "fraud_ratio": None, "metric": None})

CLI
---
    uv run python -m src.experiment.runner \\
        --dataset ai_banking --model xgboost --condition C1 --seed 42
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.loader import load_dataset, load_dataset_split
from src.data.schema import DATASETS
from src.data.splitter import split, split_info, subsample_test
from src.eval.metrics import compute_metrics
from src.models.ftm import CONTEXT_LIMITS, FTM_MODELS, build_ftm
from src.models.gbdt import GBDT_MODELS, build_gbdt

RESULTS_DIR = Path("results/runs")
GBDT_KEYS = list(GBDT_MODELS.keys())

# C3/C4 (RAP retrieval) run on these FTMs only — a settled grid-design decision
# (Phase 3, 2026-06-18); enforced in _validate_config.
_RAP_FTM_MODELS = ("tabpfn_3", "tabiclv2")
SEEDS = [42, 123, 7]

# Bump this whenever model defaults or pipeline logic changes to
# invalidate cached results.  This is included in the config hash.
# v3: banksim grouped split by customer (entity-ID leakage fix),
#     recall@FPR switched from ROC interpolation to conservative step value,
#     batch_size normalised to None for C1/C2, timing split into
#     setup_s / fit_s / predict_s.
# v4: negative-subsampled test set on large datasets (keep all positives, cap
#     negatives via schema test_neg_cap) with prevalence-corrected PR-AUC;
#     n_estimators added to the config (default 8); model-aware predict chunk
#     (does not affect results); new result columns (n_test_scored,
#     n_fraud_test_scored, test_subsample_seed/rule, context_fraud_unique,
#     tabpfn_lib/tabicl_lib library versions).
_PIPELINE_VERSION = 4

# Use 100% of FTM context limit — the limit is what the model can handle.
# Leaving headroom wastes context, especially costly for rare fraud rows.
_CONTEXT_FRACTION = 1.0

# Predict-time GPU memory chunk size per model. This does NOT affect results —
# test rows are independent given the fixed C1/C2 context, so chunking only
# groups rows. Larger chunks amortise context encoding. Conservative defaults;
# re-probe peak VRAM on the 50k-context models / wide datasets before raising.
_PREDICT_CHUNK = {
    "tabpfn_v2": 2048,
    "tabpfn_25": 1024,
    "tabpfn_26": 1024,
    "tabpfn_3":  1024,
    "tabiclv2":  2048,
}

# FTM ensemble passes. None => package default (8). Reducing it speeds predict
# ~linearly but changes outputs — keep 8 for the headline grid until the
# n=8 vs n=4 A/B (eu_cc + fifar, all FTMs/seeds) confirms metric stability.
_N_ESTIMATORS_DEFAULT = 8


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _canonical_config(cfg: dict) -> dict:
    """Return a normalised config dict with all keys present and sorted."""
    # batch_size is an experimental factor only for C3/C4, where the context
    # is retrieved per test batch. Under C1/C2 the context is fixed and test
    # rows are independent, so prediction chunking cannot affect results —
    # batch_size is normalised to None (like k/fraud_ratio/metric) to keep
    # it out of the run_id hash and the results row.
    cond = cfg["condition"]
    # Normalise factors that do not affect a given condition's result to None so a
    # stray value cannot fork the run_id of an otherwise-identical run (idempotency).
    #
    # C3/C4 use the per-group RAP path: the number of groups is chosen automatically
    # (silhouette) and the retrieval depth is derived from context_size, so neither
    # batch_size nor k is an experimental factor — both are normalised to None and
    # stay out of the hash. The real C3/C4 factors are `metric` (default "cosine")
    # and, for C4, `fraud_ratio`. C1/C2 normalisation is unchanged, so their hashes
    # and cached results are untouched.
    if cond in ("C3", "C4"):
        batch_size = None
        k = None
        metric = cfg.get("metric") or "cosine"
    else:
        batch_size = k = metric = None
    fraud_ratio = cfg.get("fraud_ratio") if cond == "C4" else None
    # n_estimators is an FTM-only factor; GBDTs ignore it (normalised to None so
    # it stays out of their run_id). FTMs default to 8 (the package default).
    if cfg["model"] in FTM_MODELS:
        n_estimators = int(cfg.get("n_estimators") or _N_ESTIMATORS_DEFAULT)
    else:
        n_estimators = None
    return {
        "dataset":          cfg["dataset"],
        "model":            cfg["model"],
        "condition":        cond,
        "seed":             int(cfg["seed"]),
        "batch_size":       batch_size,
        "k":                k,
        "fraud_ratio":      fraud_ratio,
        "metric":           metric,
        "n_estimators":     n_estimators,
        "pipeline_version": _PIPELINE_VERSION,
    }


def _lib_versions() -> dict[str, str | None]:
    """Record FTM library versions for provenance (not part of the config hash)."""
    from importlib.metadata import PackageNotFoundError, version

    out: dict[str, str | None] = {}
    for pkg in ("tabpfn", "tabicl"):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = None
    return out


# Resolved once at import — library versions are provenance, not hash inputs.
_LIB_VERSIONS = _lib_versions()


def config_hash(cfg: dict) -> str:
    """Return first 12 hex chars of SHA-256 of the canonically serialised config."""
    canonical = json.dumps(_canonical_config(cfg), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_config(cfg: dict) -> None:
    """Raise on invalid config combinations."""
    ds = cfg["dataset"]
    model = cfg["model"]
    cond = cfg["condition"]

    if ds not in DATASETS:
        raise ValueError(f"Unknown dataset: {ds!r}. Choose from {list(DATASETS)}")
    if model not in GBDT_KEYS and model not in FTM_MODELS:
        raise ValueError(f"Unknown model: {model!r}. Choose from {GBDT_KEYS + FTM_MODELS}")
    if cond not in ("C1", "C2", "C3", "C4"):
        raise ValueError(f"Unknown condition: {cond!r}. Choose from C1/C2/C3/C4")

    # GBDTs always train on full data — only C1 is meaningful
    if model in GBDT_KEYS and cond != "C1":
        raise ValueError(
            f"GBDT model {model!r} only supports condition C1 "
            f"(trains on full training set). Got {cond!r}."
        )

    # C3/C4 are FTM-only retrieval conditions (GBDTs already rejected above),
    # restricted to the two grid FTMs (design decision, Phase 3).
    if cond in ("C3", "C4") and model not in _RAP_FTM_MODELS:
        raise ValueError(
            f"Conditions C3/C4 run only on {_RAP_FTM_MODELS} (grid design). "
            f"Got model {model!r}."
        )

    # C4 needs a target fraud ratio; any other condition must not have one —
    # canonicalisation would silently null it, hiding a likely caller mistake.
    fr = cfg.get("fraud_ratio")
    if cond == "C4":
        if fr is None:
            raise ValueError("Condition C4 requires fraud_ratio (e.g. 0.05..0.5).")
        if not (0.0 < float(fr) < 1.0):
            raise ValueError(f"C4 fraud_ratio must be in (0, 1), got {fr!r}.")
    elif fr is not None:
        raise ValueError(
            f"fraud_ratio is a C4-only factor; got fraud_ratio={fr!r} with {cond}."
        )


# ---------------------------------------------------------------------------
# Context construction (C1 and C2 — Phase 2)
# ---------------------------------------------------------------------------

def _build_context_c1(
    X_train: np.ndarray,
    y_train: np.ndarray,
    context_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """C1: random subsample to context_size (natural fraud ratio preserved)."""
    n = len(y_train)
    if n <= context_size:
        return X_train, y_train
    idx = rng.choice(n, size=context_size, replace=False)
    return X_train[idx], y_train[idx]


def _build_context_c2(
    X_train: np.ndarray,
    y_train: np.ndarray,
    context_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """C2: stratified 50/50 up to context_size. Falls back if fraud is scarce."""
    fraud_idx = np.where(y_train == 1)[0]
    legit_idx = np.where(y_train == 0)[0]

    n_fraud_target = context_size // 2
    n_legit_target = context_size - n_fraud_target

    # If fewer fraud than target, take all fraud + fill remaining with legit
    if len(fraud_idx) <= n_fraud_target:
        chosen_fraud = fraud_idx
        n_legit = min(context_size - len(chosen_fraud), len(legit_idx))
        chosen_legit = rng.choice(legit_idx, size=n_legit, replace=False)
    else:
        chosen_fraud = rng.choice(fraud_idx, size=n_fraud_target, replace=False)
        chosen_legit = rng.choice(legit_idx, size=n_legit_target, replace=False)

    idx = np.concatenate([chosen_fraud, chosen_legit])
    rng.shuffle(idx)
    return X_train[idx], y_train[idx]


def _build_context(
    condition: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    context_size: int,
    rng: np.random.Generator,
    **kwargs,  # future: k, fraud_ratio, retriever, etc. for C3/C4
) -> tuple[np.ndarray, np.ndarray]:
    if condition == "C1":
        return _build_context_c1(X_train, y_train, context_size, rng)
    elif condition == "C2":
        return _build_context_c2(X_train, y_train, context_size, rng)
    elif condition in ("C3", "C4"):
        raise NotImplementedError(
            f"Condition {condition} requires RAP modules (Phase 3). "
            "Implement src/rap/ first."
        )
    else:
        raise ValueError(f"Unknown condition: {condition!r}")


# ---------------------------------------------------------------------------
# Batched prediction (avoids GPU OOM on large test sets)
# ---------------------------------------------------------------------------

def _predict_proba_batched(model, X: np.ndarray, batch_size: int = 512) -> np.ndarray:
    """Run predict_proba in chunks to avoid GPU OOM on large test sets."""
    parts = []
    n_batches = (len(X) + batch_size - 1) // batch_size
    t0 = time.perf_counter()
    for i, start in enumerate(range(0, len(X), batch_size)):
        chunk = X[start : start + batch_size]
        parts.append(model.predict_proba(chunk)[:, 1])
        if (i + 1) % 10 == 0 or (i + 1) == n_batches:
            elapsed = time.perf_counter() - t0
            rate = (i + 1) / elapsed
            eta = (n_batches - i - 1) / rate if rate > 0 else 0
            print(f"  [predict] batch {i+1}/{n_batches} | {elapsed:.0f}s elapsed | ETA {eta:.0f}s", flush=True)
    return np.concatenate(parts)


# ---------------------------------------------------------------------------
# Main run function
# ---------------------------------------------------------------------------

def run(cfg: dict, data: tuple | None = None) -> Path:
    """
    Execute one experiment run defined by cfg.

    If ``data`` is given as ``(X_train, y_train, X_test, y_test)`` the run uses it
    instead of loading from DuckDB — the grid driver loads each (dataset, seed)
    split once and reuses it across models/conditions. ``data`` must be the FULL
    seeded split for this dataset+seed (the same arrays ``run`` would have loaded);
    test subsampling is applied here, deterministically from the seed.

    Returns the path to the written parquet file, or the existing path if
    the run was already complete (idempotent).
    """
    # Validate the RAW cfg first: canonicalisation nulls stray factors (e.g. a
    # fraud_ratio passed with C3), so validating after it would let those slip
    # through silently — the exact mistake the fraud_ratio guard is meant to catch.
    _validate_config(cfg)
    cfg = _canonical_config(cfg)
    run_id = config_hash(cfg)
    out_path = RESULTS_DIR / f"{run_id}.parquet"

    if out_path.exists():
        print(f"[runner] SKIP {run_id} — already exists ({out_path})")
        return out_path

    print(f"[runner] START {run_id} | {cfg['dataset']} × {cfg['model']} × {cfg['condition']} × seed={cfg['seed']}")

    # -----------------------------------------------------------------------
    # 1. Load data
    # -----------------------------------------------------------------------
    rng = np.random.default_rng(cfg["seed"])
    dataset_id = cfg["dataset"]
    schema = DATASETS[dataset_id]

    if data is not None:
        X_train, y_train, X_test, y_test = data
    elif schema.get("fixed_split"):
        X_train, y_train, X_test, y_test, _ = load_dataset_split(dataset_id)
    else:
        X, y, feature_names = load_dataset(dataset_id)
        # Grouped split: assign each entity (e.g. banksim customer) wholly to
        # train or test so encoded entity IDs cannot leak labels across the split.
        group_col = schema.get("group_split")
        groups = X[:, feature_names.index(group_col)] if group_col else None
        X_train, X_test, y_train, y_test = split(X, y, seed=cfg["seed"], groups=groups)

    # split_info reflects the FULL test split — its counts and the true prevalence
    # pi are recorded for the prevalence correction even when test is subsampled.
    info = split_info(y_train, y_test)

    # Test-set subsampling (v4): keep ALL positives, downsample negatives to the
    # per-dataset cap (schema test_neg_cap; None => score full test set, e.g. eu_cc).
    # Cuts predict cost on large datasets; PR-AUC is recomputed at the true
    # prevalence below so it stays unbiased. Recall@FPR / ROC-AUC are invariant.
    neg_cap = schema.get("test_neg_cap")
    prevalence = info["n_fraud_test"] / info["n_test"] if info["n_test"] else float("nan")
    X_test_scored, y_test_scored = subsample_test(
        X_test, y_test, seed=cfg["seed"], max_negatives=neg_cap
    )
    subsampled = len(y_test_scored) < info["n_test"]
    test_subsample_rule = f"neg_cap_{neg_cap}" if subsampled else "full"
    test_subsample_seed = (cfg["seed"] ^ 0x5CA1E) if subsampled else None

    # -----------------------------------------------------------------------
    # 2. Build model + run inference (timed separately from data loading)
    # -----------------------------------------------------------------------
    model_key = cfg["model"]
    condition = cfg["condition"]

    # C3/C4-only outputs (NULL for GBDT and C1/C2). Populated by the per-group path.
    n_groups: int | None = None
    group_silhouette: float | None = None

    # Timing is split into three phases so one-time setup cost (model build,
    # FTM weight loading/download) never inflates fit/predict comparisons:
    #   setup_s   — model construction + weight loading
    #   fit_s     — GBDT training / FTM context fit
    #   predict_s — prediction over the full test set
    if model_key in GBDT_KEYS:
        # GBDTs always train on the FULL training set (no context limit)
        t0 = time.perf_counter()
        model = build_gbdt(model_key, seed=cfg["seed"])
        setup_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        fit_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        probs = model.predict_proba(X_test_scored)[:, 1]
        predict_s = time.perf_counter() - t0

        context_size = len(y_train)
        context_fraud_n = int(y_train.sum())
        context_fraud_unique = context_fraud_n  # GBDT: no resampling

    elif model_key in FTM_MODELS and condition in ("C1", "C2"):
        context_limit = CONTEXT_LIMITS[model_key]
        context_size = min(len(y_train), int(context_limit * _CONTEXT_FRACTION))

        # C1/C2: build context once, fit once, predict all test rows
        X_ctx, y_ctx = _build_context(condition, X_train, y_train, context_size, rng)
        context_fraud_n = int(y_ctx.sum())
        # C1/C2 sample fraud without replacement, so unique == n. C4
        # oversamples with replacement and recomputes this from the indices.
        context_fraud_unique = context_fraud_n

        t0 = time.perf_counter()
        model = build_ftm(
            model_key, device="cuda", seed=cfg["seed"], n_estimators=cfg["n_estimators"]
        )
        model.ensure_loaded()
        setup_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        model.fit(X_ctx, y_ctx)
        fit_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        # The chunk size here is a GPU memory chunk, not the experimental
        # batch_size factor (which only applies to C3/C4 retrieval batches).
        probs = _predict_proba_batched(
            model, X_test_scored, batch_size=_PREDICT_CHUNK.get(model_key, 512)
        )
        predict_s = time.perf_counter() - t0

    elif model_key in FTM_MODELS:  # C3 / C4 — RAP per-group retrieval path (Phase 3)
        from src.rap.context import RetrievalContextBuilder

        context_limit = CONTEXT_LIMITS[model_key]
        context_size = min(len(y_train), int(context_limit * _CONTEXT_FRACTION))
        predict_chunk = _PREDICT_CHUNK.get(model_key, 512)

        # setup_s = model build/load + scaler + retrieval indices + test grouping
        # (+ per-group kNN query/sampling time, accumulated below).
        t0 = time.perf_counter()
        model = build_ftm(
            model_key, device="cuda", seed=cfg["seed"], n_estimators=cfg["n_estimators"]
        )
        model.ensure_loaded()
        builder = RetrievalContextBuilder(
            X_train, y_train, metric=cfg["metric"], seed=cfg["seed"]
        )
        X_test_scaled = builder.scale(X_test_scored)
        labels, centres, n_groups, group_silhouette = builder.group_test(X_test_scaled)
        setup_s = time.perf_counter() - t0

        present = np.unique(labels)
        print(
            f"[runner] C3/C4 grouping: G={n_groups} "
            f"(silhouette={group_silhouette:.4f}), {len(present)} non-empty groups",
            flush=True,
        )

        # One retrieval + one FTM refit per group; predict that group's rows.
        probs = np.empty(len(y_test_scored), dtype=np.float64)
        fit_s = 0.0
        predict_s = 0.0
        retrieval_s = 0.0
        fraud_n_acc: list[int] = []
        fraud_unique_acc: list[int] = []
        guard_fires = 0
        for gi, g in enumerate(present):
            mask = labels == g
            # Per-group seeded RNG (order-independent) for C4 shuffle / oversampling.
            rng_g = np.random.default_rng([cfg["seed"], int(g)])
            t0 = time.perf_counter()
            X_ctx, y_ctx, stats = builder.build(
                condition, centres[g], context_size,
                fraud_ratio=cfg["fraud_ratio"], rng=rng_g,
            )
            retrieval_s += time.perf_counter() - t0
            fraud_n_acc.append(stats["fraud_n"])
            fraud_unique_acc.append(stats["fraud_unique"])
            guard_fires += int(stats["guard_fired"])

            t0 = time.perf_counter()
            model.fit(X_ctx, y_ctx)
            fit_s += time.perf_counter() - t0

            t0 = time.perf_counter()
            probs[mask] = _predict_proba_batched(
                model, X_test_scored[mask], batch_size=predict_chunk
            )
            predict_s += time.perf_counter() - t0
            print(
                f"  [group {gi+1}/{len(present)}] n={int(mask.sum())} "
                f"ctx_fraud={stats['fraud_n']} (unique={stats['fraud_unique']})",
                flush=True,
            )

        # Per-group kNN query + sampling time belongs to setup_s (documented as
        # "scaler + retrieval index + test grouping" for C3/C4) — before 2026-07-06
        # it was timed into NO column, so parquets written earlier record setup_s
        # values that are lower bounds on RAP retrieval overhead (review A-7).
        setup_s += retrieval_s

        # Per-group context fraud varies (C3) or is constant (C4); report the mean.
        context_fraud_n = int(round(float(np.mean(fraud_n_acc))))
        context_fraud_unique = int(round(float(np.mean(fraud_unique_acc))))
        if guard_fires:
            print(
                f"[runner] single-class guard fired in {guard_fires}/{len(present)} groups",
                flush=True,
            )

    else:
        raise ValueError(f"Unknown model key: {model_key!r}")

    inference_s = fit_s + predict_s

    # Sanity check: no NaN/Inf in predicted probabilities
    if not np.all(np.isfinite(probs)):
        n_bad = int(np.sum(~np.isfinite(probs)))
        raise RuntimeError(
            f"Model {model_key} produced {n_bad}/{len(probs)} non-finite probabilities "
            f"on {dataset_id} seed={cfg['seed']}."
        )

    # -----------------------------------------------------------------------
    # 3. Evaluate
    # -----------------------------------------------------------------------
    # When test is negative-subsampled, recompute PR-AUC at the true (full-split)
    # prevalence; otherwise standard AP on the full test set.
    metrics = compute_metrics(
        y_test_scored, probs, prevalence=prevalence if subsampled else None
    )

    # -----------------------------------------------------------------------
    # 4. Write parquet
    # -----------------------------------------------------------------------
    row = {
        "run_id":           run_id,
        "dataset":          cfg["dataset"],
        "model":            cfg["model"],
        "condition":        cfg["condition"],
        "seed":             cfg["seed"],
        "batch_size":       cfg["batch_size"],
        "k":                cfg["k"],
        "fraud_ratio":      cfg["fraud_ratio"],
        "metric":           cfg["metric"],
        "n_estimators":     cfg["n_estimators"],
        "pipeline_version": _PIPELINE_VERSION,
        "pr_auc":           metrics["pr_auc"],
        "recall_at_1fpr":   metrics["recall_at_1fpr"],
        "recall_at_5fpr":   metrics["recall_at_5fpr"],
        "roc_auc":          metrics["roc_auc"],
        "f1":               metrics["f1"],
        "n_train":          info["n_train"],
        "n_test":           info["n_test"],          # FULL test split
        "n_fraud_train":    info["n_fraud_train"],
        "n_fraud_test":     info["n_fraud_test"],    # FULL test split (true prevalence)
        "n_test_scored":      int(len(y_test_scored)),
        "n_fraud_test_scored": int(y_test_scored.sum()),
        "test_subsample_rule": test_subsample_rule,
        "test_subsample_seed": test_subsample_seed,
        "context_size":     context_size,
        "context_fraud_n":  context_fraud_n,
        "context_fraud_unique": context_fraud_unique,
        "n_groups":         n_groups,          # C3/C4 only (else None)
        "group_silhouette": group_silhouette,  # C3/C4 only (else None)
        "tabpfn_lib":       _LIB_VERSIONS["tabpfn"],
        "tabicl_lib":       _LIB_VERSIONS["tabicl"],
        "setup_s":          setup_s,
        "fit_s":            fit_s,
        "predict_s":        predict_s,
        "inference_s":      inference_s,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # Coerce the C3/C4-only columns to explicit nullable dtypes so a None row
    # (GBDT/C1/C2) still serialises as int64/double rather than Arrow `null`.
    # Otherwise an all-None single-row column becomes type `null`, which fails to
    # unify with the int64/double C3/C4 rows under a naive arrow::open_dataset.
    df = pd.DataFrame([row]).astype({"n_groups": "Int64", "group_silhouette": "float64"})
    df.to_parquet(out_path, index=False)

    print(
        f"[runner] DONE  {run_id} | "
        f"pr_auc={metrics['pr_auc']:.4f} recall@1fpr={metrics['recall_at_1fpr']:.4f} "
        f"({inference_s:.1f}s) → {out_path}"
    )
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    import argparse

    p = argparse.ArgumentParser(description="Run one experiment.")
    p.add_argument("--dataset",     required=True)
    p.add_argument("--model",       required=True)
    p.add_argument("--condition",   default="C1")
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--batch_size",  type=int, default=128)
    p.add_argument("--k",           type=int, default=None)
    p.add_argument("--fraud_ratio", type=float, default=None)
    p.add_argument("--metric",      default=None)
    p.add_argument("--n_estimators", type=int, default=None,
                   help="FTM ensemble passes (default: package default 8). For the n=8 vs n=4 A/B.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(vars(args))
