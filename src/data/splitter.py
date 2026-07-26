"""
Train/test splitter.

Produces a seeded 80/20 random split. For FiFAR, the pre-defined split
is used instead (via loader.load_dataset_split). For datasets with a
``group_split`` column in the schema (e.g. banksim's customer ID), a
grouped split assigns each entity wholly to train or test, preventing
entity-ID label leakage across the split.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, train_test_split


def split(
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
    test_size: float = 0.2,
    groups: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Seeded 80/20 train/test split.

    If ``groups`` is given, a GroupShuffleSplit assigns each group (entity)
    wholly to train or test — no entity straddles the split. Grouped splits
    are not label-stratified; fraud rates stay close to the dataset rate as
    long as there are many groups.

    Otherwise a stratified split is attempted; if the minority class is too
    small for stratification (< 2 samples per fold) it falls back to a
    random split.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    if groups is not None:
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, test_idx = next(gss.split(X, y, groups=groups))
        return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=seed,
            stratify=y,
        )
    except ValueError:
        # Fallback: unstratified split (minority class too rare)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=seed,
        )
    return X_train, X_test, y_train, y_test


def split_temporal(
    X: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    test_size: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Deterministic temporal split at a time-value boundary.

    The boundary is the smallest time value whose cumulative row count reaches
    ``1 - test_size`` of all rows; train = rows with ``t <= boundary``, test =
    the rest. Splitting at a value boundary rather than a row rank guarantees
    no timestamp straddles the split, so the realised train fraction can
    slightly exceed ``1 - test_size`` when many rows share the boundary value.
    No randomness — the split is identical for every seed. Not stratified and
    not grouped: entities may straddle the boundary, and the two sides may have
    different fraud rates (both are properties of the temporal protocol itself).

    Returns
    -------
    X_train, X_test, y_train, y_test, boundary
    """
    vals, counts = np.unique(t, return_counts=True)
    cum = np.cumsum(counts)
    boundary = vals[np.searchsorted(cum, (1 - test_size) * len(t))]
    train_mask = t <= boundary
    return X[train_mask], X[~train_mask], y[train_mask], y[~train_mask], float(boundary)


def subsample_test(
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    max_negatives: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Negative-subsampled test set (pipeline v4): keep ALL positives, randomly
    downsample negatives to ``max_negatives``.

    Predict cost is ~linear in test rows, which dominates FTM runtime; positives
    are few and drive metric variance, so they are always kept in full. Negatives
    are downsampled to cut cost. PR-AUC must be recomputed at the true prevalence
    (see :func:`src.eval.metrics.ap_at_prevalence`); Recall@FPR / ROC-AUC are
    invariant to random negative subsampling.

    Deterministic in ``seed`` (uses ``default_rng(seed ^ 0x5CA1E)`` so the test
    subsample is independent of the train-context RNG stream).

    Returns ``(X_test, y_test)`` unchanged when ``max_negatives`` is None or there
    are already fewer negatives than the cap.
    """
    if max_negatives is None:
        return X_test, y_test
    pos_idx = np.where(y_test == 1)[0]
    neg_idx = np.where(y_test == 0)[0]
    if len(neg_idx) <= max_negatives:
        return X_test, y_test
    rng = np.random.default_rng(seed ^ 0x5CA1E)
    keep_neg = rng.choice(neg_idx, size=max_negatives, replace=False)
    idx = np.concatenate([pos_idx, keep_neg])
    rng.shuffle(idx)
    return X_test[idx], y_test[idx]


def split_info(y_train: np.ndarray, y_test: np.ndarray) -> dict:
    """Return a dict with row counts and fraud rates for logging."""
    return {
        "n_train":       len(y_train),
        "n_test":        len(y_test),
        "n_fraud_train": int(y_train.sum()),
        "n_fraud_test":  int(y_test.sum()),
        "fraud_pct_train": float(y_train.mean() * 100),
        "fraud_pct_test":  float(y_test.mean() * 100),
    }
