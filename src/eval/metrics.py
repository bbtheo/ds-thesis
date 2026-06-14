"""
Evaluation metrics for binary fraud detection.

All functions accept:
    y_true  : int array of shape (n,), values in {0, 1}
    y_score : float array of shape (n,), predicted positive-class probabilities

Returns a dict with all metrics (or individual float for single-metric helpers).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)


def recall_at_fpr(y_true: np.ndarray, y_score: np.ndarray, target_fpr: float) -> float:
    """
    Return recall (TPR) at a given FPR budget.

    Uses the conservative step-function value of the empirical ROC curve:
    the highest TPR among operating points with FPR <= target_fpr.

    Linear interpolation between ROC points (e.g. ``np.interp``) is
    deliberately NOT used: interpolated (FPR, TPR) pairs do not correspond
    to any achievable decision threshold, so interpolation overestimates
    recall — noticeably when the positive class is small and the empirical
    curve is coarse (e.g. eu_cc with ~100 test-set frauds).

    Parameters
    ----------
    y_true : array of 0/1 labels
    y_score : predicted positive-class probabilities
    target_fpr : e.g. 0.01 for 1% FPR, 0.05 for 5% FPR

    Returns
    -------
    Recall at the largest achievable operating point with FPR <= target_fpr,
    in [0, 1].
    """
    if y_true.sum() == 0:
        return float("nan")

    fpr, tpr, _ = roc_curve(y_true, y_score)
    # roc_curve always starts at (fpr=0, tpr=0), so the mask is never empty.
    return float(tpr[fpr <= target_fpr].max())


def ap_at_prevalence(y_true: np.ndarray, y_score: np.ndarray, prevalence: float) -> float:
    """
    Average precision (PR-AUC) corrected to a target prevalence.

    When the test set is subsampled by keeping ALL positives and downsampling
    negatives (pipeline v4), empirical precision is inflated because the
    positive:negative ratio no longer matches the population. The ROC counts
    (TPR, FPR), however, are invariant to random negative subsampling, so
    precision can be recomputed at the true population prevalence ``pi``:

        precision(t) = pi * TPR(t) / (pi * TPR(t) + (1 - pi) * FPR(t))

    and integrated as AP = sum_k (recall_k - recall_{k-1}) * precision_k over the
    score-sorted thresholds. At ``pi`` equal to the observed prevalence this equals
    ``sklearn.metrics.average_precision_score`` to machine precision (scores are
    continuous so ties are negligible).
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)
    order = np.argsort(-y_score, kind="mergesort")
    y_sorted = y_true[order]
    P = int(y_sorted.sum())
    N = len(y_sorted) - P
    if P == 0 or N == 0:
        return float("nan")

    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    recall = tp / P
    fpr = fp / N
    pi = float(prevalence)
    denom = pi * recall + (1.0 - pi) * fpr
    precision = np.where(denom > 0, pi * recall / denom, 1.0)
    d_recall = np.diff(recall, prepend=0.0)
    return float(np.sum(d_recall * precision))


def compute_metrics(
    y_true: np.ndarray, y_score: np.ndarray, prevalence: float | None = None
) -> dict[str, float]:
    """
    Compute all evaluation metrics for one run.

    Parameters
    ----------
    y_true : int array of shape (n,), values in {0, 1}
    y_score : float array of shape (n,), predicted positive-class probabilities
    prevalence : float | None
        True (full-split) fraud rate. Pass this ONLY when ``y_true``/``y_score``
        come from a negative-subsampled test set (pipeline v4): PR-AUC is then
        computed via :func:`ap_at_prevalence` to undo the subsampling bias.
        When ``None`` (full test set) the standard ``average_precision_score`` is
        used. Recall@FPR and ROC-AUC are invariant to negative subsampling.

    Returns
    -------
    dict with keys: pr_auc, recall_at_1fpr, recall_at_5fpr, roc_auc, f1
    """
    y_true = np.asarray(y_true, dtype=np.int32)
    y_score = np.asarray(y_score, dtype=np.float64)

    # Guard: if all labels are the same, most metrics are undefined
    if len(np.unique(y_true)) < 2:
        nan = float("nan")
        return {
            "pr_auc": nan,
            "recall_at_1fpr": nan,
            "recall_at_5fpr": nan,
            "roc_auc": nan,
            "f1": nan,
        }

    if prevalence is None:
        pr_auc = float(average_precision_score(y_true, y_score))
    else:
        pr_auc = ap_at_prevalence(y_true, y_score, prevalence)
    roc_auc = float(roc_auc_score(y_true, y_score))
    recall_1fpr = recall_at_fpr(y_true, y_score, 0.01)
    recall_5fpr = recall_at_fpr(y_true, y_score, 0.05)

    y_pred = (y_score >= 0.5).astype(np.int32)
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    return {
        "pr_auc": pr_auc,
        "recall_at_1fpr": recall_1fpr,
        "recall_at_5fpr": recall_5fpr,
        "roc_auc": roc_auc,
        "f1": f1,
    }
