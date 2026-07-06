"""Regression tests for src/eval/metrics.py.

Focus: ap_at_prevalence must group tied scores into one threshold (C-1 in the
2026-07-06 review) — matching sklearn.average_precision_score at the observed
prevalence and independent of the incidental order of tied rows.
"""

import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from src.eval.metrics import ap_at_prevalence, recall_at_fpr


def _prevalence(y):
    return float(np.mean(y))


class TestApAtPrevalenceTies:
    def test_matches_sklearn_on_tie_free_scores(self):
        rng = np.random.default_rng(0)
        y = (rng.random(2000) < 0.05).astype(int)
        s = rng.random(2000)  # continuous, no ties
        assert ap_at_prevalence(y, s, _prevalence(y)) == pytest.approx(
            average_precision_score(y, s), abs=1e-12
        )

    def test_matches_sklearn_with_heavy_ties(self):
        rng = np.random.default_rng(1)
        y = (rng.random(2000) < 0.05).astype(int)
        # Quantized scores: ~20 distinct values, many cross-class ties.
        s = np.round(rng.random(2000), 1)
        assert ap_at_prevalence(y, s, _prevalence(y)) == pytest.approx(
            average_precision_score(y, s), abs=1e-12
        )

    def test_order_independent_under_ties(self):
        rng = np.random.default_rng(2)
        y = (rng.random(500) < 0.1).astype(int)
        s = np.round(rng.random(500), 1)
        base = ap_at_prevalence(y, s, 0.01)
        for seed in range(5):
            perm = np.random.default_rng(seed).permutation(len(y))
            assert ap_at_prevalence(y[perm], s[perm], 0.01) == pytest.approx(
                base, abs=1e-12
            )

    def test_all_scores_identical(self):
        # One threshold: recall jumps 0 -> 1, precision = prevalence.
        y = np.array([0, 0, 0, 1])
        s = np.full(4, 0.5)
        assert ap_at_prevalence(y, s, 0.25) == pytest.approx(0.25, abs=1e-12)
        assert ap_at_prevalence(y, s, 0.25) == pytest.approx(
            average_precision_score(y, s), abs=1e-12
        )

    def test_prevalence_correction_direction(self):
        # Lowering the prevalence must lower PR-AUC (precision shrinks).
        rng = np.random.default_rng(3)
        y = (rng.random(1000) < 0.2).astype(int)
        s = rng.random(1000) + 0.5 * y  # informative scores
        assert ap_at_prevalence(y, s, 0.01) < ap_at_prevalence(y, s, _prevalence(y))

    def test_degenerate_labels_return_nan(self):
        s = np.random.default_rng(4).random(10)
        assert np.isnan(ap_at_prevalence(np.zeros(10, dtype=int), s, 0.01))
        assert np.isnan(ap_at_prevalence(np.ones(10, dtype=int), s, 0.01))


class TestRecallAtFpr:
    def test_conservative_step_no_interpolation(self):
        # 1 fraud ranked above all legits: TPR=1 at FPR=0.
        y = np.array([1, 0, 0, 0])
        s = np.array([0.9, 0.5, 0.4, 0.3])
        assert recall_at_fpr(y, s, 0.01) == 1.0

    def test_no_operating_point_below_budget(self):
        # Fraud ranked below a legit: reaching it costs FPR 1/2 > 1%.
        y = np.array([0, 1, 0])
        s = np.array([0.9, 0.5, 0.4])
        assert recall_at_fpr(y, s, 0.01) == 0.0
