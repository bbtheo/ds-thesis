"""
GBDT model wrappers: XGBoost and CatBoost.

Both expose a unified sklearn-style interface:
    fit(X_train, y_train)
    predict_proba(X_test) -> ndarray of shape (n, 2)

scale_pos_weight is set automatically from the training labels.
Strong defaults with early stopping — GBDTs are reference baselines,
not the thesis focus, but should be credible comparators.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split

# Early stopping: hold out 10% of train for validation, patience 50 rounds
_VAL_FRACTION = 0.10
_EARLY_STOPPING_ROUNDS = 50


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------

class XGBoostModel:
    """XGBoost classifier with automatic class-weight balancing and early stopping."""

    def __init__(
        self,
        n_estimators: int = 2000,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        n_jobs: int = -1,
        random_state: int = 42,
        device: str = "cpu",
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.device = device
        self._model = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "XGBoostModel":
        from xgboost import XGBClassifier

        n_neg = int((y_train == 0).sum())
        n_pos = int((y_train == 1).sum())
        scale_pos_weight = n_neg / max(n_pos, 1)

        self._model = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            scale_pos_weight=scale_pos_weight,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            device=self.device,
            eval_metric="aucpr",
            early_stopping_rounds=_EARLY_STOPPING_ROUNDS,
            verbosity=0,
        )

        # Split off validation set for early stopping (stratified)
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train,
            test_size=_VAL_FRACTION,
            random_state=self.random_state,
            stratify=y_train,
        )
        self._model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self._model is not None, "Call fit() before predict_proba()"
        return self._model.predict_proba(X)


# ---------------------------------------------------------------------------
# CatBoost
# ---------------------------------------------------------------------------

class CatBoostModel:
    """CatBoost classifier with automatic class-weight balancing and early stopping."""

    def __init__(
        self,
        iterations: int = 2000,
        depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        random_seed: int = 42,
        thread_count: int = 1,
    ) -> None:
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.random_seed = random_seed
        # CatBoost 1.2.10 SIGSEGVs in its TBB thread pool during long fits
        # (2000 iters) on large training sets — reproducibly on paysim (~5M rows)
        # at the default thread count, killing the whole process. CatBoost CPU
        # training is thread-count-invariant (verified: tc=1 vs tc=24 give
        # bit-identical predictions), so thread_count=1 avoids the race with NO
        # change to results — it is just slower on the largest datasets.
        self.thread_count = thread_count
        self._model = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "CatBoostModel":
        from catboost import CatBoostClassifier

        n_neg = int((y_train == 0).sum())
        n_pos = int((y_train == 1).sum())
        scale_pos_weight = n_neg / max(n_pos, 1)

        self._model = CatBoostClassifier(
            iterations=self.iterations,
            depth=self.depth,
            learning_rate=self.learning_rate,
            scale_pos_weight=scale_pos_weight,
            subsample=self.subsample,
            random_seed=self.random_seed,
            eval_metric="PRAUC",
            early_stopping_rounds=_EARLY_STOPPING_ROUNDS,
            thread_count=self.thread_count,
            verbose=0,
        )

        # Split off validation set for early stopping (stratified)
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train,
            test_size=_VAL_FRACTION,
            random_state=self.random_seed,
            stratify=y_train,
        )
        self._model.fit(X_tr, y_tr, eval_set=(X_val, y_val))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self._model is not None, "Call fit() before predict_proba()"
        return self._model.predict_proba(X)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

GBDT_MODELS: dict[str, type] = {
    "xgboost": XGBoostModel,
    "catboost": CatBoostModel,
}


def build_gbdt(model_key: str, seed: int = 42) -> XGBoostModel | CatBoostModel:
    """Instantiate a GBDT by key."""
    if model_key not in GBDT_MODELS:
        raise ValueError(f"Unknown GBDT model: {model_key!r}. Choose from {list(GBDT_MODELS)}")
    cls = GBDT_MODELS[model_key]
    if model_key == "xgboost":
        return cls(random_state=seed)
    else:
        return cls(random_seed=seed)
