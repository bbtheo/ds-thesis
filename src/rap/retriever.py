"""
kNN retriever for the RAP pipeline (conditions C3/C4).

Thin wrapper over scikit-learn's brute-force :class:`NearestNeighbors`. The
index is built **once per split** on the *scaled* training features and queried
**once per test group** (``G <= Kmax`` queries in total), so brute force is not a
bottleneck even on the largest train pools (paysim's ~5M legit rows).

Metrics
-------
``"cosine"`` (default) or ``"euclidean"``. The choice is decided on the
``ai_banking`` dev set (see ``scripts/dev_rap_validation.py``); the metric is a
config factor for C3/C4 and is part of the run_id hash.

Contract
--------
``query(centre, k)`` returns the **local** indices (into the matrix passed to
``fit``) of the ``k`` nearest rows, nearest first. The caller maps local indices
to global train indices when the retriever was fit on a class subset.
"""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


class KNNRetriever:
    """Nearest-neighbour retriever over a fixed (scaled) feature matrix."""

    _VALID_METRICS = ("cosine", "euclidean")

    def __init__(self, metric: str = "cosine") -> None:
        if metric not in self._VALID_METRICS:
            raise ValueError(
                f"metric must be one of {self._VALID_METRICS}, got {metric!r}"
            )
        self.metric = metric
        self._nn: NearestNeighbors | None = None
        self._n = 0

    def fit(self, X_scaled: np.ndarray) -> "KNNRetriever":
        """Store the (scaled) reference matrix.

        Brute force keeps a reference to ``X_scaled`` and does no index build, so
        this is effectively O(1) — all the work happens at query time.
        """
        self._nn = NearestNeighbors(algorithm="brute", metric=self.metric)
        self._nn.fit(X_scaled)
        self._n = int(X_scaled.shape[0])
        return self

    @property
    def n(self) -> int:
        """Number of rows in the fitted reference matrix."""
        return self._n

    def query(self, centre: np.ndarray, k: int) -> np.ndarray:
        """Local indices of the ``k`` nearest rows to ``centre`` (nearest first).

        ``centre`` is a single point in scaled space, shape ``(d,)``. ``k`` is
        clipped to the number of stored rows, so an over-large request simply
        returns the whole pool in nearest-first order.
        """
        assert self._nn is not None, "Call fit() before query()"
        k = int(min(k, self._n))
        if k <= 0:
            return np.empty(0, dtype=np.int64)
        idx = self._nn.kneighbors(
            np.asarray(centre).reshape(1, -1),
            n_neighbors=k,
            return_distance=False,
        )[0]
        return idx.astype(np.int64, copy=False)
