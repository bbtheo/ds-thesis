"""
Retrieval context builder for the RAP conditions (C3/C4).

``RetrievalContextBuilder`` owns everything that is built **once per split** and
reused across test groups:

* the ``StandardScaler`` (fit on ``X_train``) used for retrieval/clustering only —
  the context rows handed to the FTM stay **unscaled** (loader contract);
* three :class:`KNNRetriever` indices over the scaled train pool — the full pool
  (C3 mixed retrieval) and the fraud / legit subsets (C4 class-conditional
  retrieval, and the C3 single-class guard);
* the k-means + silhouette **grouping** of the scored test set.

Per-group design
----------------
Rather than retrieve+refit per test batch, the scored test set is clustered into
``G`` homogeneous groups and the FTM is refit **once per group** around that
group's cluster centre. ``G`` is chosen automatically (silhouette over
``k = 2..Kmax``) and recorded as an *output*, not a config input.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from src.rap.retriever import KNNRetriever
from src.rap.sampler import sample_c3, sample_c4

# Salt mixed into the seed for the silhouette subsample RNG, kept distinct from
# the train-context and test-subsample streams so they cannot correlate.
_SIL_SEED_SALT = 0x511C0


class RetrievalContextBuilder:
    """Owns the scaler, retrievers, and test grouping for one (dataset, seed) split."""

    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        metric: str = "cosine",
        seed: int = 42,
        kmax: int = 20,
        sil_subsample: int = 5000,
    ) -> None:
        self.X_train = X_train
        self.y_train = y_train
        self.metric = metric
        self.seed = int(seed)
        self.kmax = int(kmax)
        self.sil_subsample = int(sil_subsample)

        self.fraud_idx = np.where(y_train == 1)[0]
        self.legit_idx = np.where(y_train == 0)[0]
        if len(self.fraud_idx) == 0:
            raise ValueError("RAP retrieval needs >=1 fraud row in the training pool.")
        if len(self.legit_idx) == 0:
            raise ValueError("RAP retrieval needs >=1 legit row in the training pool.")

        # Scaler is fit on train only; retrieval/clustering happen in scaled space.
        self.scaler = StandardScaler().fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)

        self.retr_all = KNNRetriever(metric).fit(X_train_scaled)
        self.retr_fraud = KNNRetriever(metric).fit(X_train_scaled[self.fraud_idx])
        self.retr_legit = KNNRetriever(metric).fit(X_train_scaled[self.legit_idx])

    # ------------------------------------------------------------------
    # Scaling
    # ------------------------------------------------------------------
    def scale(self, X: np.ndarray) -> np.ndarray:
        """Apply the train-fitted scaler (e.g. to the scored test set before grouping)."""
        return self.scaler.transform(X)

    # ------------------------------------------------------------------
    # Grouping: k-means + silhouette over the scored test set
    # ------------------------------------------------------------------
    def group_test(
        self, X_test_scaled: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, int, float]:
        """Cluster the scored test set and pick ``G`` by silhouette.

        Returns ``(labels, centres, G, silhouette)`` where ``labels`` assigns each
        scored-test row to a cluster, ``centres`` are the ``G`` cluster centres in
        scaled space (the retrieval centroids), and ``silhouette`` is the selection
        score of the chosen ``k``.

        The ``k``-selection sweep (``k = 2..Kmax``) runs on a **seeded subsample**
        (silhouette is O(n^2), and a full-set KMeans per ``k`` is the dominant cost
        on ~30k-row scored test sets); the final grouping is a single KMeans refit
        at ``best_k`` on the **full** scored test set. Falls back to one group at the
        global centroid when the test set is too small or no ``k`` yields >=2
        populated clusters.
        """
        n = len(X_test_scaled)
        if n < 3:
            return self._single_group(X_test_scaled)

        rng = np.random.default_rng([self.seed, _SIL_SEED_SALT])
        if n > self.sil_subsample:
            sub_idx = rng.choice(n, size=self.sil_subsample, replace=False)
        else:
            sub_idx = np.arange(n)
        X_sub = X_test_scaled[sub_idx]

        # Pin BLAS/OpenMP to one thread for the whole grouping block: with a fixed
        # random_state the KMeans *partition* can still differ across thread counts
        # (multithreaded float reductions), and a silhouette near-tie across k can
        # flip the same way. Grouping is a small fraction of a C3/C4 run, so the
        # single-threaded cost is negligible against thread-count-invariance.
        with threadpool_limits(limits=1):
            # k must be < #points used to fit and < n for the final refit.
            kmax = min(self.kmax, len(X_sub) - 1, n - 1)
            best_k, best_sil = None, -np.inf
            for k in range(2, kmax + 1):
                sub_labels = KMeans(n_clusters=k, random_state=self.seed, n_init=10).fit_predict(X_sub)
                if len(np.unique(sub_labels)) < 2:
                    continue  # subsample collapsed to one cluster — silhouette undefined
                sil = float(silhouette_score(X_sub, sub_labels))
                if sil > best_sil:
                    best_k, best_sil = k, sil

            if best_k is None:
                return self._single_group(X_test_scaled)

            # Final grouping: one KMeans refit at best_k over the full scored test set.
            km = KMeans(n_clusters=best_k, random_state=self.seed, n_init=10).fit(X_test_scaled)
        labels, centres = self._canonicalise(km.labels_, X_test_scaled)
        return labels, centres, int(len(centres)), float(best_sil)

    @staticmethod
    def _canonicalise(
        raw_labels: np.ndarray, X_test_scaled: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Make the grouping fully reproducible (IDs *and* centres).

        Two run-to-run nondeterminisms in KMeans must be removed, because the
        runner keys each group's RNG on its cluster ID and retrieves on its centre
        — so any drift breaks C4 idempotency:

        1. *Cluster IDs* are permuted by multithreaded float reductions even with a
           fixed ``random_state``. Reorder clusters by their smallest member row
           index (a function of the reproducible partition alone) so IDs are stable.
        2. *``cluster_centers_``* carry ~1e-15 float noise that flips the occasional
           near-tie kNN retrieval. A converged KMeans centre *is* the mean of its
           members, so recompute each centre as the exact (single-threaded,
           deterministic) ``np.mean`` of its assigned rows.
        """
        order = sorted(
            np.unique(raw_labels),
            key=lambda g: int(np.flatnonzero(raw_labels == g).min()),
        )
        remap = np.empty(int(raw_labels.max()) + 1, dtype=np.int64)
        for new_id, old_id in enumerate(order):
            remap[old_id] = new_id
        labels = remap[raw_labels]
        centres = np.vstack([X_test_scaled[labels == g].mean(axis=0) for g in range(len(order))])
        return labels, centres

    @staticmethod
    def _single_group(
        X_test_scaled: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, int, float]:
        labels = np.zeros(len(X_test_scaled), dtype=np.int64)
        centres = X_test_scaled.mean(axis=0, keepdims=True)
        return labels, centres, 1, float("nan")

    # ------------------------------------------------------------------
    # Context construction (dispatch on condition)
    # ------------------------------------------------------------------
    def build(
        self,
        condition: str,
        centre_scaled: np.ndarray,
        context_size: int,
        fraud_ratio: float | None = None,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """Build one group's context. Returns ``(X_ctx, y_ctx, stats)`` — unscaled."""
        if condition == "C3":
            idx, stats = sample_c3(
                self.retr_all, self.retr_fraud, self.retr_legit,
                self.y_train, self.fraud_idx, self.legit_idx,
                centre_scaled, context_size,
            )
        elif condition == "C4":
            if fraud_ratio is None:
                raise ValueError("C4 requires fraud_ratio.")
            if rng is None:
                raise ValueError("C4 requires a seeded numpy Generator (rng).")
            idx, stats = sample_c4(
                self.retr_fraud, self.retr_legit,
                self.fraud_idx, self.legit_idx,
                centre_scaled, context_size, float(fraud_ratio), rng,
            )
        else:
            raise ValueError(
                f"RetrievalContextBuilder handles only C3/C4, got {condition!r}."
            )

        # Context rows handed to the FTM are UNSCALED (loader contract).
        X_ctx = self.X_train[idx]
        y_ctx = self.y_train[idx]
        return X_ctx, y_ctx, stats
