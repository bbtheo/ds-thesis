"""
Tabular Foundation Model wrappers: TabPFN v2, v2.5, v2.6, v3, TabICLv2.

All wrappers expose:
    fit(X_ctx, y_ctx)                        — load context
    predict_proba(X_test) -> (n, 2) ndarray  — predict on test batch

Model weights are loaded lazily on the first fit() call and reused for
subsequent calls (critical for C3/C4 where fit() is called per batch).

Context limits (rows; CONTEXT_LIMITS is the *experimental context budget* the
runner subsamples to, NOT necessarily the model's own cap):
    tabpfn_v2  : 10,000 × 500    (= checkpoint MAX_NUMBER_OF_SAMPLES)
    tabpfn_25  : 50,000 × 2,000  (= checkpoint MAX_NUMBER_OF_SAMPLES)
    tabpfn_26  : 50,000 × 2,000  (checkpoint allows 100,000 — 50k is our budget)
    tabpfn_3   : 50,000 × 2,000  (checkpoint allows 1,000,000 — 50k is our budget)
    tabiclv2   : 48,000 × 500    (no hard cap; pre-training upper bound)

Checkpoint values verified 2026-07-06 from each model's
``inference_config_.MAX_NUMBER_OF_SAMPLES`` under tabpfn 8.0.8 (review C-3).
"""

from __future__ import annotations

import numpy as np

# Context limits per model key (rows; used by runner to subsample training context)
CONTEXT_LIMITS: dict[str, int] = {
    "tabpfn_v2": 10_000,
    "tabpfn_25": 50_000,
    "tabpfn_26": 50_000,
    "tabpfn_3":  50_000,
    "tabiclv2":  48_000,
}

FTM_MODELS = list(CONTEXT_LIMITS.keys())

_TABPFN_VERSION_MAP = {
    "tabpfn_v2": "v2",
    "tabpfn_25":  "2.5",
    "tabpfn_26":  "2.6",
    "tabpfn_3":   "3",
}


# ---------------------------------------------------------------------------
# TabPFN (v2, v2.5, v2.6, v3)
# ---------------------------------------------------------------------------

class TabPFNModel:
    """
    Wrapper for TabPFN v2 / v2.5 / v2.6 / v3 (package: `tabpfn`).

    Model weights are loaded on first fit() and reused on subsequent calls.
    """

    _VALID_VERSIONS = ("v2", "2.5", "2.6", "3")

    def __init__(
        self,
        version: str = "v2",
        device: str = "cuda",
        seed: int = 42,
        n_estimators: int | None = None,
        context_limit_override: int | None = None,
    ) -> None:
        if version not in self._VALID_VERSIONS:
            raise ValueError(f"version must be one of {self._VALID_VERSIONS}, got {version!r}")
        self.version = version
        self.device = device
        self.seed = seed
        # Number of ensemble passes. None => package default (8). Reducing it
        # speeds predict ~linearly but changes outputs (kept None/8 until validated).
        self.n_estimators = n_estimators
        # Raise the fit() guard above the CONTEXT_LIMITS design budget (large-context
        # C2 extension). Must stay within the checkpoint's own cap (v3: 1M; 2.5/2.6
        # checkpoints reject beyond 50k/100k regardless of this override).
        self.context_limit_override = context_limit_override
        self._model = None  # lazy-init on first fit()

    @property
    def model_key(self) -> str:
        return {v: k for k, v in _TABPFN_VERSION_MAP.items()}[self.version]

    @property
    def context_limit(self) -> int:
        if self.context_limit_override is not None:
            return self.context_limit_override
        return CONTEXT_LIMITS[self.model_key]

    def _ensure_model(self) -> None:
        """Load model weights once (lazy init)."""
        if self._model is not None:
            return
        from tabpfn import TabPFNClassifier
        from tabpfn.model_loading import ModelVersion

        mv_map = {
            "v2": ModelVersion.V2,
            "2.5": ModelVersion.V2_5,
            "2.6": ModelVersion.V2_6,
            "3": ModelVersion.V3,
        }
        kwargs = {}
        if self.n_estimators is not None:
            kwargs["n_estimators"] = self.n_estimators
        self._model = TabPFNClassifier.create_default_for_version(
            mv_map[self.version], device=self.device, random_state=self.seed, **kwargs
        )

    def ensure_loaded(self) -> None:
        """Load model weights now, so setup cost is not attributed to fit()."""
        self._ensure_model()

    def fit(self, X_ctx: np.ndarray, y_ctx: np.ndarray) -> "TabPFNModel":
        if X_ctx.shape[0] > self.context_limit:
            raise ValueError(
                f"Context size {X_ctx.shape[0]} exceeds limit {self.context_limit} "
                f"for {self.model_key}. Subsample before calling fit()."
            )
        self._ensure_model()
        self._model.fit(X_ctx, y_ctx)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        import torch

        assert self._model is not None, "Call fit() before predict_proba()"
        proba = self._model.predict_proba(X)
        if self.device == "cuda":
            torch.cuda.empty_cache()
        return np.asarray(proba)


# ---------------------------------------------------------------------------
# TabICLv2
# ---------------------------------------------------------------------------

class TabICLModel:
    """
    Wrapper for TabICLv2 (package: `tabicl`, BSD-3 licence).

    Model weights are loaded on first fit() and reused on subsequent calls.
    """

    def __init__(
        self,
        device: str = "cuda",
        seed: int = 42,
        n_estimators: int | None = None,
        context_limit_override: int | None = None,
        disk_offload_dir: str | None = None,
    ) -> None:
        self.device = device
        self.seed = seed
        # None => package default (8). See TabPFNModel.n_estimators note.
        self.n_estimators = n_estimators
        # See TabPFNModel.context_limit_override. For TabICL there is no hard
        # checkpoint cap, but 48k is the pre-training upper bound — beyond it the
        # model extrapolates.
        self.context_limit_override = context_limit_override
        # Memory management only (result-neutral): with a dir set, TabICL's
        # offload_mode='auto' memory-maps oversized encoder outputs to disk
        # instead of raising CUDA OOM. Needed for ~228k-row full-fold contexts
        # on 12GB VRAM (sota_cv study); None keeps the library default.
        self.disk_offload_dir = disk_offload_dir
        self._model = None  # lazy-init on first fit()

    @property
    def context_limit(self) -> int:
        if self.context_limit_override is not None:
            return self.context_limit_override
        return CONTEXT_LIMITS["tabiclv2"]

    def _ensure_model(self) -> None:
        """Load model weights once (lazy init)."""
        if self._model is not None:
            return
        from tabicl import TabICLClassifier

        kwargs = {}
        if self.n_estimators is not None:
            kwargs["n_estimators"] = self.n_estimators
        if self.disk_offload_dir is not None:
            kwargs["disk_offload_dir"] = self.disk_offload_dir
        self._model = TabICLClassifier(
            device=self.device,
            random_state=self.seed,
            verbose=False,
            **kwargs,
        )

    def ensure_loaded(self) -> None:
        """Load model weights now, so setup cost is not attributed to fit()."""
        self._ensure_model()

    def fit(self, X_ctx: np.ndarray, y_ctx: np.ndarray) -> "TabICLModel":
        if X_ctx.shape[0] > self.context_limit:
            raise ValueError(
                f"Context size {X_ctx.shape[0]} exceeds limit {self.context_limit} "
                f"for tabiclv2. Subsample before calling fit()."
            )
        self._ensure_model()
        self._model.fit(X_ctx, y_ctx)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        import torch

        assert self._model is not None, "Call fit() before predict_proba()"
        proba = self._model.predict_proba(X)
        if self.device == "cuda":
            torch.cuda.empty_cache()
        return np.asarray(proba)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_ftm(
    model_key: str,
    device: str = "cuda",
    seed: int = 42,
    n_estimators: int | None = None,
    context_limit_override: int | None = None,
    disk_offload_dir: str | None = None,
) -> TabPFNModel | TabICLModel:
    """
    Instantiate an FTM wrapper by key.

    Supported keys: "tabpfn_v2", "tabpfn_25", "tabpfn_26", "tabpfn_3", "tabiclv2"
    ``n_estimators`` is the number of ensemble passes (None => package default 8).
    ``context_limit_override`` raises the fit() guard above the CONTEXT_LIMITS
    design budget (large-context C2 extension only).
    """
    if model_key in _TABPFN_VERSION_MAP:
        return TabPFNModel(
            version=_TABPFN_VERSION_MAP[model_key], device=device, seed=seed,
            n_estimators=n_estimators, context_limit_override=context_limit_override,
        )
    elif model_key == "tabiclv2":
        return TabICLModel(
            device=device, seed=seed, n_estimators=n_estimators,
            context_limit_override=context_limit_override,
            disk_offload_dir=disk_offload_dir,
        )
    else:
        raise ValueError(
            f"Unknown FTM model: {model_key!r}. Supported: {FTM_MODELS}"
        )
