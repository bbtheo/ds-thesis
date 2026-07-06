"""
Shared split-loading / test-subsampling helpers for the grid and ablation study
scripts under ``scripts/``.

Factored out of ``run_pending_c1c2.py``, ``run_pending_c3c4.py``, ``run_c2_grid.py``,
``run_rap_designspace.py``, and ``run_relevance_ablation.py``, which each carried
their own copy of the same "load the (dataset, seed) split" block, and three of
which also carried the same "negative-subsample the test set + compute prevalence"
block (verified byte-identical across sites as of the 2026-07-06 review). This is a
pure refactor: same seeds, same call order, same RNG streams as every site it
replaces — see each script for the (now-removed) local copy it used to have.

Not used by ``src/experiment/runner.py`` itself (out of scope for this refactor;
the runner has its own internal loading path).
"""

from __future__ import annotations

import numpy as np

from src.data.loader import load_dataset, load_dataset_split
from src.data.schema import DATASETS
from src.data.splitter import split, split_info, subsample_test


def load_full_split(
    dataset_id: str, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the FULL (X_train, y_train, X_test, y_test) split for this dataset+seed.

    Mirrors ``runner.run()``'s own loading exactly, so preloaded data is identical
    to what the runner would have produced. No test-negative subsampling is applied
    here; callers that need it should use :func:`load_and_subsample_test`, or (as
    the runner does internally) call :func:`src.data.splitter.subsample_test`
    themselves.
    """
    schema = DATASETS[dataset_id]
    if schema.get("fixed_split"):
        X_train, y_train, X_test, y_test, _ = load_dataset_split(dataset_id)
        return X_train, y_train, X_test, y_test
    X, y, feature_names = load_dataset(dataset_id)
    group_col = schema.get("group_split")
    groups = X[:, feature_names.index(group_col)] if group_col else None
    X_train, X_test, y_train, y_test = split(X, y, seed=seed, groups=groups)
    return X_train, y_train, X_test, y_test


def load_and_subsample_test(dataset_id: str, seed: int):
    """Load the full split, then apply the schema's ``test_neg_cap`` negative subsample.

    Returns ``(X_train, y_train, X_test_scored, y_test_scored, info, prevalence,
    subsampled, negcap)``:

    - ``info`` is ``split_info(y_train, y_test)`` on the FULL (pre-subsample) test split.
    - ``prevalence`` is the full-split fraud rate (``n_fraud_test / n_test``, ``nan``
      if ``n_test == 0`` — a guard that is a no-op in practice since every dataset
      here has ``n_test > 0``).
    - ``subsampled`` is True iff the scored test set is smaller than the full test set.
    - ``negcap`` is the schema's ``test_neg_cap`` value (``None`` means no cap), handed
      back so callers that log a ``test_subsample_rule`` string can build it themselves.
    """
    X_train, y_train, X_test, y_test = load_full_split(dataset_id, seed)
    info = split_info(y_train, y_test)
    prevalence = info["n_fraud_test"] / info["n_test"] if info["n_test"] else float("nan")
    negcap = DATASETS[dataset_id].get("test_neg_cap")
    X_test_s, y_test_s = subsample_test(X_test, y_test, seed=seed, max_negatives=negcap)
    subsampled = len(y_test_s) < info["n_test"]
    return X_train, y_train, X_test_s, y_test_s, info, prevalence, subsampled, negcap
