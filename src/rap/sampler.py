"""
Context assembly for the RAP retrieval conditions (C3/C4).

Both samplers operate on a single (scaled) retrieval ``centre`` — the centroid
of one test group — and return **global train indices** plus per-group stats.
They never touch the FTM or the unscaled feature matrix; turning indices into an
unscaled context is the builder's job (see ``context.py``).

Stats returned (per group)
--------------------------
``fraud_n``      : number of fraud rows placed in the context (counts duplicates)
``fraud_unique`` : number of *distinct* fraud rows (``< fraud_n`` only when C4
                   oversamples with replacement)
``guard_fired``  : whether the single-class guard injected a row (C3 only)

All randomness flows through a caller-supplied ``numpy.random.Generator`` so a
run is reproducible from its seed.
"""

from __future__ import annotations

import numpy as np

from src.rap.retriever import KNNRetriever


def sample_c3(
    retr_all: KNNRetriever,
    retr_fraud: KNNRetriever,
    retr_legit: KNNRetriever,
    y_train: np.ndarray,
    fraud_idx: np.ndarray,
    legit_idx: np.ndarray,
    centre: np.ndarray,
    context_size: int,
) -> tuple[np.ndarray, dict]:
    """C3 — mixed retrieval at the natural ratio, with a single-class guard.

    Pull the ``context_size`` globally-nearest train rows to ``centre`` (whatever
    fraud:legit mix that yields). If the retrieved context has zero fraud (or, far
    more rarely, zero legit), inject the single nearest row of the missing class —
    a single-class context would break ``predict_proba(...)[:, 1]``. The injected
    row replaces the farthest retrieved row so the context size is preserved.

    ``retr_all`` is fit on the FULL scaled train pool, so its indices are already
    global; ``retr_fraud``/``retr_legit`` are fit on class subsets and their local
    indices are mapped back via ``fraud_idx``/``legit_idx``.
    """
    idx = retr_all.query(centre, context_size)  # global indices (full-pool retriever)
    fraud_n = int(y_train[idx].sum())
    legit_n = len(idx) - fraud_n
    guard_fired = False

    if fraud_n == 0:
        inject = fraud_idx[retr_fraud.query(centre, 1)]
        idx = np.concatenate([idx[:-1], inject])
        guard_fired = True
    elif legit_n == 0:
        inject = legit_idx[retr_legit.query(centre, 1)]
        idx = np.concatenate([idx[:-1], inject])
        guard_fired = True

    fraud_n = int(y_train[idx].sum())  # recount after a possible guard injection
    stats = {"fraud_n": fraud_n, "fraud_unique": fraud_n, "guard_fired": guard_fired}
    return idx, stats


def sample_c4(
    retr_fraud: KNNRetriever,
    retr_legit: KNNRetriever,
    fraud_idx: np.ndarray,
    legit_idx: np.ndarray,
    centre: np.ndarray,
    context_size: int,
    fraud_ratio: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict]:
    """C4 (RAP) — class-conditional retrieval with a controlled fraud ratio.

    Retrieve the nearest fraud and nearest legit rows to ``centre`` *separately*,
    so the context holds exactly ``round(fraud_ratio * context_size)`` fraud rows
    and the rest legit. When the global fraud pool is smaller than the fraud quota
    the deficit is filled **with replacement** by duplicating retrieved fraud rows
    — oversampling is the method, so the realised ratio is honoured and the
    duplication factor is recorded via ``fraud_unique``.
    """
    n_fraud = int(round(fraud_ratio * context_size))
    n_legit = context_size - n_fraud
    n_global_fraud = len(fraud_idx)
    n_global_legit = len(legit_idx)

    # Fraud: nearest unique frauds, then top up with replacement if short.
    n_fraud_unique = min(n_fraud, n_global_fraud)
    if n_fraud_unique > 0:
        chosen_fraud = fraud_idx[retr_fraud.query(centre, n_fraud_unique)]
    else:
        chosen_fraud = np.empty(0, dtype=np.int64)
    if n_fraud > n_fraud_unique and n_fraud_unique > 0:
        extra = rng.choice(chosen_fraud, size=n_fraud - n_fraud_unique, replace=True)
        chosen_fraud = np.concatenate([chosen_fraud, extra])

    # Legit: nearest legits (clipped to the legit pool, which is never the binder).
    n_legit_take = min(n_legit, n_global_legit)
    if n_legit_take > 0:
        chosen_legit = legit_idx[retr_legit.query(centre, n_legit_take)]
    else:
        chosen_legit = np.empty(0, dtype=np.int64)

    idx = np.concatenate([chosen_fraud, chosen_legit]).astype(np.int64, copy=False)
    rng.shuffle(idx)  # break the fraud-then-legit ordering before the FTM sees it
    stats = {
        "fraud_n": int(n_fraud),
        "fraud_unique": int(n_fraud_unique),
        "guard_fired": False,
    }
    return idx, stats
