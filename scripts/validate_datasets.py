"""
Phase 1 validation: load every dataset, print shape, fraud rate, and NaN count.
Run from the ds-gradu root: uv run python scripts/validate_datasets.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.data.loader import load_dataset, load_dataset_split
from src.data.schema import DATASETS
from src.data.splitter import split, split_info

SEED = 42
# Use small limit for large datasets during validation (set None for full load)
LIMITS = {
    "paysim": 200_000,
    "fifar":  None,    # uses pre-defined split
    "baf":    None,
}

print(f"{'Dataset':<12} {'Rows':>8} {'Features':>9} {'Fraud%':>7} {'NaNs':>6} {'Split OK':>9}")
print("-" * 58)

for ds_id in DATASETS:
    try:
        if ds_id == "fifar":
            X_tr, y_tr, X_te, y_te, feat = load_dataset_split(ds_id)
            X = np.concatenate([X_tr, X_te])
            y = np.concatenate([y_tr, y_te])
        else:
            limit = LIMITS.get(ds_id)
            X, y, feat = load_dataset(ds_id, limit=limit)

            X_tr, X_te, y_tr, y_te = split(X, y, seed=SEED)

        n, d = X.shape
        fraud_pct = y.mean() * 100
        n_nans = int(np.isnan(X).sum())
        info = split_info(y_tr, y_te)
        split_ok = "OK" if info["n_fraud_train"] > 0 and info["n_fraud_test"] > 0 else "WARN"

        print(f"{ds_id:<12} {n:>8,} {d:>9} {fraud_pct:>6.2f}% {n_nans:>6} {split_ok:>9}")

    except Exception as e:
        print(f"{ds_id:<12} ERROR: {e}")

print()
print("Done.")
