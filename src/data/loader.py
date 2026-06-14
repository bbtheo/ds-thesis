"""
Dataset loader.

Loads a dataset from DuckDB, applies column drops, resolves the label column
for datasets where it is ambiguous, ordinal-encodes categoricals, and returns
(X, y) as float32 numpy arrays.

Feature scaling note
--------------------
The loader returns *unscaled* features. For kNN retrieval in the RAP pipeline
(conditions C3/C4), apply ``sklearn.preprocessing.StandardScaler`` after
splitting: fit on X_train, transform both X_train and X_test. The scaled
arrays are used only for the kNN index — the actual context rows passed to the
FTM remain unscaled. GBDTs and FTM conditions C1/C2 do not need scaling.
"""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

from src.data.schema import DATASETS, DUCKDB_PATH

# Candidate label column names used when schema entry has label=None
_LABEL_CANDIDATES = ["class", "fraud", "is_fraud", "isfraud", "fraud_bool", "label", "fraud_flag"]

# FiFAR columns that are analyst predictions / testbed metadata — not transaction features
_FIFAR_DROP_PATTERNS = [
    "expert", "analyst", "deployment", "capacity", "batch",
    "prediction", "score", "decision",
]


def load_dataset(
    dataset_id: str,
    *,
    con: duckdb.DuckDBPyConnection | None = None,
    limit: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load a dataset by ID and return (X, y, feature_names).

    Parameters
    ----------
    dataset_id:
        Key from DATASETS registry.
    con:
        Optional existing DuckDB connection. If None, opens DUCKDB_PATH read-only.
    limit:
        If set, loads only the first `limit` rows (useful for quick dev checks).

    Returns
    -------
    X : float32 ndarray, shape (n, d)
    y : int32 ndarray, shape (n,)  — 0/1 binary label
    feature_names : list of str
    """
    cfg = DATASETS[dataset_id]
    own_con = con is None
    if own_con:
        con = duckdb.connect(DUCKDB_PATH, read_only=True)

    try:
        table = cfg["table"]
        query = f"SELECT * FROM {table}"
        if limit:
            query += f" LIMIT {limit}"
        df = con.execute(query).df()
    finally:
        if own_con:
            con.close()

    df = _preprocess(df, cfg, dataset_id)
    label_col = _resolve_label(df, cfg, dataset_id)

    y = df[label_col].to_numpy(dtype=np.int32)
    X_df = df.drop(columns=[label_col])
    feature_names = list(X_df.columns)
    X = X_df.to_numpy(dtype=np.float32)

    return X, y, feature_names


def load_dataset_split(
    dataset_id: str,
    *,
    con: duckdb.DuckDBPyConnection | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Load FiFAR using its pre-defined train/test split.
    Only valid for dataset_id == 'fifar'.
    Returns (X_train, y_train, X_test, y_test, feature_names).
    """
    assert dataset_id == "fifar", "load_dataset_split only supports 'fifar'"
    cfg = DATASETS[dataset_id]
    own_con = con is None
    if own_con:
        con = duckdb.connect(DUCKDB_PATH, read_only=True)

    try:
        train_df = con.execute(f"SELECT * FROM {cfg['table']}").df()
        test_df = con.execute(f"SELECT * FROM {cfg['test_table']}").df()
    finally:
        if own_con:
            con.close()

    # Normalise column names before any processing
    train_df.columns = [c.strip().lower() for c in train_df.columns]
    test_df.columns = [c.strip().lower() for c in test_df.columns]

    # Resolve label columns (train and test may differ, e.g. fifar)
    train_label = cfg["label"].lower()
    test_label = cfg.get("test_label", train_label).lower()

    drop_cols = [c.lower() for c in cfg.get("drop", [])]
    cat_cols_cfg = [c.lower() for c in cfg.get("categoricals", [])]

    # Drop + detect analyst cols
    extra_drop = _detect_fifar_analyst_cols(train_df)
    all_drop = list(set(drop_cols + extra_drop))

    for df in (train_df, test_df):
        cols_to_drop = [c for c in all_drop if c in df.columns]
        df.drop(columns=cols_to_drop, inplace=True)

    # Fit ordinal encoder on train categoricals, apply to both
    cat_cols = [c for c in cat_cols_cfg if c in train_df.columns]
    if cat_cols:
        enc = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )
        train_df[cat_cols] = enc.fit_transform(train_df[cat_cols].astype(str))
        test_cat = [c for c in cat_cols if c in test_df.columns]
        test_df[test_cat] = enc.transform(test_df[test_cat].astype(str))

    feature_names = [c for c in train_df.columns if c != train_label]
    # Ensure test has same feature columns
    test_feature_names = [c for c in feature_names if c in test_df.columns]

    X_train = train_df[feature_names].to_numpy(dtype=np.float32)
    y_train = train_df[train_label].to_numpy(dtype=np.int32)
    X_test = test_df[test_feature_names].to_numpy(dtype=np.float32)
    y_test = test_df[test_label].to_numpy(dtype=np.int32)

    return X_train, y_train, X_test, y_test, feature_names


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _preprocess(df: pd.DataFrame, cfg: dict, dataset_id: str) -> pd.DataFrame:
    """Drop unwanted columns, clean up, encode categoricals."""
    # Normalise column names to lowercase + strip whitespace
    df.columns = [c.strip().lower() for c in df.columns]

    # Lowercase config keys too
    drop_cols = [c.lower() for c in cfg.get("drop", [])]
    cat_cols = [c.lower() for c in cfg.get("categoricals", [])]

    # Dataset-specific extra drops
    if dataset_id == "fifar":
        drop_cols = drop_cols + _detect_fifar_analyst_cols(df)

    # Drop requested columns (ignore missing ones)
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Ordinal-encode categoricals (fit+transform in one shot — no split leakage
    # at this stage; splitter handles seeded splits after load)
    present_cats = [c for c in cat_cols if c in df.columns]
    if present_cats:
        enc = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )
        df[present_cats] = enc.fit_transform(df[present_cats].astype(str))

    # Drop any remaining non-numeric columns (safety net)
    non_numeric = df.select_dtypes(exclude="number").columns.tolist()
    if non_numeric:
        label_candidate = _find_label_col(df, cfg)
        non_numeric_features = [c for c in non_numeric if c != label_candidate]
        if non_numeric_features:
            print(f"[loader] dropping non-numeric cols in {dataset_id}: {non_numeric_features}")
            df = df.drop(columns=non_numeric_features)

    return df


def _resolve_label(df: pd.DataFrame, cfg: dict, dataset_id: str) -> str:
    """Return the label column name, resolving ambiguous cases."""
    if cfg.get("label"):
        col = cfg["label"].lower()
        if col in df.columns:
            return col
    # Auto-detect
    col = _find_label_col(df, cfg)
    if col is None:
        raise ValueError(
            f"Cannot resolve label column for '{dataset_id}'. "
            f"Columns: {list(df.columns)}"
        )
    return col


def _find_label_col(df: pd.DataFrame, cfg: dict) -> str | None:
    for candidate in _LABEL_CANDIDATES:
        if candidate in df.columns:
            return candidate
    return None


def _detect_fifar_analyst_cols(df: pd.DataFrame) -> list[str]:
    """Identify FiFAR analyst/testbed columns to drop."""
    return [
        c for c in df.columns
        if any(pat in c for pat in _FIFAR_DROP_PATTERNS)
    ]
