"""
Dataset registry.

Each entry defines how to load and preprocess one dataset from DuckDB.
Keys are the short dataset IDs used throughout the experiment pipeline.
"""

DUCKDB_PATH = "data/duckdb/fraud_datasets.duckdb"

# Columns to always drop before modelling (IDs, leakage, temporal keys)
# Categoricals are ordinal-encoded; everything else is cast to float32.

DATASETS: dict[str, dict] = {
    "ai_banking": {
        "table": "raw_ai_banking_2025_banking_fraud_dataset",
        "label": "is_fraud",
        "drop": ["transaction_id", "customer_id", "transaction_location", "transaction_time"],
        "categoricals": ["transaction_type", "device_used"],
        "dev_only": True,
        "notes": "~10k rows; 28% fraud (nearly balanced); dev/debug only — excluded from formal experiment grid",
    },
    "eu_cc": {
        "table": "raw_eu_cc_kaggle_creditcard",
        "label": "Class",
        "drop": [],
        "categoricals": [],
        "dev_only": False,
        # Only ~98 test frauds: rare-event metrics are indefensible under
        # negative subsampling, so eu_cc is always scored on the FULL test set.
        "test_neg_cap": None,
        "notes": "284k rows; 0.17% fraud; PCA features V1-V28",
    },
    "cc_2025": {
        "table": "raw_cc_fraud_2025_credit_card_fraud_2025",
        "label": "Fraud_Flag",
        "drop": ["Transaction_ID", "Customer_ID", "Transaction_Date", "Merchant_ID"],
        "categoricals": [
            "Merchant_Category",
            "Card_Type",
            "Transaction_Type",
            "Country",
            "Device_Type",
        ],
        "dev_only": False,
        "notes": "500k rows; 1.5% fraud; synthetic",
    },
    "banksim": {
        "table": "raw_banksim_bs140513_032310",
        "label": "fraud",
        "drop": ["zipcodeori", "zipMerchant"],
        "categoricals": ["category", "customer", "merchant", "age", "gender"],
        # customer is an entity ID (4,112 customers over 594k rows). With a
        # random row-level split the same customer appears in train and test,
        # so the encoded ID acts as a memorisation key and leaks labels.
        # group_split assigns each customer wholly to train or test instead.
        "group_split": "customer",
        "dev_only": False,
        "test_neg_cap": 30_000,
        "notes": "~594k rows; agent-based simulator; grouped split by customer to prevent entity-ID label leakage",
    },
    "paysim": {
        "table": "raw_paysim_ps_20174392719_1491204439457_log",
        "label": "isFraud",
        "drop": ["nameOrig", "nameDest", "isFlaggedFraud"],
        "categoricals": ["type"],
        "dev_only": False,
        "test_neg_cap": 30_000,
        "notes": "6.3M rows; 0.13% fraud; fraud only in TRANSFER+CASH_OUT. Dev runs at 200k subsample; formal grid uses full 6.3M.",
    },
    "fifar": {
        "table": "raw_fifar_train",
        "label": "fraud_bool",
        "test_table": "raw_fifar_test",
        "test_label": "fraud_label",
        "drop": ["case_id", "month", "model_score", "assignment"],
        "categoricals": [
            "payment_type",
            "employment_status",
            "housing_status",
            "source",
            "device_os",
        ],
        "dev_only": False,
        "fixed_split": True,
        "n_seeds": 1,
        "test_neg_cap": 30_000,
        "notes": "1M rows; pre-defined train/test split; bank account opening. Single seed (split is fixed, seed has no effect).",
    },
    "baf": {
        "table": "raw_baf_base",
        "label": "fraud_bool",
        "drop": ["month"],
        "categoricals": [
            "payment_type",
            "employment_status",
            "housing_status",
            "source",
            "device_os",
        ],
        "dev_only": False,
        "test_neg_cap": 30_000,
        "notes": "1M rows; 1.1% fraud; BAF Base variant",
    },
}
