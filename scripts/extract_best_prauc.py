"""Extract the best PR-AUC pivot for the data chapter.

Reads the formal-grid runs (pipeline v4, grid n_estimators) from
results/runs/ and writes data/external/dataset_best_prauc.csv with, per
formal dataset, the best seed-mean PR-AUC over model x condition cells and
the cell that attains it. Consumed by inline R in sections/4_data.qmd.

Run from the repo root: python3 scripts/extract_best_prauc.py
"""

from pathlib import Path

import pandas as pd

FORMAL = ["eu_cc", "banksim", "paysim", "fifar", "baf"]

runs = pd.concat(
    [pd.read_parquet(p) for p in Path("results/runs").glob("*.parquet")],
    ignore_index=True,
)
grid = runs[
    (runs["pipeline_version"] == 4)
    & (runs["dataset"].isin(FORMAL))
    # FTM grid runs use n_estimators=4; GBDTs carry None (no ensemble knob).
    & ((runs["n_estimators"] == 4) | runs["n_estimators"].isna())
]

cell_means = (
    grid.groupby(["dataset", "model", "condition"], as_index=False)["pr_auc"]
    .mean()
)
best = cell_means.loc[cell_means.groupby("dataset")["pr_auc"].idxmax()]
best = best.rename(columns={"pr_auc": "best_pr_auc", "model": "best_model",
                            "condition": "best_condition"})
best["best_pr_auc"] = best["best_pr_auc"].round(3)
best = best.sort_values("best_pr_auc", ascending=False)

out = Path("data/external/dataset_best_prauc.csv")
best.to_csv(out, index=False)
print(best.to_string(index=False))
print(f"wrote {out}")
