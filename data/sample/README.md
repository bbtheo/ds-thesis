# data/sample/

Frozen EDA extracts of the six formal-grid fraud datasets, committed to
the repo so the thesis qmd can render without the unversioned 1.6 GB
DuckDB database (`data/duckdb/fraud_datasets.duckdb`). Each file carries
a `sampled` column (TRUE/FALSE) recording which regime it used.

**Full** files (banksim, cc_2025) contain every row at the true class
balance and are valid for any view, including fraud rates and counts.

**Sampled** files (eu_cc, paysim, baf, fifar) contain ALL fraud rows
(`y = 1`) plus a reservoir sample of up to 100000 legitimate rows
(`y = 0`), via `USING SAMPLE reservoir(100000 ROWS) REPEATABLE (42)`
(seed 42). In a sampled file fraud is complete and heavily
overrepresented relative to the true class balance, so use sampled
files ONLY for class-conditional views (e.g. feature distributions by
class). NEVER use one to estimate prevalence, fraud rate, or
time-series shares, use the full-data aggregates in `data/external/`
instead (`scripts/export_data_summaries.R`).

Regenerate with:

```
Rscript scripts/export_sample_data.R
```

| file | regime | rows | fraud | legit | size (MB) |
|---|---|---|---|---|---|
| eu_cc.parquet | sampled | 100300 | 492 | 99808 | 27.82 |
| banksim.parquet | full | 594643 | 7200 | 587443 | 2.79 |
| paysim.parquet | sampled | 108100 | 8213 | 99887 | 4.64 |
| baf.parquet | sampled | 109934 | 11029 | 98905 | 8.52 |
| fifar.parquet | sampled | 105942 | 7133 | 98809 | 10.43 |
| cc_2025.parquet | full | 500000 | 7500 | 492500 | 9.66 |

