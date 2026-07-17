# scripts/export_sample_data.R
# Freeze EDA extracts of six fraud datasets into data/sample/, one parquet
# per dataset, so the thesis qmd can render without the (unversioned,
# 1.6 GB) DuckDB database.
#
# Small datasets (see FULL_DATASETS) are shipped in FULL: every row, true
# class balance, valid for any view including rates and time series.
#
# Larger datasets are SAMPLED: ALL fraud rows (y = 1) plus a reservoir
# sample of N_LEGIT legitimate rows (y = 0), seed 42. In a sampled file
# fraud is complete and heavily overrepresented relative to the true class
# balance — use those ONLY for class-conditional views (e.g. feature
# distributions by class). NEVER use a sampled file to estimate prevalence,
# fraud rate, or time-series shares; use the full-data exports in
# data/external/ (scripts/export_data_summaries.R) for that. The `sampled`
# column (TRUE/FALSE) records which regime each file used.
#
# Run from the repo root:
#   Rscript scripts/export_sample_data.R

# duckdb/DBI/arrow only exist as R 4.5 builds (renv broken); load them with
# the 4.5 lib appended LAST, then restore the path — that lib also holds a
# broken ragg build that breaks plot devices if it stays discoverable.
local({
  lp <- .libPaths()
  lib45 <- path.expand("~/R/x86_64-pc-linux-gnu-library/4.5")
  if (!requireNamespace("duckdb", quietly = TRUE) && dir.exists(lib45)) {
    .libPaths(c(lp, lib45))
    on.exit(.libPaths(lp), add = TRUE)
  }
  suppressPackageStartupMessages({
    library(DBI)
    library(duckdb)
    library(arrow)
  })
})
suppressPackageStartupMessages(library(dplyr))

OUT_DIR <- "data/sample"
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

SEED <- 42
N_LEGIT <- 100000  # for SAMPLED datasets. Largest of {100000, 250000,
                    # 500000} that keeps every sampled file <= 50 MB:
                    # eu_cc alone hits ~63 MB at N=250000.

# Datasets small enough (as full zstd parquet) to ship whole: banksim 2.8 MB,
# cc_2025 9.6 MB. Full data drops the fraud-overrepresentation caveat for
# these, so their rate/count views are exact. The rest stay sampled
# (paysim full is ~280 MB; eu_cc/baf/fifar full trip GitHub's 50 MB warning).
FULL_DATASETS <- c("banksim", "cc_2025")

# Mirrors src/data/schema.py / scripts/export_data_summaries.R. `src` is a
# SQL relation exposing every raw column plus an INTEGER label `y`.
EDA_DATASETS <- list(
  eu_cc = list(
    src = "(SELECT *, CAST(Class AS INT) AS y FROM raw_eu_cc_kaggle_creditcard)"
  ),
  banksim = list(
    src = "(SELECT *, CAST(fraud AS INT) AS y FROM raw_banksim_bs140513_032310)"
  ),
  paysim = list(
    src = "(SELECT *, CAST(isFraud AS INT) AS y FROM raw_paysim_ps_20174392719_1491204439457_log)"
  ),
  baf = list(
    src = "(SELECT *, CAST(fraud_bool AS INT) AS y FROM raw_baf_base)"
  ),
  fifar = list(
    src = paste(
      "(SELECT * EXCLUDE (fraud_bool), CAST(fraud_bool AS INT) AS y FROM raw_fifar_train",
      " UNION ALL BY NAME",
      " SELECT * EXCLUDE (fraud_label), CAST(fraud_label AS INT) AS y FROM raw_fifar_test)"
    )
  ),
  cc_2025 = list(
    src = "(SELECT *, CAST(Fraud_Flag AS INT) AS y FROM raw_cc_fraud_2025_credit_card_fraud_2025)"
  )
)

con <- dbConnect(duckdb::duckdb(), "data/duckdb/fraud_datasets.duckdb",
                 read_only = TRUE)
dbExecute(con, "SET threads TO 1")

# Full extract: every row, `class` labelled from y.
full_sql <- function(src) sprintf(
  "SELECT *, CASE WHEN y = 1 THEN 'Fraud' ELSE 'Legitimate' END AS class
   FROM %s", src)

# Sampled extract: all fraud + a reservoir sample of n legit rows.
sample_sql <- function(src, n) sprintf("
  SELECT *, 'Fraud' AS class FROM %s WHERE y = 1
  UNION ALL
  SELECT *, 'Legitimate' AS class FROM (
    SELECT * FROM %s WHERE y = 0
    USING SAMPLE reservoir(%d ROWS) REPEATABLE (%d)
  )", src, src, n, SEED)

results <- list()

for (key in names(EDA_DATASETS)) {
  d <- EDA_DATASETS[[key]]
  is_full <- key %in% FULL_DATASETS
  df <- dbGetQuery(con, if (is_full) full_sql(d$src) else sample_sql(d$src, N_LEGIT))
  df$sampled <- !is_full

  path <- file.path(OUT_DIR, paste0(key, ".parquet"))
  ok <- tryCatch({
    write_parquet(df, path, compression = "zstd")
    TRUE
  }, error = function(e) FALSE)
  if (!ok) write_parquet(df, path)

  n_fraud <- sum(df$y == 1)
  n_legit <- sum(df$y == 0)
  size_mb <- file.size(path) / 1e6

  results[[key]] <- tibble(dataset = key, sampled = !is_full, n = nrow(df),
                            n_fraud = n_fraud, n_legit = n_legit,
                            size_mb = size_mb)

  cat(sprintf("%-8s  %-7s  n=%7d  fraud=%6d  legit=%7d  %6.2f MB\n",
              key, if (is_full) "full" else "sampled",
              nrow(df), n_fraud, n_legit, size_mb))
}

dbDisconnect(con, shutdown = TRUE)

summary_tbl <- bind_rows(results)
cat(sprintf("\nlargest file = %.2f MB, total = %.2f MB\n",
            max(summary_tbl$size_mb), sum(summary_tbl$size_mb)))

# --- README -----------------------------------------------------------
readme_rows <- summary_tbl |>
  mutate(line = sprintf("| %s.parquet | %s | %d | %d | %d | %.2f |",
                         dataset, ifelse(sampled, "sampled", "full"),
                         n, n_fraud, n_legit, size_mb))

readme <- c(
  "# data/sample/",
  "",
  "Frozen EDA extracts of the six formal-grid fraud datasets, committed to",
  "the repo so the thesis qmd can render without the unversioned 1.6 GB",
  "DuckDB database (`data/duckdb/fraud_datasets.duckdb`). Each file carries",
  "a `sampled` column (TRUE/FALSE) recording which regime it used.",
  "",
  "**Full** files (banksim, cc_2025) contain every row at the true class",
  "balance and are valid for any view, including fraud rates and counts.",
  "",
  "**Sampled** files (eu_cc, paysim, baf, fifar) contain ALL fraud rows",
  sprintf("(`y = 1`) plus a reservoir sample of up to %d legitimate rows", N_LEGIT),
  sprintf("(`y = 0`), via `USING SAMPLE reservoir(%d ROWS) REPEATABLE (%d)`", N_LEGIT, SEED),
  sprintf("(seed %d). In a sampled file fraud is complete and heavily", SEED),
  "overrepresented relative to the true class balance, so use sampled",
  "files ONLY for class-conditional views (e.g. feature distributions by",
  "class). NEVER use one to estimate prevalence, fraud rate, or",
  "time-series shares, use the full-data aggregates in `data/external/`",
  "instead (`scripts/export_data_summaries.R`).",
  "",
  "Regenerate with:",
  "",
  "```",
  "Rscript scripts/export_sample_data.R",
  "```",
  "",
  "| file | regime | rows | fraud | legit | size (MB) |",
  "|---|---|---|---|---|---|",
  readme_rows$line,
  ""
)
writeLines(readme, file.path(OUT_DIR, "README.md"))
cat(sprintf("wrote %s\n", file.path(OUT_DIR, "README.md")))
