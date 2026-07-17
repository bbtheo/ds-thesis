# scripts/export_data_summaries.R
# Compute full-data EDA summary statistics for the data chapter inside
# DuckDB and write them to data/external/ as parquet, so the thesis qmd
# chunks can render without the (unversioned) DuckDB file. Everything is an
# exact full-data aggregate — no sampling.
#
# Run from the repo root:
#   Rscript scripts/export_data_summaries.R
#
# Outputs (data/external/):
#   eda_feature_auc.parquet        dataset, feature, auc, auc_abs, n_fraud, n_legit
#                                  (single-feature ROC-AUC, tie-aware rank-sum)
#   eda_feature_stats.parquet      dataset, feature, class, n, mean, sd, min, max
#   eda_feature_quantiles.parquet  dataset, feature, class, p, q
#   eda_time_fraud_rate.parquet    dataset, time_unit, t, n, n_fraud
#   eda_categorical_fraud_rate.parquet
#                                  dataset, feature, level, n, n_fraud,
#                                  n_levels_total (top 30 levels by volume)

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

OUT_DIR <- "data/external"
PROBS <- c(0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
MAX_LEVELS <- 30

# Mirrors src/data/schema.py. `src` is a SQL relation exposing every raw
# column plus an INTEGER label `y`. `drop` columns (IDs, leakage, temporal
# keys) and `label_col` are excluded from the feature list but stay in `src`
# so the time queries can still see them. `time` gives the binning
# expression for the fraud-share-over-time export (NULL = none).
EDA_DATASETS <- list(
  eu_cc = list(
    label_col = "Class",
    src = "(SELECT *, CAST(Class AS INT) AS y FROM raw_eu_cc_kaggle_creditcard)",
    drop = character(),
    categoricals = character(),
    time = list(bin = "floor(Time / 3600)", unit = "hour")
  ),
  banksim = list(
    label_col = "fraud",
    src = "(SELECT *, CAST(fraud AS INT) AS y FROM raw_banksim_bs140513_032310)",
    drop = c("zipcodeOri", "zipMerchant"),
    categoricals = c("category", "customer", "merchant", "age", "gender"),
    time = list(bin = "step", unit = "day (step)")
  ),
  paysim = list(
    label_col = "isFraud",
    src = "(SELECT *, CAST(isFraud AS INT) AS y FROM raw_paysim_ps_20174392719_1491204439457_log)",
    drop = c("nameOrig", "nameDest", "isFlaggedFraud"),
    categoricals = "type",
    time = list(bin = "step", unit = "hour (step)")
  ),
  baf = list(
    label_col = "fraud_bool",
    src = "(SELECT *, CAST(fraud_bool AS INT) AS y FROM raw_baf_base)",
    drop = "month",
    categoricals = c("payment_type", "employment_status", "housing_status",
                     "source", "device_os"),
    time = list(bin = "month", unit = "month")
  ),
  fifar = list(
    label_col = NULL,  # normalised to y in src (train/test label names differ)
    src = paste(
      "(SELECT * EXCLUDE (fraud_bool), CAST(fraud_bool AS INT) AS y FROM raw_fifar_train",
      " UNION ALL BY NAME",
      " SELECT * EXCLUDE (fraud_label), CAST(fraud_label AS INT) AS y FROM raw_fifar_test)"
    ),
    drop = c("case_id", "month", "model_score", "batch", "assignment", "decision"),
    categoricals = c("payment_type", "employment_status", "housing_status",
                     "source", "device_os"),
    time = list(bin = "month", unit = "month")
  ),
  cc_2025 = list(
    label_col = "Fraud_Flag",
    src = "(SELECT *, CAST(Fraud_Flag AS INT) AS y FROM raw_cc_fraud_2025_credit_card_fraud_2025)",
    drop = c("Transaction_ID", "Customer_ID", "Transaction_Date", "Merchant_ID"),
    categoricals = c("Merchant_Category", "Card_Type", "Transaction_Type",
                     "Country", "Device_Type"),
    time = NULL  # degenerate dataset; appendix evidence is the AUC table
  )
)

con <- dbConnect(duckdb::duckdb(), "data/duckdb/fraud_datasets.duckdb",
                 read_only = TRUE)

# Numeric modelling features (excludes label, drops, categoricals).
numeric_features <- function(key) {
  d <- EDA_DATASETS[[key]]
  info <- dbGetQuery(con, sprintf("DESCRIBE SELECT * FROM %s", d$src))
  is_num <- grepl("^(TINYINT|SMALLINT|INTEGER|BIGINT|HUGEINT|FLOAT|DOUBLE|DECIMAL)",
                  info$column_type)
  setdiff(info$column_name[is_num],
          c("y", d$label_col, d$drop, d$categoricals))
}

probs_sql <- paste0("[", paste(PROBS, collapse = ", "), "]")

# Tie-aware single-feature ROC-AUC (Mann-Whitney): group rows by feature
# value, then AUC = sum over values of npos * (negatives below + half the
# negatives tied) / (Npos * Nneg).
auc_sql <- function(src, feat) sprintf("
  WITH vals AS (
    SELECT CAST(%s AS DOUBLE) AS x, y FROM %s WHERE %s IS NOT NULL
  ), grp AS (
    SELECT x,
           count(*) FILTER (WHERE y = 1) AS npos,
           count(*) FILTER (WHERE y = 0) AS nneg
    FROM vals GROUP BY x
  ), cum AS (
    SELECT npos, nneg,
           coalesce(sum(nneg) OVER (
             ORDER BY x ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
           ), 0) AS neg_below
    FROM grp
  )
  SELECT sum(npos * (neg_below + nneg / 2.0))
           / (sum(npos)::DOUBLE * sum(nneg)::DOUBLE) AS auc,
         sum(npos) AS n_fraud, sum(nneg) AS n_legit
  FROM cum", feat, src, feat)

# Class-conditional moments + quantiles in one scan per feature.
stats_sql <- function(src, feat) sprintf("
  WITH vals AS (
    SELECT CAST(%s AS DOUBLE) AS x, y FROM %s WHERE %s IS NOT NULL
  ), agg AS (
    SELECT y, count(*) AS n, avg(x) AS mean, stddev(x) AS sd,
           min(x) AS min, max(x) AS max,
           quantile_cont(x, %s) AS qs
    FROM vals GROUP BY y
  )
  SELECT y, n, mean, sd, min, max,
         unnest(%s) AS p, unnest(qs) AS q
  FROM agg", feat, src, feat, probs_sql, probs_sql)

quote_ident <- function(x) paste0('"', x, '"')
class_of <- function(y) ifelse(y == 1, "fraud", "legit")

auc_rows <- list(); stat_rows <- list(); quant_rows <- list()
time_rows <- list(); cat_rows <- list()

for (key in names(EDA_DATASETS)) {
  d <- EDA_DATASETS[[key]]
  feats <- numeric_features(key)
  cat(sprintf("%-8s %d numeric features, %d categoricals\n",
              key, length(feats), length(d$categoricals)))

  for (f in feats) {
    fq <- quote_ident(f)

    a <- dbGetQuery(con, auc_sql(d$src, fq))
    auc_rows[[length(auc_rows) + 1]] <- tibble(
      dataset = key, feature = f,
      auc = a$auc, auc_abs = pmax(a$auc, 1 - a$auc),
      n_fraud = a$n_fraud, n_legit = a$n_legit
    )

    s <- dbGetQuery(con, stats_sql(d$src, fq)) |>
      mutate(dataset = key, feature = f, class = class_of(y))
    stat_rows[[length(stat_rows) + 1]] <-
      distinct(s, dataset, feature, class, n, mean, sd, min, max)
    quant_rows[[length(quant_rows) + 1]] <-
      select(s, dataset, feature, class, p, q)
  }

  for (f in d$categoricals) {
    lv <- dbGetQuery(con, sprintf(
      "SELECT CAST(%s AS VARCHAR) AS level, count(*) AS n, sum(y) AS n_fraud
       FROM %s GROUP BY 1 ORDER BY n DESC",
      quote_ident(f), d$src
    ))
    cat_rows[[length(cat_rows) + 1]] <- lv |>
      slice_head(n = MAX_LEVELS) |>
      mutate(dataset = key, feature = f, n_levels_total = nrow(lv))
  }

  if (!is.null(d$time)) {
    time_rows[[length(time_rows) + 1]] <- dbGetQuery(con, sprintf(
      "SELECT CAST(%s AS DOUBLE) AS t, count(*) AS n, sum(y) AS n_fraud
       FROM %s GROUP BY 1 ORDER BY 1",
      d$time$bin, d$src
    )) |>
      mutate(dataset = key, time_unit = d$time$unit)
  }
}

dbDisconnect(con, shutdown = TRUE)

exports <- list(
  eda_feature_auc = bind_rows(auc_rows),
  eda_feature_stats = bind_rows(stat_rows),
  eda_feature_quantiles = bind_rows(quant_rows),
  eda_time_fraud_rate = bind_rows(time_rows) |>
    select(dataset, time_unit, t, n, n_fraud),
  eda_categorical_fraud_rate = bind_rows(cat_rows) |>
    select(dataset, feature, level, n, n_fraud, n_levels_total)
)

for (name in names(exports)) {
  path <- file.path(OUT_DIR, paste0(name, ".parquet"))
  write_parquet(exports[[name]], path)
  cat(sprintf("wrote %s  (%d rows)\n", path, nrow(exports[[name]])))
}
