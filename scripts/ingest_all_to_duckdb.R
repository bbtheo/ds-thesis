#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  if (!requireNamespace("DBI", quietly = TRUE)) {
    stop("Package 'DBI' is required. Install with renv::install('DBI').")
  }
  if (!requireNamespace("duckdb", quietly = TRUE)) {
    stop("Package 'duckdb' is required. Install with renv::install('duckdb').")
  }
})

`%||%` <- function(x, y) {
  if (is.null(x) || length(x) == 0) {
    return(y)
  }
  x
}

parse_args <- function(args) {
  defaults <- list(
    manifest = if (file.exists("scratchpad/dataset_manifest.csv")) {
      "scratchpad/dataset_manifest.csv"
    } else {
      "scratchpad/dataset_manifest_template.csv"
    },
    db = "data/duckdb/fraud_datasets.duckdb",
    raw_root = "data/raw",
    stage_root = "data/stage",
    log_root = "logs/ingest"
  )

  if (length(args) == 0) {
    return(defaults)
  }

  for (arg in args) {
    if (!startsWith(arg, "--")) {
      next
    }
    kv <- strsplit(sub("^--", "", arg), "=", fixed = TRUE)[[1]]
    key <- kv[[1]]
    value <- if (length(kv) > 1) kv[[2]] else ""
    if (key %in% names(defaults) && nzchar(value)) {
      defaults[[key]] <- value
    }
  }
  defaults
}

sanitize_token <- function(x, max_chars = 96) {
  if (is.null(x) || length(x) == 0) {
    x <- ""
  } else {
    x <- x[[1]]
  }
  if (is.na(x)) {
    x <- ""
  }
  x <- tolower(as.character(x))
  x <- gsub("[^a-z0-9]+", "_", x)
  x <- gsub("_+", "_", x)
  x <- gsub("^_|_$", "", x)
  if (!nzchar(x)) {
    x <- "x"
  }
  if (grepl("^[0-9]", x)) {
    x <- paste0("t_", x)
  }
  substr(x, 1, max_chars)
}

strip_known_extensions <- function(path) {
  stem <- basename(path)
  stem <- gsub(
    "\\.(csv|tsv|txt|parquet|json|jsonl|ndjson)(\\.gz)?$",
    "",
    stem,
    ignore.case = TRUE
  )
  stem <- gsub("\\.(zip|tar|tgz|gz|bz2|xz)$", "", stem, ignore.case = TRUE)
  stem
}

detect_file_kind <- function(path) {
  name <- tolower(basename(path))
  if (grepl("\\.parquet$", name)) return("parquet")
  if (grepl("\\.csv(\\.gz)?$", name)) return("csv")
  if (grepl("\\.tsv(\\.gz)?$", name)) return("tsv")
  if (grepl("\\.txt(\\.gz)?$", name)) return("txt")
  "unsupported"
}

is_archive <- function(path) {
  grepl("\\.(zip|tar|tgz|tar\\.gz|tar\\.bz2|tbz2|tar\\.xz)$", tolower(path))
}

safe_md5 <- function(path) {
  val <- tryCatch(unname(tools::md5sum(path)[[1]]), error = function(e) NA_character_)
  ifelse(is.na(val) || !nzchar(val), NA_character_, val)
}

run_command <- function(cmd, args = character()) {
  tryCatch({
    out <- system2(cmd, args = args, stdout = TRUE, stderr = TRUE)
    status <- attr(out, "status")
    if (is.null(status)) {
      status <- 0L
    }
    list(ok = as.integer(status) == 0L, status = as.integer(status), output = out)
  }, error = function(e) {
    list(ok = FALSE, status = 127L, output = as.character(e$message))
  })
}

cell_value <- function(row, col, default = "") {
  if (!(col %in% names(row))) {
    return(default)
  }
  val <- row[[col]]
  if (is.null(val) || length(val) == 0) {
    return(default)
  }
  one <- val[[1]]
  if (is.null(one) || is.na(one)) {
    return(default)
  }
  as.character(one)
}

pick_download_mode <- function(source_type, source_locator) {
  st <- tolower(source_type %||% "")
  loc <- source_locator %||% ""
  if (st == "kaggle_competition") return("kaggle_competition")
  if (grepl("kaggle_dataset", st, fixed = TRUE)) return("kaggle_dataset")
  if (grepl("competition", st, fixed = TRUE) && !grepl("^https?://", loc)) return("kaggle_competition")
  if (grepl("^https?://", loc)) return("http")
  if (grepl("kaggle", st, fixed = TRUE)) return("kaggle_dataset")
  if (grepl("http|uci|paper_supplement|github_release", st)) return("http")
  "manual"
}

download_dataset <- function(dataset_row, raw_dir, log_msg) {
  source_type <- cell_value(dataset_row, "source_type")
  locator <- trimws(cell_value(dataset_row, "source_locator"))
  if (!nzchar(locator)) {
    return(list(ok = FALSE, detail = "missing source_locator"))
  }

  mode <- pick_download_mode(source_type, locator)
  log_msg(sprintf("[%s] download mode = %s", dataset_row$dataset_id, mode))

  if (mode == "kaggle_dataset") {
    if (!nzchar(Sys.which("kaggle"))) {
      return(list(ok = FALSE, detail = "kaggle CLI not found"))
    }
    res <- run_command(
      "kaggle",
      c("datasets", "download", "-d", locator, "-p", raw_dir, "--force")
    )
    return(list(ok = res$ok, detail = paste(res$output, collapse = "\n")))
  }

  if (mode == "kaggle_competition") {
    if (!nzchar(Sys.which("kaggle"))) {
      return(list(ok = FALSE, detail = "kaggle CLI not found"))
    }
    res <- run_command(
      "kaggle",
      c("competitions", "download", "-c", locator, "-p", raw_dir, "--force")
    )
    return(list(ok = res$ok, detail = paste(res$output, collapse = "\n")))
  }

  if (mode == "http") {
    urls <- trimws(unlist(strsplit(locator, ";", fixed = TRUE)))
    urls <- urls[nzchar(urls)]
    if (length(urls) == 0) {
      return(list(ok = FALSE, detail = "no URLs parsed from source_locator"))
    }

    if (nzchar(Sys.which("curl"))) {
      all_ok <- TRUE
      lines <- character(0)
      for (i in seq_along(urls)) {
        url <- urls[[i]]
        file_name <- sub("\\?.*$", "", basename(url))
        if (!nzchar(file_name)) {
          file_name <- sprintf("download_%03d", i)
        }
        out_file <- file.path(raw_dir, file_name)
        res <- run_command(
          "curl",
          c("-L", "--fail", "--retry", "3", "-o", out_file, url)
        )
        lines <- c(lines, sprintf("url=%s status=%s", url, res$status))
        if (!res$ok) {
          all_ok <- FALSE
          lines <- c(lines, paste(res$output, collapse = "\n"))
        }
      }
      return(list(ok = all_ok, detail = paste(lines, collapse = "\n")))
    }

    all_ok <- TRUE
    lines <- character(0)
    for (i in seq_along(urls)) {
      url <- urls[[i]]
      file_name <- sub("\\?.*$", "", basename(url))
      if (!nzchar(file_name)) {
        file_name <- sprintf("download_%03d", i)
      }
      out_file <- file.path(raw_dir, file_name)
      ok <- TRUE
      err <- NULL
      tryCatch({
        utils::download.file(url, out_file, mode = "wb", quiet = TRUE)
      }, error = function(e) {
        ok <<- FALSE
        err <<- e$message
      })
      lines <- c(lines, sprintf("url=%s status=%s", url, ifelse(ok, "ok", "failed")))
      if (!ok) {
        all_ok <- FALSE
        lines <- c(lines, err %||% "download failed")
      }
    }
    return(list(ok = all_ok, detail = paste(lines, collapse = "\n")))
  }

  list(ok = FALSE, detail = paste("unsupported source_type/mode:", source_type, mode))
}

extract_to_stage <- function(raw_dir, stage_dir, log_msg) {
  if (!dir.exists(raw_dir)) {
    return(list(ok = FALSE, detail = "raw dir missing"))
  }

  files <- list.files(raw_dir, full.names = TRUE, recursive = TRUE)
  if (length(files) == 0) {
    return(list(ok = TRUE, detail = "no files found in raw dir"))
  }

  errs <- character(0)
  for (f in files) {
    if (dir.exists(f)) {
      next
    }

    lower <- tolower(f)
    if (grepl("\\.zip$", lower)) {
      ok <- TRUE
      tryCatch({
        utils::unzip(f, exdir = stage_dir)
      }, error = function(e) {
        ok <<- FALSE
        errs <<- c(errs, sprintf("unzip failed: %s (%s)", f, e$message))
      })
      if (ok) log_msg(sprintf("Extracted zip: %s", f))
      next
    }

    if (grepl("\\.(tar|tgz|tar\\.gz|tar\\.bz2|tbz2|tar\\.xz)$", lower)) {
      ok <- TRUE
      tryCatch({
        utils::untar(f, exdir = stage_dir)
      }, error = function(e) {
        ok <<- FALSE
        errs <<- c(errs, sprintf("untar failed: %s (%s)", f, e$message))
      })
      if (ok) log_msg(sprintf("Extracted tar: %s", f))
      next
    }

    # Copy non-archive files to stage for uniform loading.
    out <- file.path(stage_dir, basename(f))
    ok <- tryCatch(file.copy(f, out, overwrite = TRUE), error = function(e) FALSE)
    if (!ok) {
      errs <- c(errs, sprintf("copy failed: %s", f))
    }
  }

  if (length(errs) > 0) {
    return(list(ok = FALSE, detail = paste(errs, collapse = "\n")))
  }
  list(ok = TRUE, detail = "extraction/copy complete")
}

make_table_name <- function(dataset_id, raw_table_prefix, file_path) {
  prefix <- raw_table_prefix %||% ""
  if (!nzchar(prefix) || prefix == "meta_only") {
    prefix <- paste0("raw_", sanitize_token(dataset_id), "_")
  }
  prefix <- paste0(sanitize_token(prefix, max_chars = 80), "_")
  prefix <- gsub("_+", "_", prefix)
  prefix <- gsub("^_|_$", "", prefix)
  if (!grepl("^raw_", prefix)) {
    prefix <- paste0("raw_", prefix)
  }
  if (!grepl("_$", prefix)) {
    prefix <- paste0(prefix, "_")
  }

  stem <- sanitize_token(strip_known_extensions(file_path), max_chars = 80)
  table_name <- paste0(prefix, stem)
  table_name <- gsub("_+", "_", table_name)
  substr(table_name, 1, 120)
}

load_file_into_duckdb <- function(con, file_path, table_name) {
  kind <- detect_file_kind(file_path)
  if (kind == "unsupported") {
    return(list(ok = FALSE, kind = kind, row_count = NA_integer_, error = "unsupported file type"))
  }

  table_ident <- as.character(DBI::dbQuoteIdentifier(con, table_name))
  file_sql <- as.character(DBI::dbQuoteString(con, normalizePath(file_path, winslash = "/", mustWork = TRUE)))

  sql <- switch(
    kind,
    parquet = sprintf("CREATE OR REPLACE TABLE %s AS SELECT * FROM read_parquet(%s)", table_ident, file_sql),
    tsv = sprintf("CREATE OR REPLACE TABLE %s AS SELECT * FROM read_csv_auto(%s, delim='\\t')", table_ident, file_sql),
    txt = sprintf("CREATE OR REPLACE TABLE %s AS SELECT * FROM read_csv_auto(%s)", table_ident, file_sql),
    csv = sprintf("CREATE OR REPLACE TABLE %s AS SELECT * FROM read_csv_auto(%s)", table_ident, file_sql),
    NA_character_
  )

  ok <- TRUE
  err <- NULL
  n <- NA_integer_
  tryCatch({
    DBI::dbExecute(con, sql)
    n <- DBI::dbGetQuery(con, sprintf("SELECT COUNT(*) AS n FROM %s", table_ident))$n[[1]]
  }, error = function(e) {
    ok <<- FALSE
    err <<- e$message
  })

  list(ok = ok, kind = kind, row_count = n, error = err)
}

capture_table_schema <- function(con, run_id, dataset_id, table_name) {
  schema <- DBI::dbGetQuery(
    con,
    sprintf(
      paste(
        "SELECT ordinal_position, column_name, data_type",
        "FROM information_schema.columns",
        "WHERE table_schema = 'main' AND table_name = %s",
        "ORDER BY ordinal_position"
      ),
      as.character(DBI::dbQuoteString(con, table_name))
    )
  )

  if (nrow(schema) == 0) {
    return(invisible(NULL))
  }

  for (i in seq_len(nrow(schema))) {
    row <- schema[i, , drop = FALSE]
    DBI::dbExecute(
      con,
      sprintf(
        paste(
          "INSERT INTO meta_table_schemas",
          "(run_id, dataset_id, table_name, ordinal_pos, column_name, column_type)",
          "VALUES (%s, %s, %s, %s, %s, %s)"
        ),
        as.character(DBI::dbQuoteString(con, run_id)),
        as.character(DBI::dbQuoteString(con, dataset_id)),
        as.character(DBI::dbQuoteString(con, table_name)),
        as.integer(row$ordinal_position[[1]]),
        as.character(DBI::dbQuoteString(con, row$column_name[[1]])),
        as.character(DBI::dbQuoteString(con, row$data_type[[1]]))
      )
    )
  }
}

cfg <- parse_args(commandArgs(trailingOnly = TRUE))
dir.create(dirname(cfg$db), recursive = TRUE, showWarnings = FALSE)
dir.create(cfg$raw_root, recursive = TRUE, showWarnings = FALSE)
dir.create(cfg$stage_root, recursive = TRUE, showWarnings = FALSE)
dir.create(cfg$log_root, recursive = TRUE, showWarnings = FALSE)

if (!file.exists(cfg$manifest)) {
  stop(sprintf("Manifest not found: %s", cfg$manifest))
}

run_id <- format(Sys.time(), "%Y%m%dT%H%M%S")
log_file <- file.path(cfg$log_root, sprintf("ingest_%s.log", run_id))

log_msg <- function(msg) {
  line <- sprintf("[%s] %s", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), msg)
  cat(line, "\n")
  cat(line, "\n", file = log_file, append = TRUE)
}

log_msg(sprintf("Run ID: %s", run_id))
log_msg(sprintf("Manifest: %s", cfg$manifest))
log_msg(sprintf("DuckDB: %s", cfg$db))

manifest <- utils::read.csv(
  cfg$manifest,
  stringsAsFactors = FALSE,
  check.names = FALSE,
  na.strings = c("", "NA")
)

required_cols <- c(
  "dataset_id", "dataset_name", "source_type", "source_locator",
  "access_status", "download_status", "license", "raw_table_prefix",
  "timestamp_col", "label_col", "notes"
)

missing_cols <- setdiff(required_cols, names(manifest))
if (length(missing_cols) > 0) {
  stop(sprintf("Manifest missing columns: %s", paste(missing_cols, collapse = ", ")))
}

manifest <- manifest[!is.na(manifest$dataset_id) & nzchar(trimws(manifest$dataset_id)), , drop = FALSE]
if (nrow(manifest) == 0) {
  stop("Manifest has no datasets.")
}

drv <- duckdb::duckdb(cfg$db, read_only = FALSE)
con <- DBI::dbConnect(drv)
on.exit({
  try(DBI::dbDisconnect(con, shutdown = TRUE), silent = TRUE)
}, add = TRUE)

invisible(DBI::dbExecute(
  con,
  paste(
    "CREATE TABLE IF NOT EXISTS meta_dataset_registry (",
    "dataset_id TEXT PRIMARY KEY,",
    "dataset_name TEXT,",
    "source_type TEXT,",
    "source_locator TEXT,",
    "access_status TEXT,",
    "download_status TEXT,",
    "license TEXT,",
    "raw_table_prefix TEXT,",
    "timestamp_col TEXT,",
    "label_col TEXT,",
    "notes TEXT,",
    "first_seen_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP,",
    "last_run_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
    ")"
  )
))

invisible(DBI::dbExecute(
  con,
  paste(
    "CREATE TABLE IF NOT EXISTS meta_ingest_runs (",
    "run_id TEXT,",
    "event_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP,",
    "dataset_id TEXT,",
    "stage TEXT,",
    "status TEXT,",
    "detail TEXT",
    ")"
  )
))

invisible(DBI::dbExecute(
  con,
  paste(
    "CREATE TABLE IF NOT EXISTS meta_dataset_files (",
    "run_id TEXT,",
    "dataset_id TEXT,",
    "file_path TEXT,",
    "file_ext TEXT,",
    "file_size_bytes BIGINT,",
    "md5 TEXT,",
    "load_status TEXT,",
    "table_name TEXT,",
    "row_count BIGINT,",
    "error_message TEXT,",
    "observed_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
    ")"
  )
))

invisible(DBI::dbExecute(
  con,
  paste(
    "CREATE TABLE IF NOT EXISTS meta_blocked_datasets (",
    "run_id TEXT,",
    "dataset_id TEXT,",
    "reason TEXT,",
    "recorded_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
    ")"
  )
))

invisible(DBI::dbExecute(
  con,
  paste(
    "CREATE TABLE IF NOT EXISTS meta_table_schemas (",
    "run_id TEXT,",
    "dataset_id TEXT,",
    "table_name TEXT,",
    "ordinal_pos INTEGER,",
    "column_name TEXT,",
    "column_type TEXT,",
    "observed_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
    ")"
  )
))

q <- function(x) {
  if (is.null(x) || length(x) == 0 || is.na(x) || !nzchar(trimws(as.character(x)))) {
    return("NULL")
  }
  as.character(DBI::dbQuoteString(con, as.character(x)))
}

record_run <- function(dataset_id, stage, status, detail = "") {
  DBI::dbExecute(
    con,
    sprintf(
      "INSERT INTO meta_ingest_runs (run_id, dataset_id, stage, status, detail) VALUES (%s, %s, %s, %s, %s)",
      q(run_id), q(dataset_id), q(stage), q(status), q(detail)
    )
  )
  log_msg(sprintf("[%s] %s: %s %s", dataset_id, stage, status, detail))
}

record_blocked <- function(dataset_id, reason) {
  DBI::dbExecute(
    con,
    sprintf(
      "INSERT INTO meta_blocked_datasets (run_id, dataset_id, reason) VALUES (%s, %s, %s)",
      q(run_id), q(dataset_id), q(reason)
    )
  )
}

record_file <- function(dataset_id, file_path, file_ext, file_size_bytes, md5, load_status, table_name, row_count, error_message) {
  DBI::dbExecute(
    con,
    sprintf(
      paste(
        "INSERT INTO meta_dataset_files",
        "(run_id, dataset_id, file_path, file_ext, file_size_bytes, md5, load_status, table_name, row_count, error_message)",
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
      ),
      q(run_id),
      q(dataset_id),
      q(file_path),
      q(file_ext),
      ifelse(is.na(file_size_bytes), "NULL", as.character(as.numeric(file_size_bytes))),
      q(md5),
      q(load_status),
      q(table_name),
      ifelse(is.na(row_count), "NULL", as.character(as.numeric(row_count))),
      q(error_message)
    )
  )
}

update_registry_status <- function(dataset_id, download_status) {
  DBI::dbExecute(
    con,
    sprintf(
      "UPDATE meta_dataset_registry SET download_status = %s, last_run_ts = NOW() WHERE dataset_id = %s",
      q(download_status), q(dataset_id)
    )
  )
}

upsert_registry <- function(row) {
  DBI::dbExecute(
    con,
    sprintf(
      paste(
        "INSERT INTO meta_dataset_registry",
        "(dataset_id, dataset_name, source_type, source_locator, access_status, download_status, license, raw_table_prefix, timestamp_col, label_col, notes)",
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
        "ON CONFLICT(dataset_id) DO UPDATE SET",
        "dataset_name = excluded.dataset_name,",
        "source_type = excluded.source_type,",
        "source_locator = excluded.source_locator,",
        "access_status = excluded.access_status,",
        "download_status = excluded.download_status,",
        "license = excluded.license,",
        "raw_table_prefix = excluded.raw_table_prefix,",
        "timestamp_col = excluded.timestamp_col,",
        "label_col = excluded.label_col,",
        "notes = excluded.notes,",
        "last_run_ts = NOW()"
      ),
      q(row$dataset_id),
      q(row$dataset_name),
      q(row$source_type),
      q(row$source_locator),
      q(row$access_status),
      q(row$download_status),
      q(row$license),
      q(row$raw_table_prefix),
      q(row$timestamp_col),
      q(row$label_col),
      q(row$notes)
    )
  )
}

for (i in seq_len(nrow(manifest))) {
  row <- manifest[i, , drop = FALSE]
  dataset_id <- trimws(cell_value(row, "dataset_id"))
  access_status <- tolower(trimws(cell_value(row, "access_status")))
  source_type <- trimws(cell_value(row, "source_type"))

  upsert_registry(row)
  record_run(dataset_id, "start", "ok", sprintf("source_type=%s", source_type))

  if (identical(access_status, "blocked_confidential") ||
      identical(source_type, "not_downloadable_publicly") ||
      tolower(trimws(cell_value(row, "download_status"))) == "blocked") {
    reason <- "blocked by access policy/non-public source"
    record_blocked(dataset_id, reason)
    update_registry_status(dataset_id, "blocked")
    record_run(dataset_id, "access_check", "blocked", reason)
    next
  }

  if (identical(access_status, "needs_split")) {
    msg <- "manifest row needs split into specific datasets before ingest"
    update_registry_status(dataset_id, "failed")
    record_run(dataset_id, "access_check", "failed", msg)
    next
  }

  dataset_raw_dir <- file.path(cfg$raw_root, dataset_id)
  dataset_stage_dir <- file.path(cfg$stage_root, dataset_id)
  dir.create(dataset_raw_dir, recursive = TRUE, showWarnings = FALSE)
  dir.create(dataset_stage_dir, recursive = TRUE, showWarnings = FALSE)

  existing_files <- list.files(dataset_raw_dir, recursive = TRUE, full.names = TRUE)
  existing_files <- existing_files[!file.info(existing_files)$isdir]
  if (length(existing_files) > 0) {
    log_msg(sprintf("[%s] raw dir already has %d file(s); skipping download", dataset_id, length(existing_files)))
    record_run(dataset_id, "download", "skipped", sprintf("pre-existing files: %d", length(existing_files)))
  } else {
    download <- download_dataset(row, dataset_raw_dir, log_msg)
    if (!download$ok) {
      update_registry_status(dataset_id, "failed")
      record_run(dataset_id, "download", "failed", download$detail)
      next
    }
    record_run(dataset_id, "download", "ok", "download complete")
  }

  extracted <- extract_to_stage(dataset_raw_dir, dataset_stage_dir, log_msg)
  if (!extracted$ok) {
    update_registry_status(dataset_id, "failed")
    record_run(dataset_id, "extract", "failed", extracted$detail)
    next
  }
  record_run(dataset_id, "extract", "ok", extracted$detail)

  staged_files <- list.files(dataset_stage_dir, full.names = TRUE, recursive = TRUE)
  staged_files <- staged_files[file.info(staged_files)$isdir == FALSE]

  if (length(staged_files) == 0) {
    update_registry_status(dataset_id, "downloaded")
    record_run(dataset_id, "load_raw", "ok", "no staged files")
    next
  }

  loaded_n <- 0L
  failed_n <- 0L
  skipped_n <- 0L

  for (f in staged_files) {
    kind <- detect_file_kind(f)
    file_info <- file.info(f)
    file_ext <- tolower(tools::file_ext(f))
    md5 <- safe_md5(f)

    if (kind == "unsupported" || is_archive(f)) {
      skipped_n <- skipped_n + 1L
      record_file(
        dataset_id = dataset_id,
        file_path = f,
        file_ext = file_ext,
        file_size_bytes = file_info$size[[1]],
        md5 = md5,
        load_status = "skipped_binary",
        table_name = NA_character_,
        row_count = NA_integer_,
        error_message = "unsupported file type"
      )
      next
    }

    table_name <- make_table_name(dataset_id, cell_value(row, "raw_table_prefix"), f)
    loaded <- load_file_into_duckdb(con, f, table_name)
    if (loaded$ok) {
      loaded_n <- loaded_n + 1L
      record_file(
        dataset_id = dataset_id,
        file_path = f,
        file_ext = kind,
        file_size_bytes = file_info$size[[1]],
        md5 = md5,
        load_status = "loaded",
        table_name = table_name,
        row_count = loaded$row_count,
        error_message = NA_character_
      )
      capture_table_schema(con, run_id, dataset_id, table_name)
    } else {
      failed_n <- failed_n + 1L
      record_file(
        dataset_id = dataset_id,
        file_path = f,
        file_ext = kind,
        file_size_bytes = file_info$size[[1]],
        md5 = md5,
        load_status = "load_failed",
        table_name = table_name,
        row_count = NA_integer_,
        error_message = loaded$error %||% "unknown load error"
      )
    }
  }

  final_status <- if (failed_n > 0 && loaded_n == 0) {
    "failed"
  } else if (failed_n > 0 && loaded_n > 0) {
    "partial"
  } else {
    "downloaded"
  }

  update_registry_status(dataset_id, final_status)
  record_run(
    dataset_id,
    "load_raw",
    final_status,
    sprintf("loaded=%d failed=%d skipped=%d", loaded_n, failed_n, skipped_n)
  )
}

log_msg("Run complete.")
