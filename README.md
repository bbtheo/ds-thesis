# Retrieval-Augmented Prediction for Tabular Foundation Models under Heavy Class Imbalance

Master's thesis (Data Science, University of Helsinki), evaluated on financial fraud detection datasets.

## Overview

Tabular foundation models (FTMs) such as TabPFN and TabICL classify by *in-context learning*: instead of training weights, they take a labelled context of training rows and predict query rows in a single forward pass. This gives them a bounded context budget. Under heavy class imbalance (fraud rates of 0.1 to 1%), a context drawn by random subsampling is flooded with legitimate transactions and contains almost no fraud, so the model has little to learn the minority class from.

**Retrieval-Augmented Prediction (RAP)** is a test-time context-construction strategy that addresses this without touching the pre-trained model: for each group of test rows it retrieves a structurally similar context by kNN search and rebalances it to a controlled fraud ratio by oversampling. The thesis decomposes RAP into its two levers, *relevance* (retrieval) and *ratio* (balancing), through a controlled chain of context-construction conditions, and benchmarks five FTMs against gradient-boosted trees trained on the full data.

## Findings

The full experiment grid is complete. Two headline results, one negative and one positive:

- **Retrieval does not help (negative result).** kNN-retrieved contexts underperform random ones at matched fraud ratios (C3 < C1 and C4 < C2 on all five informative datasets). A ratio-matched relevance ablation and an anchor-injection case study attribute the failure to the retrieval of the *legitimate* context rows: locally retrieved legitimate neighbours remove the global contrast the FTM needs to separate the classes.
- **Stratified context balancing wins (positive result).** Simply rebalancing a random context to 50/50 (C2) improves FTM PR-AUC by about +0.06 on average, with roughly 95% of FTM runs improving. Under C2, zero-shot FTMs beat the best tuned GBDT baseline on every informative dataset and match or exceed soundly evaluated published results, with no per-dataset training or tuning.

One of the six datasets (CC Fraud 2025) is shown to be *degenerate*: its fraud labels are statistically independent of all features, so every model scores at chance. It is documented as a case study in synthetic-benchmark quality rather than used for comparisons.

## Experimental design

Context-construction conditions (FTM context held to its limit; everything else fixed):

| Condition | Context | Fraud ratio | Isolates |
|---|---|---|---|
| **C1** Random | Uniform random subsample | Natural | Baseline |
| **C2** Stratified | Class-balanced subsample | 50/50 | Ratio effect (vs C1) |
| **C3** kNN | Retrieval around test groups | Natural | Relevance effect (vs C1) |
| **C4** RAP | Retrieval + fraud oversampling | Swept 5 to 50% | Ratio tuning (vs C3) |

- **Models:** TabPFN v2 / 2.5 / 2.6 / v3 and TabICLv2 (`tabpfn` 8.0.8); GBDT baselines XGBoost and CatBoost, trained on the full dataset (the context limit is an inherent FTM property).
- **Datasets (6):** EU CC (ULB), CC Fraud 2025, BankSim, PaySim, FiFAR, BAF, spanning 285k to 6.3M rows and fraud rates of 0.13% to 1.5%.
- **Primary metrics:** PR-AUC (prevalence-corrected under test-set negative subsampling), Recall@1%FPR, Recall@5%FPR (conservative ROC step, no interpolation).

## Repository layout

```
src/data/          dataset registry, DuckDB loader, train/test splitters
src/models/        FTM wrappers (ftm.py) and GBDT wrappers (gbdt.py)
src/rap/           RAP pipeline: kNN retriever, ratio sampler, context builder
src/eval/          evaluation metrics
src/experiment/    idempotent run driver (one run = dataset x model x condition x seed)
scripts/           grid drivers, ablation/sweep drivers, dataset validation, ingestion
tests/             metric regression tests (pytest)
results/runs/                 formal C1-C4 grid (one Parquet per run)
results/runs_c2grid/          C2 fraud-ratio x context-size sweep
results/runs_rapds/           RAP design-space sweep (metric x granularity x sampling)
results/runs_ablation/        ratio-matched relevance ablation
results/runs_cc2025_degenerate/  pre-v4 runs documenting the degenerate dataset
thesis.qmd         thesis document (Quarto -> Typst PDF / self-contained HTML)
thesis-outline.md  chapter-by-chapter outline with per-claim result provenance
data-doc.qmd       dataset documentation
CLAUDE.md          project working notes and current status
```

Python handles modelling and inference; R + Quarto + Typst handle analysis and the thesis document. They communicate only through the Parquet files under `results/`, read from R via `arrow::open_dataset()`.

## Setup and usage

Python dependencies are managed with [uv](https://docs.astral.sh/uv/).

```bash
# Validate all datasets (shapes, fraud rates)
uv run python scripts/validate_datasets.py

# Run a single experiment
uv run python -m src.experiment.runner --dataset eu_cc --model tabpfn_v2 --condition C1 --seed 42

# Fill all pending formal-grid runs (idempotent, safe to kill and restart)
uv run python scripts/run_pending_c1c2.py   # C1/C2 + GBDT baselines
uv run python scripts/run_pending_c3c4.py   # C3/C4 (RAP)

# Metric regression tests
uv run python -m pytest
```

Raw datasets are ingested into a local DuckDB file with `scripts/ingest_all_to_duckdb.R` (the data itself is not versioned). The gated TabPFN models require a `TABPFN_TOKEN` in the environment. All FTMs run on GPU (`device="cuda"`).

## Reproducibility

All randomness flows through `numpy.random.default_rng(seed)`; library versions are pinned in `uv.lock`. Each run's config hashes to a `run_id`, and the runner skips configs whose Parquet already exists, so every driver is idempotent. A `_PIPELINE_VERSION` integer is part of the config hash: any change to pipeline logic invalidates cached results, and analysis filters to a single consistent pipeline version (currently v4).

## Thesis document

`thesis.qmd` renders to PDF (via Typst) and self-contained HTML. A GitHub Actions workflow re-renders it on every push to `main` and publishes both files to the rolling [`latest` release](../../releases/tag/latest).
