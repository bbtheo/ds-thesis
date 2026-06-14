# Retrieval-Augmented Prediction for Tabular Foundation Models under Heavy Class Imbalance

Master's thesis (Data Science, University of Helsinki) — evaluated on financial fraud detection datasets.

## Overview

Tabular foundation models (FTMs) such as TabPFN and TabICL classify by *in-context learning*: instead of training weights, they take a labelled context of training rows and predict query rows in a single forward pass. This gives them a bounded context budget. Under heavy class imbalance (fraud rates of 0.1–1%), a context drawn by random subsampling is flooded with legitimate transactions and contains almost no fraud — so the model has little to learn the minority class from.

**Retrieval-Augmented Prediction (RAP)** is a test-time context-construction strategy that addresses this without touching the pre-trained model. For each test batch it builds a context that is

1. **relevant** — retrieved by kNN search around the test batch, so the context resembles the queries; and
2. **label-balanced** — oversampled to a controlled fraud ratio, so the minority class is adequately represented.

The thesis isolates the two effects through a controlled chain of context-construction conditions and benchmarks FTMs against gradient-boosted trees (XGBoost, CatBoost) trained on the full data.

## Approach

Context-construction conditions (FTM context held to its limit; everything else fixed):

| Condition | Context | Fraud ratio | Isolates |
|---|---|---|---|
| **C1** Random | Uniform random subsample | Natural | Baseline |
| **C2** Stratified | Class-balanced subsample | 50/50 | Ratio effect (vs C1) |
| **C3** kNN | Retrieval around test batch | Natural | Relevance effect (vs C1) |
| **C4** RAP | Retrieval + fraud oversampling | Swept 5–50% | Ratio tuning (vs C3) |

- **Models:** TabPFN v2 / 2.5 / 2.6 / v3 and TabICLv2 (`tabpfn` 8.0.8); GBDT baselines XGBoost and CatBoost (trained on the full dataset — the context limit is an inherent FTM property).
- **Datasets (6):** EU CC (ULB), CC Fraud 2025, BankSim, PaySim, FiFAR, BAF — spanning 285k–6.3M rows and 0.13–1.5% fraud.
- **Primary metrics:** PR-AUC (prevalence-corrected), Recall@1%FPR, Recall@5%FPR.

## Repository layout

```
src/data/        dataset registry, DuckDB loader, train/test splitter
src/models/      FTM wrappers (ftm.py) and GBDT wrappers (gbdt.py)
src/rap/         RAP retrieval pipeline (Phase 3, in progress)
src/eval/        evaluation metrics
src/experiment/  idempotent run driver (one run = dataset × model × condition × seed)
scripts/         dataset validation and grid drivers
results/runs/    one Parquet row per completed run
methodology.qmd  thesis methodology chapter (Quarto + Typst)
data-doc.qmd     dataset documentation
CLAUDE.md        authoritative project notes and current status
```

Python handles modelling and inference; R + Quarto + Typst handle analysis and the thesis document. They communicate only through the Parquet files in `results/runs/`.

## Setup and usage

Dependencies are managed with [uv](https://docs.astral.sh/uv/).

```bash
# Validate all datasets (shapes, fraud rates)
uv run python scripts/validate_datasets.py

# Run a single experiment
uv run python -m src.experiment.runner --dataset eu_cc --model tabpfn_v2 --condition C1 --seed 42

# Run all pending C1/C2 grid runs (idempotent, safe to restart)
uv run python scripts/run_pending_c1c2.py
```

The gated TabPFN models require `TABPFN_TOKEN` set in the environment. All FTMs run on GPU (`device="cuda"`). The runner is idempotent: each config hashes to a `run_id` and skips if its Parquet already exists.

## Reproducibility

All randomness flows through `numpy.random.default_rng(seed)`; library versions are pinned in `uv.lock`. A `_PIPELINE_VERSION` integer is part of the config hash, so any change to pipeline logic invalidates cached results and analysis can filter to a single, consistent pipeline version.

## Status

Phase 1 (data) and Phase 2 (baselines, C1/C2 grid) are complete or in progress under pipeline v4. The RAP retrieval pipeline (C3/C4) is the next phase. See `CLAUDE.md` for detailed current status.
