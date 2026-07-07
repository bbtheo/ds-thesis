# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Master's thesis project: **Retrieval-Augmented Prediction (RAP) for Tabular Foundation Models under Heavy Class Imbalance** — evaluated on financial fraud detection datasets.

The core contribution is a test-time context construction strategy for ICL-based tabular foundation models (FTMs). Instead of random subsampling the training context (which under heavy class imbalance floods the model with irrelevant legitimate transactions), RAP retrieves a structurally similar, label-balanced context around each test batch using kNN retrieval and controlled fraud ratio oversampling.

**Models:** TabPFN v2, TabPFN 2.5, TabPFN 2.6, TabPFN v3, TabICLv2 (`tabpfn` 8.0.8)
**Baselines:** XGBoost, CatBoost (trained on full dataset — the context limit is an inherent FTM property)
**Formal grid datasets (6):** EU CC (ULB), CC Fraud 2025, BankSim, PaySim, FiFAR, BAF
**Dev-only dataset:** AI Banking 2025 (28% fraud — nearly balanced, used for RAP pipeline dev only)
**Degenerate dataset:** CC Fraud 2025 (cc_2025) — fraud labels are statistically independent of all features (Amount, Distance, Is_International etc. are identical between fraud/non-fraud). All models score PR-AUC ≈ fraud rate (0.015). **Do not queue further cc_2025 runs** — existing results are sufficient to document it as degenerate in the thesis (an example of a poorly constructed synthetic dataset).
**Primary metrics:** PR-AUC, Recall@1%FPR, Recall@5%FPR
**Recall@FPR definition:** conservative step-function value of the empirical ROC curve (highest TPR among operating points with FPR ≤ target). Linear ROC interpolation is deliberately not used — interpolated points are not achievable decision thresholds and overestimate recall on coarse curves (changed in pipeline v3; thesis methods section must state this).

See `scratchpad/experiment-plan.md` for the full experiment design. **Note:** that plan predates the v3/v4 work (last updated 2026-04-13) — it still lists 4 FTMs, pipeline v2, interpolated Recall@FPR, and the old `runtime_s` results schema. Where the two disagree, **this file (CLAUDE.md) is authoritative**; the plan is kept for the RAP pipeline design (§5) and RQ mapping (§1), which are unchanged.

---

## Current Status (2026-07-07)

- **Pipeline version 4** (`_PIPELINE_VERSION` in `src/experiment/runner.py`). v4 changes (all fold into one bump, full rerun): negative-subsampled test set on large datasets (keep ALL positives, cap negatives via `test_neg_cap` in schema; default 30,000) with **prevalence-corrected PR-AUC** (`ap_at_prevalence` in `metrics.py`); `n_estimators` added to the config (default 8); model-aware predict chunk (`_PREDICT_CHUNK`, does not affect results); new result columns (see Results Schema). v3 changes were: banksim grouped split, conservative Recall@FPR, `batch_size=None` for C1/C2, timing split. Bump the version whenever model defaults or pipeline logic changes — it is part of the config hash and invalidates cached results.
- **`tabpfn` upgraded to 8.0.8** (was 7.1.1) and **TabPFN v3 added as a 5th FTM** (`tabpfn_3`). The library version is recorded per-run (`tabpfn_lib`/`tabicl_lib`) but is NOT in the config hash — the v4 bump + full rerun keeps all results on one library, avoiding a silent mix. **v3 license accepted 2026-06-14** at https://ux.priorlabs.ai for the `TABPFN_TOKEN` account (verified live); `tabpfn_3` is in `run_pending_c1c2.py` `FTM_MODELS`. (All v2/2.5/2.6/v3 licenses are accepted.)
- **`results/runs/` holds only pipeline-v4 rows** (the v4 rerun is complete). The only surviving pre-v4 results are the 23 cc_2025 parquets in `results/runs_cc2025_degenerate/`, kept deliberately as documentation of that dataset's degeneracy (do not rerun).
- **C1/C2 v4 rerun** via `scripts/run_pending_c1c2.py` (now loads each (dataset, seed) split once and reuses it across models/conditions; idempotent). Enumerates all pending C1/C2 configs for the formal grid (cc_2025 excluded).
- **CatBoost runs with `thread_count=1`** (set in `gbdt.py`). At the default thread count CatBoost 1.2.10 SIGSEGVs in its TBB pool during the long fit on paysim (~5M rows) and took down the whole unattended driver on 2026-06-14. CatBoost CPU training is thread-count-invariant (verified bit-identical), so this changes no results — it is just slower on the largest datasets. NOT a memory/data/metric issue (load is 3.5 GB/4 s). See the auto-memory note `catboost-paysim-segfault`.
- **C3/C4 (RAP retrieval) are implemented (Phase 3, 2026-06-18).** `src/rap/{retriever,sampler,context}.py` plus a per-group FTM path in the runner. Per-group design (one retrieval + one FTM refit per silhouette-chosen test cluster); no `pipeline_version` bump (C1/C2/GBDT hashes/results untouched, only new C3/C4 rows). C3/C4 FTMs are `tabpfn_3` + `tabiclv2` only. Smoke-tested on `ai_banking` (tabiclv2 C3+C4) and `eu_cc` (tabpfn_3 C3) — idempotent, finite metrics. Dev inspection: `scripts/dev_rap_validation.py`. See `scratchpad/phase3-plan.md`. **Phase-4 grid driver built + launched (2026-06-18):** `scripts/run_pending_c3c4.py` enumerates every pending C3/C4 formal-grid cell (cc_2025 excluded): C3 + C4 ratio sweep {0.05,0.10,0.20,0.30,0.50} × {`tabiclv2`,`tabpfn_3`} × 13 dataset-seed units = **156 runs** (`metric=cosine`, `n_estimators=4`); idempotent, loads each split once. **All grids are COMPLETE** (formal C1–C4, the C2 ratio × context-size grid, the RAP design-space sweep, and the relevance ablation). Headline: **RAP retrieval is a negative result** (C3 < C1 and C4 < C2 on all 5 informative datasets); **C2 stratified balancing is the positive finding** (zero-shot FTMs at/above external SOTA). See `thesis-outline.md` for the full result narrative with provenance.
- **Full review applied (2026-07-06):** the corrections in `scratchpad/review-2026-07-06/correction-log.md` are implemented — outline/bib fixes, `ap_at_prevalence` tie fix (+ `tests/test_metrics.py`), thread-pinned C3/C4 grouping (`threadpoolctl`), per-group retrieval now timed into `setup_s`, runner validation (C3/C4 model restriction, stray `fraud_ratio` rejected), verified checkpoint context limits (v3 = 1M, 2.6 = 100k — our 50k is a design budget), stale parquets quarantined. BLK-1 resolved: the eu_cc SOTA figure 0.897 was the single best run (tabpfn_3, C2, seed 7); the fair seed-mean is **0.869** vs external 0.867 → "at par", not "above".
- **Repo cleaned for public release (2026-07-07):** deleted the abandoned-direction lit files, one-off probe scripts, superseded drivers (`run_c2.py`, `diag_c1_vs_c4.py`, `diag_rap_designspace.py`, the `*_daemon.sh` wrappers), rendered artifacts (`datasets.html`, `data-doc.typ`, tracked `.quarto/` cache), the dead `references.bib`, `main.py`, and the broken renv machinery (`.Rprofile`, `renv.lock`, `renv/`). `methodology.qmd` moved to `scratchpad/methodology-draft.qmd` (superseded draft; its bibliography path is stale). `results/runs_stale_prev4/` trimmed to the 23 cc_2025 files and renamed `results/runs_cc2025_degenerate/`; `results/diag/` trimmed to the two anchor-injection evidence files. **`scratchpad/eda/` was deleted entirely (user decision)** — the EDA figures cited in `thesis-outline.md` no longer exist in the repo and must be regenerated for the thesis document itself.
- **Not created yet:** the `analysis/` R scripts (Phase 5). The run scripts in `scripts/` serve as the grid driver. `thesis.qmd` renders only `sections/introduction.qmd` so far — the remaining chapters exist as `thesis-outline.md`, not as prose.

---

## Project Structure

```
ds-gradu/
├── src/
│   ├── data/
│   │   ├── schema.py          # dataset registry (dev_only flags, label cols, categoricals, group_split)
│   │   ├── loader.py          # DuckDB → numpy/pandas; ordinal encoding
│   │   └── splitter.py        # seeded stratified 80/20 split; grouped split for entity-ID datasets
│   ├── models/
│   │   ├── gbdt.py            # XGBoost + CatBoost wrappers (CatBoost thread_count=1 — see note)
│   │   └── ftm.py             # TabPFN v2 / 2.5 / 2.6 / v3, TabICLv2 wrappers; CONTEXT_LIMITS
│   ├── rap/                   # Phase 3 — RAP retrieval for C3/C4 (per-group)
│   │   ├── retriever.py       #   KNNRetriever: brute NearestNeighbors (cosine / euclidean)
│   │   ├── sampler.py         #   sample_c3 (mixed + single-class guard), sample_c4 (class-conditional + with-replacement top-up)
│   │   └── context.py         #   RetrievalContextBuilder: scaler + retrievers + kmeans/silhouette grouping; build() dispatch
│   ├── eval/
│   │   └── metrics.py         # PR-AUC, Recall@k%FPR, ROC-AUC, F1
│   └── experiment/
│       └── runner.py          # one idempotent run: dataset × model × condition × seed
├── scripts/
│   ├── validate_datasets.py       # loads all datasets, prints shapes and fraud rates
│   ├── run_pending_c1c2.py        # enumerate + run all pending C1/C2 formal-grid runs
│   ├── run_pending_c3c4.py        # enumerate + run all pending C3/C4 (RAP) formal-grid runs (cosine, n_est=4)
│   ├── run_c2_grid.py             # C2 fraud-ratio × context-size cross-tab sweep → results/runs_c2grid
│   ├── run_rap_designspace.py     # RAP design-space sweep (metric × granularity × sampling) → results/runs_rapds
│   ├── run_relevance_ablation.py  # ratio-matched 2×2 relevance ablation (KR/RK arms) → results/runs_ablation
│   ├── diag_hybrid_anchor.py      # anchor-injection case study (banksim/tabiclv2) → results/diag
│   ├── analyze_c3c4.py            # read-only summary pivots over the C3/C4 grid rows
│   ├── dev_rap_validation.py      # Phase 3 dev: inspect C3/C4 grouping + retrieval on ai_banking
│   └── ingest_all_to_duckdb.R     # raw CSVs → DuckDB ingestion (manifest: scratchpad/dataset_manifest.csv)
├── results/
│   ├── runs/                      # formal C1–C4 grid — one parquet per run (all pipeline v4)
│   ├── runs_c2grid/               # C2 ratio × context-size sweep
│   ├── runs_rapds/                # RAP design-space sweep
│   ├── runs_ablation/             # relevance ablation
│   ├── runs_cc2025_degenerate/    # pre-v4 cc_2025 runs kept as degeneracy documentation
│   └── diag/                      # diag_hybrid.log + preds parquet (anchor-injection evidence, cited in outline §7.2)
├── data/
│   ├── duckdb/
│   │   └── fraud_datasets.duckdb   # all 7 datasets ingested (not versioned)
│   └── raw/                        # original source CSVs (not versioned)
├── tests/                     # metrics regression tests (pytest, dev dependency via uv)
├── sections/                  # thesis chapter sources included by thesis.qmd (introduction.qmd so far)
├── thesis.qmd                 # main thesis document (Quarto → Typst PDF / HTML; CI renders on push)
├── thesis-outline.md          # distilled 9-chapter outline with per-claim result provenance
├── data-doc.qmd               # dataset documentation (Quarto + Typst)
├── scratchpad/                # planning docs and working notes (experiment-plan.md lives here)
└── pyproject.toml             # Python deps managed by uv
```

---

## Package Management

This project uses **uv** (not pip, not conda).

```bash
uv add <package>              # add a dependency
uv run python <script>        # run scripts in the venv
uv run python -m src.experiment.runner ...  # run experiment
```

---

## Build Commands

**Run a single experiment:**
```bash
uv run python -m src.experiment.runner --dataset eu_cc --model tabpfn_v2 --condition C1 --seed 42
```

**Run all pending C1/C2 formal-grid runs (idempotent, safe to restart):**
```bash
uv run python scripts/run_pending_c1c2.py
```

**Validate all datasets:**
```bash
uv run python scripts/validate_datasets.py
```

**Render documents (R + Quarto):**
```bash
quarto render data-doc.qmd
```

TabPFN 2.5/2.6 are gated: runner commands need `TABPFN_TOKEN` set in the environment (stored in `~/.bashrc`).

---

## Technical Notes

- **Python** for all model training, inference, and RAP pipeline
- **R + Quarto + Typst** for analysis, tables, figures, and thesis document
- **No reticulate** — Python writes parquet to `results/runs/`, R reads via `arrow::open_dataset()`
- **GPU:** NVIDIA 5070; all FTMs use `device="cuda"`
- **Feature scaling:** loader returns unscaled features. For kNN retrieval (C3/C4), apply `StandardScaler` (fit on train only) before building the retriever index. Context rows passed to FTMs remain unscaled.
- **Idempotent runner:** config hash → run_id → output path. Skips if file exists. Safe to kill and restart.
- **Reproducibility:** all randomness via `numpy.random.default_rng(seed)`. Versions pinned in `uv.lock`.

---

## Data

All datasets are in `data/duckdb/fraud_datasets.duckdb`. Schema registry is `src/data/schema.py`.

| Dataset | Rows | Features | Fraud % | Train fraud rows (~) | In formal grid? | Seeds |
|---|---|---|---|---|---|---|
| ai_banking | 10k | 6 | 28.4% | — | No (dev only) | — |
| eu_cc | 285k | 30 | 0.17% | 394 | Yes | 3 |
| cc_2025 | 500k | 11 | 1.5% | — | Yes (degenerate — do not run further) | 3 |
| banksim | 595k | 7 | 1.2% | 5,760 | Yes (grouped split — see note) | 3 |
| paysim | 6.3M | 7 | 0.13% | 6,570 | Yes (200k dev, full formal) | 3 |
| fifar | 603k | 29 | 1.2% | 5,705 | Yes (pre-defined split) | 1 |
| baf | 1M | 30 | 1.1% | 8,823 | Yes | 3 |

Train fraud counts matter for C2/C4 feasibility (see Experiment Conditions below).

**Split protocol decisions:**
- Random stratified 80/20 splits (banksim: grouped by `customer`; fifar: fixed pre-defined split).
- **Test-set negative subsampling (v4):** large datasets score all positives + `test_neg_cap` (=30,000) negatives; eu_cc scores the full test set (`test_neg_cap=None`). PR-AUC is prevalence-corrected. This is the dominant speedup (paysim ~42×). Changing a cap requires a pipeline-version bump.
- Time columns (`Time` in eu_cc, `step` in banksim/paysim) are **deliberately kept as features** under the random split; the thesis methods section must state this choice explicitly.
- The temporal-split question is handled as an **ablation study**, not in the main grid — natural vehicle: BAF split by `month` (its published protocol; `month` is already excluded as a feature).

---

## Experiment Conditions

| ID | Name | Context construction | Fraud ratio | Notes |
|---|---|---|---|---|
| C1 | Random | Random subsample to FTM context limit | Natural | Baseline |
| C2 | Stratified | Stratified subsample | 50/50 | Isolates ratio effect vs C1 |
| C3 | kNN (control) | kNN retrieval around test batch centroid | Natural | Isolates relevance effect (C3 vs C1) |
| C4 | kNN + ratio (RAP) | kNN retrieval + oversampled fraud | Swept: 5/10/20/30/50% | Full RAP; C4 vs C3 isolates ratio tuning |

GBDTs run under C1 only — trained on full training set (no context limit). This is intentional: the limited context window is an inherent FTM property being evaluated.

**C4 sampler policy (decided 2026-06-12, applies when implementing `src/rap/sampler.py`):**
- Fill the fraud quota **with replacement** (duplicating fraud rows) when unique train fraud runs out. Oversampling *is* the method — do not cap at available fraud (capping would silently record `fraud_ratio=0.5` while the context contains far less).
- Log `context_fraud_unique` alongside `context_fraud_n` so the duplication factor is computable per run (requires a results-schema addition → fold into the next pipeline-version bump).
- **Uniform protocol across all datasets** — no special-casing eu_cc. Duplication-heavy cells are annotated or excluded at *analysis* time via a stated duplication threshold, not removed from the grid.
- Feasibility background: at the 10k context (tabpfn_v2), every dataset except eu_cc covers the full ratio sweep with unique fraud rows; at the 48–50k contexts (tabpfn_25/26, tabiclv2) **no dataset** reaches ratios ≥ 20% without duplication; eu_cc (≈394 train frauds) reaches no swept ratio on any model without duplication.
- C3 must define a guard for fraud-free retrieved contexts (single-class `y_ctx` breaks `predict_proba(...)[:, 1]`).

---

## Experiment Grid

**5 FTMs** (tabpfn_v2, tabpfn_25, tabpfn_26, tabpfn_3, tabiclv2). The v3 license was accepted 2026-06-14; `tabpfn_3` is in `run_pending_c1c2.py` `FTM_MODELS`. The grid is fully run.

- 16 dataset-seed units (5 datasets × 3 seeds + FiFAR × 1 seed)
- FTM core: 5 models × 4 conditions × 16 = 320
- GBDT core: 2 models × 1 condition × 16 = 32
- C4 fraud ratio sweep: 5 FTMs × 5 ratios × 16 = 400 (the C4 cell in "FTM core" is the default-ratio point; the sweep enumerates all 5 ratios, so the default ratio is shared, not additive — net new C4 runs are 5 × 4 × 16 = 320)
- Batch size sweep: 1 × 4 × 2 datasets × 3 seeds = 24

cc_2025 cells beyond the already-collected C1/C2 results are dropped (degenerate dataset).

---

## Results Schema

Each run writes one row to `results/runs/{run_id}.parquet`:

```
run_id, dataset, model, condition, seed, batch_size, k, fraud_ratio, metric,
n_estimators, pipeline_version, pr_auc, recall_at_1fpr, recall_at_5fpr, roc_auc, f1,
n_train, n_test, n_fraud_train, n_fraud_test,
n_test_scored, n_fraud_test_scored, test_subsample_rule, test_subsample_seed,
context_size, context_fraud_n, context_fraud_unique, n_groups, group_silhouette,
tabpfn_lib, tabicl_lib,
setup_s, fit_s, predict_s, inference_s
```

Notes (pipeline v4):
- `n_test`/`n_fraud_test` are the **FULL** test-split counts (true prevalence). `n_test_scored`/`n_fraud_test_scored` are what was actually scored after negative subsampling. `test_subsample_rule` is `'full'` or `'neg_cap_<N>'`; `test_subsample_seed` is the subsample RNG seed (`seed ^ 0x5CA1E`) or `None`.
- **PR-AUC on subsampled datasets is prevalence-corrected** (`ap_at_prevalence` at the full-split prevalence) — equals standard AP to machine precision when not subsampled. Recall@FPR / ROC-AUC are invariant to negative subsampling. eu_cc is always scored full (`test_neg_cap=None`, only ~98 test frauds). Methods note: the conservative-step Recall@FPR has a tiny downward bias (step = 1/N_neg_scored), negligible at N≈30k.
- `n_estimators` = FTM ensemble passes (GBDT `None`). Reducing it speeds predict ~linearly but changes outputs. **All reported FTM results use `n_estimators=4`** — a settled decision (driver `N_ESTIMATORS = 4` in `run_pending_c1c2.py`) to keep the full grid tractable on one GPU. The runner's own default stays 8, so single-run commands without `--n_estimators` use 8 and are NOT comparable to grid runs — always check this column when comparing.
- `context_fraud_unique` = unique fraud rows in context (== `context_fraud_n` for C1/C2; differs under C4 with-replacement oversampling).
- `tabpfn_lib`/`tabicl_lib` = library versions for provenance (not hashed).
- `setup_s` = build + weight load (**+ scaler + retrieval index + test grouping + Σ per-group kNN query/sampling for C3/C4**); `fit_s` = GBDT train / FTM context fit (**Σ per-group fit for C3/C4**); `predict_s` = test prediction (**Σ per-group predict for C3/C4**); `inference_s` = `fit_s + predict_s`. Use `fit_s`/`predict_s` for timing. **Caveat (review A-7):** C3/C4 parquets written before 2026-07-06 timed per-group retrieval into NO column — their recorded components are lower bounds on RAP overhead.
- **C3/C4 (RAP retrieval) — Phase 3.** Per-group design: the scored test set is clustered (k-means; `G` chosen by silhouette over `k=2..20`, recorded as the **output** `n_groups`; `group_silhouette` = selection score), and the FTM is refit once per group around its cluster centre. `n_groups`/`group_silhouette` are **C3/C4-only** (`NULL` for GBDT and C1/C2; written as nullable `Int64`/`float64` so an all-None row never becomes Arrow `null` type). `metric` (`'cosine'` default / `'euclidean'`) and, for C4, `fraud_ratio` are the real C3/C4 factors; **`batch_size` and `k` are dropped from the C3/C4 config hash** (grouping is automatic, retrieval depth is derived from `context_size`) and are `None` for all conditions. C1/C2/GBDT hashes are unchanged → existing results stay valid, **no `pipeline_version` bump**. `context_fraud_n` is the per-group mean fraud count (C3 varies by group; C4 = `round(fraud_ratio·context_size)`); `context_fraud_unique` < `context_fraud_n` only under C4 with-replacement oversampling. C3/C4 FTMs are `tabpfn_3` + `tabiclv2` only.
- banksim uses a grouped 80/20 split by `customer`; other random-split datasets use stratified splits.
- Filter analysis to `pipeline_version == 4` — v2/v3 rows are stale (except cc_2025, kept deliberately).
- **Stale pre-v4 parquets:** quarantined 2026-07-06, then trimmed 2026-07-07 — only the 23 cc_2025 degeneracy-documentation files survive, in `results/runs_cc2025_degenerate/`. `results/runs/` holds only v4 rows (312 formal-grid + 4 ai_banking dev runs), so `arrow::open_dataset("results/runs")` unifies cleanly; still filter to the 5 formal datasets.
- **`ap_at_prevalence` tie handling fixed 2026-07-06** (review C-1): tied scores are now grouped into one threshold (sklearn-equivalent); regression tests in `tests/test_metrics.py`. Recorded v4 PR-AUCs were computed pre-fix; measured bias ≤ +0.002 on the worst collapsed C4 columns, ~+0.00001 on C1/C2 — below every claimed effect; state this bound in the thesis methods.

---

## Bibliography

Citation keys follow: `authorYYYY_short_descriptor`
Examples: `hollmann2025accurate_tabular_foundation_model`, `jesus2022baf_arxiv`

Canonical bibliography: `thesis-refs.bib` (repo root). `/home/theo/thesis-lit/bibliography.bib` is the old pre-consolidation file — do **not** re-sync from it (its mcdowell entry carries a malformed note that `thesis-refs.bib` fixes).

---

## Writing Style

- Academic but concise; avoid hype and marketing language
- Clear topic sentences and short paragraphs
- Define domain terms on first use; expand acronyms
- Active voice; concrete claims supported by citations
- All quantitative claims must cite a source or an experiment run ID
