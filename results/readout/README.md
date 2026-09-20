# Decoder-readout case study (NOT grid results)

Files in this directory come from `scripts/run_readout.py`, run on 2026-09-20 in a
separate environment (`.venv-readout`: tabpfn 8.5.0, tabpfn-extensions 0.6.2,
torch 2.14.0+cu130) because `tabpfn_extensions.interpretability.get_decoder_readout`
needs `tabpfn>=8.4`, while the formal grid is pinned at tabpfn 8.0.8. The library
version is a column in every file. **Never concatenate these files with
`results/runs*`**: they are a different library version and a subset of test rows.

Design: eu_cc and banksim, seed 42, `tabpfn_3`, `n_estimators=4`, 50k context.
Conditions C1 (random, natural ratio), C2 (stratified at the 10 % plateau ratio,
same sampler and RNG stream as `results/runs_c2grid`; NOT the formal grid's 50/50
C2) and C4 (cosine retrieval + 10 % ratio, silhouette grouping, same code and RNG
streams as `results/runs` C4). Each cell's context is therefore identical to an
existing grid cell; only the library version differs.

All scored test rows are predicted (metrics files); the readout is taken for all
scored test frauds plus 2,000 seeded random scored legits (`test_row_id` indexes
the scored test arrays, same convention as `results/preds`).

Files per (dataset, condition):
- `{ds}_{cond}.parquet`: one row per readout test row. `attn_fraud_mass` is the
  summed decoder attention on fraud context rows (= the model's pre-temperature
  fraud vote; rows sum to 1 with `attn_legit_mass`). `top10_*` are list columns
  with the ten most-attended context rows as global train row indices, labels
  and weights; `top10_mass` is their summed weight. `group_id` is the C4 test
  cluster (NA under C1/C2); `n_ctx_fraud`/`n_ctx_legit` describe that row's context.
- `{ds}_{cond}_top10.parquet`: the same top-10 in long form (10 rows per test row)
  for readers that do not handle list columns.
- `{ds}_{cond}_metrics.parquet`: PR-AUC, Recall@FPR, ROC-AUC over the full scored
  test set under tabpfn 8.5.0, for the drift check below, plus wall time.

## Library-version drift check (tabpfn 8.5.0 here vs 8.0.8 in the grid)

Same seed, same context (verified: C4 grouping reproduced G=9 on eu_cc and G=2 on
banksim, same silhouette), same scored test set; only the library differs.

| dataset | condition | grid_source | pr_auc_grid | pr_auc_readout | d_pr_auc | r5_grid | r5_readout | d_r5 | wall_s |
|---|---|---|---|---|---|---|---|---|---|
| banksim | C1 | results/runs C1 | 0.9022 | 0.9027 | 0.0005 | 0.9917 | 0.9924 | 0.0007 | 361 |
| banksim | C2 | results/runs_c2grid r=0.10/50k | 0.9028 | 0.9028 | -0.0 | 0.9972 | 0.9972 | 0.0 | 360 |
| banksim | C4 | results/runs C4 r=0.10 | 0.1538 | 0.1542 | 0.0004 | 0.6232 | 0.6232 | 0.0 | 382 |
| eu_cc | C1 | results/runs C1 | 0.8451 | 0.8424 | -0.0027 | 0.9184 | 0.9184 | 0.0 | 649 |
| eu_cc | C2 | results/runs_c2grid r=0.10/50k | 0.8636 | 0.864 | 0.0004 | 0.949 | 0.949 | 0.0 | 649 |
| eu_cc | C4 | results/runs C4 r=0.10 | 0.7576 | 0.7576 | -0.0 | 0.9286 | 0.9286 | 0.0 | 748 |

Largest drift: 0.0027 PR-AUC (eu_cc C1), within the 0.005 tie band used in the
results chapter; Recall@5%FPR is identical in five of six cells. The readout
files therefore describe the same predictions the grid reports. Total wall time
52 min on the RTX 5070 (predict over the full scored set dominates; the readout
subset adds roughly a minute per cell).

## First reading of the numbers (2026-09-20, for the figures still to be made)

Mean attention mass on fraud context rows, by true label of the scored row:

| dataset | condition | legit rows | fraud rows |
|---|---|---|---|
| banksim | C1 | 0.003 | 0.738 |
| banksim | C2 | 0.010 | 0.873 |
| banksim | C4 | 0.516 | 0.966 |
| eu_cc | C1 | 0.000 | 0.705 |
| eu_cc | C2 | 0.001 | 0.796 |
| eu_cc | C4 | 0.009 | 0.827 |

The expected reading in plan_run.md (fraud rows lose fraud mass under C4) does
NOT hold: fraud rows attend to fraud context MORE under C4 than under C2. What
breaks banksim C4 is the legit side: a random legit test row places half its
mass on fraud context, because class-conditional retrieval pulls the 5,000
nearest frauds towards every group centre and those retrieved frauds are the
legit rows' nearest context neighbours. On eu_cc the legit side stays clean
(0.009), matching the near-tie in the grid. Figure 1 should therefore be read
as a legit-row story, not a fraud-row story.
