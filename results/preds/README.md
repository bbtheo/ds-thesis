# results/preds — per-row test predictions (seed-42 grid slice)

One parquet per run, named `{run_id}.parquet`, where `run_id` is the config hash
of the *formal-grid* cell it reproduces (same id as the file in `results/runs`).

## Columns

| column | type | meaning |
|---|---|---|
| `row_idx` | int64 | 0..n_scored-1, position in the scored test arrays (deterministic given the seed) |
| `y_true` | int8 | true label (1 = fraud) |
| `score` | float64 | model's predicted P(fraud) |
| `dataset` | str | dataset key (constant per file) |
| `model` | str | model key (constant per file) |
| `condition` | str | context condition (constant per file) |
| `seed` | int | split/context seed (constant per file) |

The constant columns are repeated per row so the files can simply be
concatenated downstream, with no join back to the metrics rows.

## Protocol

Produced by `scripts/run_confidence_preds.py`, which calls the same
`src/experiment/runner.py::run` as the formal grid: same seeded split, same
negative-capped test subsample (all positives kept; negatives capped per the
schema's `test_neg_cap`), same stratified 50/50 context sampling, `n_estimators=4`.
Cells: {C2} x {tabiclv2, tabpfn_3} and {C1} x {xgboost, catboost} (full
training set, no context limit, as in the grid) x {eu_cc, banksim, paysim,
fifar, baf}, seed 42.
Metrics rows for these reruns go to `results/runs_conf/` — never mix them into
grid pivots; use `results/runs` for all reported grid numbers.

Note that `score` covers only the SCORED test rows. On negative-subsampled
datasets that is every fraud plus a capped sample of legitimate rows, so
score-rank statistics over legitimate rows are estimates from that sample.

## Percentile definition (downstream)

For each fraud row, its percentile is the share of scored **legitimate** test
rows with score **strictly below** the fraud's score. Ties count against the
fraud, which is the conservative reading: a fraud tied with a legitimate row is
not ranked above it.
