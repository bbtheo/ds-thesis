"""
Per-row prediction dump for the model-confidence appendix (2026-09-19).

The formal grid records only aggregate metrics, which cannot answer "how
confidently does a SOTA tabular foundation model score the frauds it sees?".
This driver re-executes a small, already-run slice of the grid — the two SOTA
FTMs (tabiclv2, tabpfn_3) under condition C2, n_estimators=4, plus the two GBDT
baselines (xgboost, catboost) under C1 on the full training set, seed 42, on the
5 informative datasets — and persists every scored test row's probability to
``results/preds/{run_id}.parquet``.

The protocol is the grid protocol, not a new one: the same ``runner.run`` code
path, the same seeded split, the same negative-capped test subsample, the same
stratified 50/50 context sampling. Only the output location differs
(``results/runs_conf`` for the metrics row, so the grid parquets in
``results/runs`` are never touched and the runner's idempotent skip on the
existing grid file does not block the rerun). The run_id is therefore identical
to the grid cell's run_id, and each run is checked against it: PR-AUC,
Recall@1%FPR and Recall@5%FPR are compared and the deltas printed. GPU inference
is not guaranteed bit-reproducible, so a small delta is expected; the appendix
reports it. Only |delta PR-AUC| > 1e-3 is treated as a failure.

Known non-reproductions (2026-09-19): all five tabiclv2 cells drift by <= 3.3e-4
(GPU nondeterminism); catboost on fifar and baf drifts by <= 1e-3 PR-AUC and
<= 0.0063 recall@FPR because those grid rows were fitted at CatBoost's default
24 threads before ``thread_count=1`` was pinned in ``gbdt.py`` (a 24-thread
refit reproduces the grid row exactly). tabpfn_3, xgboost and the other catboost
cells reproduce to float noise.

Usage
-----
    uv run python scripts/run_confidence_preds.py --dry-run
    uv run python scripts/run_confidence_preds.py
"""
import argparse
import os
import sys
from pathlib import Path

# Make the project root importable when run as a script (uv run python scripts/...)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.eval.metrics import compute_metrics
from src.experiment.runner import config_hash, run
from src.experiment.split_util import load_full_split

_ROOT = Path(__file__).resolve().parent.parent

DATASETS = ['paysim', 'banksim', 'fifar', 'baf', 'eu_cc']  # fast → slow per run
MODELS = ['tabiclv2', 'tabpfn_3']                          # fast → slow
CONDITION = 'C2'
# GBDT baselines (added 2026-09-19): full-training-set fits under C1, exactly
# as in the formal grid (GBDTs never run under C2; the context limit is a TFM
# property). Their config carries no n_estimators key, matching
# run_pending_c1c2.py so the run_id equals the grid cell's.
GBDT_MODELS = ['xgboost', 'catboost']                      # fast → slow
GBDT_CONDITION = 'C1'
SEED = 42
N_ESTIMATORS = 4           # grid standard — runner default 8 is NOT comparable
# Anchored to the repo root, not the cwd: this driver is launched detached
# (systemd-run), where the working directory is not the repo unless set.
RESULTS_DIR = _ROOT / 'results/runs_conf'  # metrics rows (grid rows stay untouched)
PREDS_DIR = _ROOT / 'results/preds'        # per-row scores
GRID_DIR = _ROOT / 'results/runs'          # the original grid rows, for comparison

# Max tolerated |pr_auc - grid pr_auc|. 1e-6 is the aspiration (identical inputs,
# identical code path); 1e-3 allows for non-deterministic GPU kernel reductions.
PR_AUC_TOL = 1e-3

README = """\
# results/preds — per-row test predictions (confidence appendix)

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
"""


def configs() -> list[tuple[str, dict]]:
    out: list[tuple[str, dict]] = []
    for ds in DATASETS:
        for model in MODELS:
            out.append((ds, {'dataset': ds, 'model': model, 'condition': CONDITION,
                             'seed': SEED, 'batch_size': 128,
                             'k': None, 'fraud_ratio': None, 'metric': None,
                             'n_estimators': N_ESTIMATORS}))
        for model in GBDT_MODELS:
            out.append((ds, {'dataset': ds, 'model': model,
                             'condition': GBDT_CONDITION,
                             'seed': SEED, 'batch_size': 128,
                             'k': None, 'fraud_ratio': None, 'metric': None}))
    return out


def _pending(cfg: dict) -> bool:
    """A cell is done when its PREDS file exists (the reason this study runs)."""
    return not (PREDS_DIR / f"{config_hash(cfg)}.parquet").exists()


def _check_reproduction(cfg: dict) -> float | None:
    """Compare the rerun against its formal-grid row; return |delta pr_auc| or None."""
    h = config_hash(cfg)
    new_path = RESULTS_DIR / f"{h}.parquet"
    grid_path = GRID_DIR / f"{h}.parquet"
    new = pd.read_parquet(new_path).iloc[0]

    # Recompute PR-AUC straight from the saved predictions, with the same
    # prevalence correction the runner applies, to confirm the dump is faithful
    # (not merely that two metrics rows agree).
    preds = pd.read_parquet(PREDS_DIR / f"{h}.parquet")
    prevalence = (new['n_fraud_test'] / new['n_test']
                  if new['test_subsample_rule'] != 'full' else None)
    from_preds = compute_metrics(
        preds['y_true'].to_numpy(), preds['score'].to_numpy(), prevalence=prevalence
    )['pr_auc']
    print(f"  pr_auc recomputed from preds: {from_preds:.6f} "
          f"(row {new['pr_auc']:.6f}, delta {abs(from_preds - new['pr_auc']):.2e})",
          flush=True)

    if not grid_path.exists():
        print(f"  WARNING: no grid row at {grid_path} — cannot verify reproduction",
              flush=True)
        return None

    old = pd.read_parquet(grid_path).iloc[0]
    for col in ('pr_auc', 'recall_at_1fpr', 'recall_at_5fpr'):
        print(f"  {col:16s} grid={old[col]:.6f}  rerun={new[col]:.6f}  "
              f"delta={abs(new[col] - old[col]):.3e}", flush=True)
    return abs(float(new['pr_auc']) - float(old['pr_auc']))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true',
                    help='print pending count and exit')
    args = ap.parse_args()

    all_cfgs = configs()
    pending = [(ds, c) for ds, c in all_cfgs if _pending(c)]
    print(f"pending: {len(pending)} / {len(all_cfgs)}", flush=True)
    if args.dry_run:
        return

    # The gated TabPFN checkpoints authenticate from TABPFN_TOKEN (or a cached
    # token file). Detached launchers do not inherit the shell's exports, and the
    # failure would otherwise surface only when the first tabpfn_3 cell is reached,
    # hours in. Fail now instead. TabICL (BSD-3) needs no token.
    needs_token = any(c['model'].startswith('tabpfn') for _, c in pending)
    have_token = bool(os.environ.get('TABPFN_TOKEN')) or any(
        p.exists() for p in (Path.home() / '.cache/tabpfn/auth_token',
                             Path.home() / '.tabpfn/token')
    )
    if needs_token and not have_token:
        print("ERROR: TABPFN_TOKEN is not set and no cached token file exists; "
              "the tabpfn_3 cells cannot authenticate. Launch with the token in "
              "the environment (systemd-run --setenv=TABPFN_TOKEN=...).", flush=True)
        sys.exit(1)

    PREDS_DIR.mkdir(parents=True, exist_ok=True)
    (PREDS_DIR / 'README.md').write_text(README)

    failures: list[tuple[dict, str]] = []
    drift: list[tuple[dict, float]] = []
    i = 0
    for ds in DATASETS:
        cfgs = [c for d, c in pending if d == ds]
        if not cfgs:
            continue
        print(f"\n### loading {ds} seed={SEED} (once) ###", flush=True)
        try:
            data = load_full_split(ds, SEED)
        except Exception as e:
            print(f"ERROR loading {ds}: {e}", flush=True)
            failures.extend((c, f"load failed: {e!r}") for c in cfgs)
            continue
        for cfg in cfgs:
            i += 1
            print(f"\n=== [{i}/{len(pending)}] {ds} × {cfg['model']} × "
                  f"{cfg['condition']} × seed={SEED} ===", flush=True)
            # The runner writes the metrics row before the preds file, so a crash
            # between the two leaves a row whose preds are missing — and the run
            # would then be skipped as "already done" forever. Clear it first.
            stale = RESULTS_DIR / f"{config_hash(cfg)}.parquet"
            # Hard guard: this path must never resolve into the formal grid. The
            # run_id is shared with the grid cell, so a mis-set RESULTS_DIR would
            # delete grid results rather than this study's rerun row.
            assert stale.resolve().parent != GRID_DIR.resolve(), (
                f"refusing to delete inside the formal grid: {stale}"
            )
            if stale.exists():
                print(f"  removing metrics row with no preds: {stale}", flush=True)
                stale.unlink()
            try:
                run(cfg, data=data, results_dir=RESULTS_DIR, preds_dir=PREDS_DIR)
                delta = _check_reproduction(cfg)
            except Exception as e:
                print(f"ERROR: {e}", flush=True)
                failures.append((cfg, repr(e)))
                continue
            # Collect tolerance violations rather than raising immediately, so a
            # single drifting cell does not cost the remaining runs' GPU time.
            if delta is not None and delta > PR_AUC_TOL:
                drift.append((cfg, delta))
                print(f"  DRIFT: |delta pr_auc|={delta:.3e} > tol {PR_AUC_TOL:g}",
                      flush=True)

    if failures:
        print(f"\n{len(failures)} run(s) FAILED:")
        for cfg, err in failures:
            print(f"  FAILED {cfg} -> {err}")
    if drift:
        # Everything is written by this point; the raise only flags that a rerun
        # did not reproduce its grid cell within tolerance.
        msg = "; ".join(f"{c['dataset']}×{c['model']}: {d:.3e}" for c, d in drift)
        raise RuntimeError(
            f"{len(drift)} run(s) exceeded the PR-AUC reproduction tolerance "
            f"({PR_AUC_TOL:g}): {msg}"
        )
    if failures:
        sys.exit(1)
    print("\nAll done.")


if __name__ == "__main__":
    main()
