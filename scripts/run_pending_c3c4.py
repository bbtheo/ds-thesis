"""
Run all pending C3 and C4 (RAP retrieval) runs for the formal grid.

Phase-4 grid driver. Mirrors ``run_pending_c1c2.py``: loads each (dataset, seed)
split ONCE and reuses it across all models/conditions (load+split is deterministic
in dataset+seed, so this is result-neutral), and skips already-completed runs
(the runner is idempotent on config_hash).

Scope (locked in the Phase-3 plan, 2026-06-18):
  * C3/C4 FTMs are ``tabpfn_3`` + ``tabiclv2`` ONLY (their C1/C2 baselines exist,
    so RQ3/RQ4 comparisons stay intact; other FTMs keep C1/C2 only).
  * C3 = mixed kNN retrieval, natural-as-found ratio (1 run per model×split).
  * C4 = class-conditional kNN + ratio-controlled fraud oversampling, swept over
    {0.05, 0.10, 0.20, 0.30, 0.50} (5 runs per model×split).
  * metric = cosine (the plan default; euclidean is a separate A/B, not the grid).
  * n_estimators = 4 (settled grid decision, matches the C1/C2 v4 rerun).

Formal grid units (cc_2025 excluded — degenerate): fifar×1 + (banksim, eu_cc, baf,
paysim)×3 = 13. Per unit: 2 models × (C3 + 5×C4) = 12 runs → 156 total.

For each model, C3 and its 5 C4 ratios run back-to-back so the per-(dataset,seed,
model,metric) grouping/retrieval work stays adjacent (the runner rebuilds the
retrieval index per run; sharing it across ratios is a future optimisation — the
grouping cost is a small fraction of the per-group FTM predict).

Usage
-----
    TABPFN_TOKEN=... uv run python scripts/run_pending_c3c4.py            # run
    uv run python scripts/run_pending_c3c4.py --dry-run                   # enumerate only
"""
import argparse
import sys
from pathlib import Path

# Make the project root importable when run as a script (uv run python scripts/...)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.experiment.runner import config_hash, run
from src.experiment.split_util import load_full_split

FORMAL_DATASETS = {
    # ordered by expected inference speed (fast → slow), matching run_pending_c1c2.py.
    # cc_2025 excluded — degenerate dataset, not worth compute.
    'fifar':   [42],           # 603k rows, 29 features, fixed split
    'banksim': [42, 123, 7],   # 595k rows, 7 features
    'eu_cc':   [42, 123, 7],   # 285k rows, 30 features — slow FTMs
    'baf':     [42, 123, 7],   # 1M rows, 30 features
    'paysim':  [42, 123, 7],   # 6.3M rows — largest, last
}
# C3/C4 are restricted to these two FTMs (phase3-plan.md decision 5).
C3C4_FTM_MODELS = ['tabiclv2', 'tabpfn_3']   # fast → slow
RATIO_SWEEP = (0.05, 0.10, 0.20, 0.30, 0.50)
METRIC = 'cosine'           # plan default; euclidean is a separate A/B, not the grid
N_ESTIMATORS = 4            # settled grid decision (matches the C1/C2 v4 rerun)
RESULTS_DIR = Path('results/runs')


def configs_for(ds: str, seed: int) -> list[dict]:
    """All C3/C4 configs for one (dataset, seed) unit, grouped by model."""
    cfgs = []
    for model in C3C4_FTM_MODELS:
        # C3 — mixed retrieval, natural-as-found ratio (no fraud_ratio).
        cfgs.append({'dataset': ds, 'model': model, 'condition': 'C3',
                     'seed': seed, 'batch_size': None, 'k': None,
                     'fraud_ratio': None, 'metric': METRIC,
                     'n_estimators': N_ESTIMATORS})
        # C4 — ratio sweep.
        for r in RATIO_SWEEP:
            cfgs.append({'dataset': ds, 'model': model, 'condition': 'C4',
                         'seed': seed, 'batch_size': None, 'k': None,
                         'fraud_ratio': r, 'metric': METRIC,
                         'n_estimators': N_ESTIMATORS})
    return cfgs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dry-run', action='store_true',
                    help='enumerate pending runs and exit (no model execution)')
    args = ap.parse_args()

    # Enumerate all configs, count pending up front.
    all_cfgs = [(ds, seed, cfg)
                for ds, seeds in FORMAL_DATASETS.items()
                for seed in seeds
                for cfg in configs_for(ds, seed)]
    done = {f.stem for f in RESULTS_DIR.glob('*.parquet')}
    pending = [(ds, seed, c) for ds, seed, c in all_cfgs if config_hash(c) not in done]
    print(f"Pending: {len(pending)} / {len(all_cfgs)} total")

    if args.dry_run:
        for ds, seed, c in pending:
            r = c['fraud_ratio']
            print(f"  PENDING {config_hash(c)} | {ds} × {c['model']} × {c['condition']}"
                  f"{f' r={r}' if r is not None else ''} × seed={seed}")
        return

    failures = []
    i = 0
    for ds, seeds in FORMAL_DATASETS.items():
        for seed in seeds:
            cfgs = configs_for(ds, seed)
            # Skip the (expensive) load entirely if every config for this unit is done.
            done = {f.stem for f in RESULTS_DIR.glob('*.parquet')}
            if all(config_hash(c) in done for c in cfgs):
                continue
            print(f"\n### loading {ds} seed={seed} (once) ###", flush=True)
            data = load_full_split(ds, seed)
            for cfg in cfgs:
                if config_hash(cfg) in done:
                    continue
                i += 1
                r = cfg['fraud_ratio']
                print(f"\n=== [{i}/{len(pending)}] {cfg['dataset']} × {cfg['model']} × "
                      f"{cfg['condition']}{f' r={r}' if r is not None else ''} × "
                      f"seed={cfg['seed']} ===", flush=True)
                try:
                    run(cfg, data=data)
                except Exception as e:
                    print(f"ERROR: {e}")
                    failures.append((cfg, repr(e)))
                    continue

    if failures:
        print(f"\n{len(failures)} run(s) FAILED:")
        for cfg, err in failures:
            print(f"  FAILED {cfg} -> {err}")
    else:
        print("\nAll done.")

    if failures:
        sys.exit(1)


if __name__ == '__main__':
    main()
