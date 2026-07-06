"""
Run all pending C1 and C2 runs for the formal grid.

Loads each (dataset, seed) split ONCE and reuses it across all models/conditions
(load+split is deterministic in dataset+seed, so this is result-neutral and just
avoids re-reading DuckDB — paysim is 6.3M rows). Skips already-completed runs
(runner is idempotent).
"""
import sys
from pathlib import Path

# Make the project root importable when run as a script (uv run python scripts/...)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.experiment.runner import config_hash, run
from src.experiment.split_util import load_full_split

FORMAL_DATASETS = {
    # ordered by expected inference speed (fast → slow)
    # cc_2025 excluded — degenerate dataset, all models ≈ fraud rate (0.015), not worth compute
    'fifar':   [42],           # 603k rows, 29 features, fixed split
    'banksim': [42, 123, 7],   # 595k rows, 7 features
    'eu_cc':   [42, 123, 7],   # 285k rows, 30 features — slow FTMs
    'baf':     [42, 123, 7],   # 1M rows, 30 features
    'paysim':  [42, 123, 7],   # 6.3M rows — largest, last
}
# v3 license accepted at ux.priorlabs.ai (2026-06-14) — tabpfn_3 now included;
# the idempotent rerun fills in only the pending v3 cells across the grid.
FTM_MODELS  = ['tabiclv2', 'tabpfn_v2', 'tabpfn_25', 'tabpfn_26', 'tabpfn_3']  # fast → slow
GBDT_MODELS = ['xgboost', 'catboost']
# FTM ensemble passes for this run (user-requested; default would be 8).
N_ESTIMATORS = 4
RESULTS_DIR = Path('results/runs')


def configs_for(ds: str, seed: int) -> list[dict]:
    cfgs = []
    for model in GBDT_MODELS:
        cfgs.append({'dataset': ds, 'model': model, 'condition': 'C1',
                     'seed': seed, 'batch_size': 128,
                     'k': None, 'fraud_ratio': None, 'metric': None})
    for model in FTM_MODELS:
        for cond in ['C1', 'C2']:
            cfgs.append({'dataset': ds, 'model': model, 'condition': cond,
                         'seed': seed, 'batch_size': 128,
                         'k': None, 'fraud_ratio': None, 'metric': None,
                         'n_estimators': N_ESTIMATORS})
    return cfgs


def main() -> None:
    # Enumerate all configs, count pending up front.
    all_cfgs = [(ds, seed, cfg)
                for ds, seeds in FORMAL_DATASETS.items()
                for seed in seeds
                for cfg in configs_for(ds, seed)]
    done = {f.stem for f in RESULTS_DIR.glob('*.parquet')}
    n_pending = sum(1 for _, _, c in all_cfgs if config_hash(c) not in done)
    print(f"Pending: {n_pending} / {len(all_cfgs)} total")

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
                print(f"\n=== [{i}/{n_pending}] {cfg['dataset']} × {cfg['model']} × {cfg['condition']} × seed={cfg['seed']} ===", flush=True)
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


if __name__ == "__main__":
    main()
