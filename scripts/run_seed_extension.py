"""
Seed extension for CI repetitions on the two SOTA FTMs (2026-07-18).

Extends tabpfn_3 + tabiclv2 × {C1, C2} on the 5 formal datasets to 10 seeds
per cell so per-dataset bootstrap CIs and paired C2−C1 deltas can be computed
over repetitions. The 10-seed list is fixed a priori (before any extension run)
and identical for every dataset; all seeds are reported, none dropped.

Iterates seed-major (datasets fast → slow within a seed, C2 before C1 within a
model) so that when the wall-clock deadline hits, finished seeds form complete
C1/C2 pairs across all datasets and the C2 half of any partial unit exists.
Loads each (dataset, seed) split once, like run_pending_c1c2.py. Idempotent:
completed runs are skipped via config-hash parquets, safe to kill and restart.
"""
import argparse
import sys
import time
from pathlib import Path

# Make the project root importable when run as a script (uv run python scripts/...)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.experiment.runner import config_hash, run
from src.experiment.split_util import load_full_split

# Fixed a-priori seed list. 42/123/7 already exist for banksim/eu_cc/baf/paysim
# (hash-skipped); fifar had seed 42 only, so 123/7 are new there (run last).
SEEDS = [0, 1, 2, 3, 4, 5, 6, 42, 123, 7]
DATASETS = ['paysim', 'banksim', 'fifar', 'baf', 'eu_cc']  # fast → slow per run
MODELS = ['tabiclv2', 'tabpfn_3']                          # fast → slow
CONDITIONS = ['C2', 'C1']  # C2 first: the SOTA half survives a mid-unit deadline
N_ESTIMATORS = 4           # grid standard — runner default 8 is NOT comparable
RESULTS_DIR = Path('results/runs')


def configs_for(ds: str, seed: int) -> list[dict]:
    return [{'dataset': ds, 'model': model, 'condition': cond,
             'seed': seed, 'batch_size': 128,
             'k': None, 'fraud_ratio': None, 'metric': None,
             'n_estimators': N_ESTIMATORS}
            for model in MODELS for cond in CONDITIONS]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true',
                    help='print pending count and exit')
    ap.add_argument('--deadline-ts', type=float, default=None,
                    help='unix timestamp; start no new run at/after this time')
    args = ap.parse_args()

    all_cfgs = [(ds, seed, cfg)
                for seed in SEEDS
                for ds in DATASETS
                for cfg in configs_for(ds, seed)]
    done = {f.stem for f in RESULTS_DIR.glob('*.parquet')}
    n_pending = sum(1 for _, _, c in all_cfgs if config_hash(c) not in done)
    print(f"pending: {n_pending} / {len(all_cfgs)} total")
    if args.dry_run:
        return

    failures = []
    i = 0
    for seed in SEEDS:
        for ds in DATASETS:
            cfgs = configs_for(ds, seed)
            done = {f.stem for f in RESULTS_DIR.glob('*.parquet')}
            if all(config_hash(c) in done for c in cfgs):
                continue
            print(f"\n### loading {ds} seed={seed} (once) ###", flush=True)
            data = load_full_split(ds, seed)
            for cfg in cfgs:
                if config_hash(cfg) in done:
                    continue
                if args.deadline_ts is not None and time.time() >= args.deadline_ts:
                    print(f"DEADLINE reached before {ds}×{cfg['model']}×"
                          f"{cfg['condition']}×seed={seed} — exiting cleanly",
                          flush=True)
                    return
                i += 1
                print(f"\n=== [{i}/{n_pending}] {ds} × {cfg['model']} × "
                      f"{cfg['condition']} × seed={seed} ===", flush=True)
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
        sys.exit(1)
    print("\nAll done.")


if __name__ == "__main__":
    main()
