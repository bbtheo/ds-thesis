"""Launch all C2 (stratified 50/50) runs for FTM models on the formal grid."""
import sys
sys.path.insert(0, ".")

from src.experiment.runner import run
from src.data.schema import DATASETS

FTM_MODELS = ["tabpfn_v2", "tabpfn_25", "tabpfn_26", "tabiclv2"]
SEEDS = [42, 123, 7]

FORMAL_DATASETS = [k for k, v in DATASETS.items() if not v.get("dev_only")]

configs = []
for ds in FORMAL_DATASETS:
    schema = DATASETS[ds]
    n_seeds = schema.get("n_seeds", 3)
    seeds = SEEDS[:n_seeds]
    for seed in seeds:
        for model in FTM_MODELS:
            configs.append({
                "dataset": ds, "model": model, "condition": "C2",
                "seed": seed, "batch_size": 128,
                "k": None, "fraud_ratio": None, "metric": None,
            })

print(f"Total C2 runs to attempt: {len(configs)}")
for i, cfg in enumerate(configs, 1):
    print(f"\n--- [{i}/{len(configs)}] {cfg['dataset']} × {cfg['model']} × seed={cfg['seed']} ---")
    run(cfg)

print("\nAll C2 runs complete.")
