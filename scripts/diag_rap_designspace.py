"""
Diagnostic 3: exhaustive RAP design-space sweep — can ANY retrieval design beat random?

Levers explored:
  granularity   : forced #groups G (2..64). G large => each centroid is genuinely
                  local (approximates per-batch / per-test-point RAP).
  legit source  : kNN-local vs random-global
  fraud source  : kNN-local vs random-global vs ALL train frauds (global signal)
  ratio         : fraud fraction (when not using all frauds)
  diversify     : retrieve d*context nearest, then random-subsample (locality+diversity)

Key designs requested: "global/all fraud + kNN-local legit", and the granularity
sweep (per-batch ≈ fine grouping). References: C1 random, C2 strat-50/50.

Run on a cell with headroom (baf: C1~0.16, G=10) as the real test; banksim as the
fast sanity cell. Usage:
    uv run python scripts/diag_rap_designspace.py --dataset banksim
    uv run python scripts/diag_rap_designspace.py --dataset baf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.data.loader import load_dataset
from src.data.schema import DATASETS
from src.data.splitter import split, split_info, subsample_test
from src.eval.metrics import compute_metrics
from src.experiment.runner import _PREDICT_CHUNK, _predict_proba_batched
from src.models.ftm import CONTEXT_LIMITS, build_ftm
from src.rap.retriever import KNNRetriever

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="banksim")
ap.add_argument("--model", default="tabiclv2")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--predict-chunk", type=int, default=None,
                help="override _PREDICT_CHUNK (result-neutral; uses spare VRAM)")
ap.add_argument("--chunk-probe", action="store_true",
                help="time C1 predict at several chunk sizes (peak VRAM + wall-time), then exit")
ap.add_argument("--only", default=None,
                help="comma-separated substrings; run only matching variants (refs always run)")
args = ap.parse_args()
_ONLY = [s for s in (args.only.split(",") if args.only else []) if s]
DATASET, MODEL, SEED, NEST = args.dataset, args.model, args.seed, 4

import time as _time
try:
    import torch
except Exception:  # pragma: no cover
    torch = None


def _peak_vram_mb():
    if torch is not None and torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 ** 2
    return float("nan")


def _reset_vram():
    if torch is not None and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
OUT = Path("results/diag"); OUT.mkdir(parents=True, exist_ok=True)

schema = DATASETS[DATASET]
X, y, names = load_dataset(DATASET)
gcol = schema.get("group_split")
groups = X[:, names.index(gcol)] if gcol else None
if schema.get("fixed_split"):
    from src.data.loader import load_dataset_split
    Xtr, ytr, Xte, yte, _ = load_dataset_split(DATASET)
else:
    Xtr, Xte, ytr, yte = split(X, y, seed=SEED, groups=groups)
info = split_info(ytr, yte)
prevalence = info["n_fraud_test"] / info["n_test"]
negcap = schema.get("test_neg_cap")
Xtes, ytes = subsample_test(Xte, yte, seed=SEED, max_negatives=negcap)
subsampled = len(ytes) < info["n_test"]
ctx = min(len(ytr), int(CONTEXT_LIMITS[MODEL] * 1.0))
fidx = np.where(ytr == 1)[0]
lidx = np.where(ytr == 0)[0]
print(f"{DATASET} seed={SEED} {MODEL}: train={len(ytr)} (fraud {len(fidx)}) "
      f"test_scored={len(ytes)} (fraud {int(ytes.sum())}) ctx={ctx} prev={prevalence:.4f}")

scaler = StandardScaler().fit(Xtr)
Xs = scaler.transform(Xtr)
Xtes_s = scaler.transform(Xtes)
rf = KNNRetriever("cosine").fit(Xs[fidx])
rl = KNNRetriever("cosine").fit(Xs[lidx])

model = build_ftm(MODEL, device="cuda", seed=SEED, n_estimators=NEST)
model.ensure_loaded()
chunk = args.predict_chunk or _PREDICT_CHUNK.get(MODEL, 512)
print(f"predict_chunk={chunk}" + (" (overridden)" if args.predict_chunk else " (default)"))
summary = []
_group_cache: dict[int, tuple] = {}


def metr(p):
    return compute_metrics(ytes, p, prevalence=prevalence if subsampled else None)


def grouping(k):
    if k in _group_cache:
        return _group_cache[k]
    if k <= 1:
        lab = np.zeros(len(Xtes), int); centres = Xtes_s.mean(0, keepdims=True)
    else:
        km = KMeans(n_clusters=k, random_state=SEED, n_init=10).fit(Xtes_s)
        present = np.unique(km.labels_)
        remap = {g: i for i, g in enumerate(present)}
        lab = np.array([remap[g] for g in km.labels_])
        centres = np.vstack([Xtes_s[lab == i].mean(0) for i in range(len(present))])
    _group_cache[k] = (lab, centres)
    return lab, centres


def build_ctx(lab, centres, legit_src, fraud_src, ratio, fraud_all, diversify):
    G = len(centres)
    nf = len(fidx) if fraud_all else int(round(ratio * ctx))
    nl = ctx - nf
    out = []
    for g in range(G):
        c = centres[g]
        nfu = min(nf, len(fidx))
        if fraud_all:
            cf = fidx
        elif fraud_src == "knn":
            cand = rf.query(c, min(diversify * nfu, len(fidx)))
            cf = fidx[np.random.default_rng([SEED, g, 1]).choice(len(cand), nfu, replace=False)] \
                if diversify > 1 else fidx[cand[:nfu]]
        else:
            cf = np.random.default_rng([SEED, g, 0xF]).choice(fidx, nfu, replace=False)
        nlt = min(nl, len(lidx))
        if legit_src == "knn":
            cand = rl.query(c, min(diversify * nlt, len(lidx)))
            cl = lidx[np.random.default_rng([SEED, g, 2]).choice(len(cand), nlt, replace=False)] \
                if diversify > 1 else lidx[cand[:nlt]]
        else:
            cl = np.random.default_rng([SEED, g, 0xA]).choice(lidx, nlt, replace=False)
        idx = np.concatenate([np.asarray(cf), np.asarray(cl)]).astype(np.int64)
        np.random.default_rng([SEED, g]).shuffle(idx)
        out.append(idx)
    return out, int(round(nf))


def evaluate(name, idx_per_group, lab):
    _reset_vram()
    t0 = _time.perf_counter()
    probs = np.empty(len(ytes), dtype=np.float64)
    for g in range(len(idx_per_group)):
        idx = idx_per_group[g]
        model.fit(Xtr[idx], ytr[idx])
        m = lab == g
        if m.any():
            probs[m] = _predict_proba_batched(model, Xtes[m], batch_size=chunk)
        # Release per-fit VRAM so high-G variants (many sequential refits) don't
        # accumulate cached blocks and hard-fault. Result-neutral.
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
    dt = _time.perf_counter() - t0
    mm = metr(probs)
    yt = ytes.astype(bool)
    summary.append(dict(variant=name, G=len(idx_per_group), pr_auc=mm["pr_auc"],
                        r1=mm["recall_at_1fpr"], r5=mm["recall_at_5fpr"],
                        roc=mm["roc_auc"], legit=float(probs[~yt].mean()),
                        wall_s=dt, peak_mb=_peak_vram_mb()))
    print(f"  [{name:22s}] G={len(idx_per_group):>2} PR-AUC={mm['pr_auc']:.4f} "
          f"R@1={mm['recall_at_1fpr']:.3f} ROC={mm['roc_auc']:.4f} "
          f"legit={probs[~yt].mean():.3f} {dt:.0f}s", flush=True)


def global_ref(kind):
    rng = np.random.default_rng(SEED)
    if kind == "C1":
        sel = rng.choice(len(ytr), size=min(ctx, len(ytr)), replace=False)
    else:
        nfh = ctx // 2
        cf = fidx if len(fidx) <= nfh else rng.choice(fidx, nfh, replace=False)
        cl = rng.choice(lidx, ctx - len(cf), replace=False)
        sel = np.concatenate([cf, cl]); rng.shuffle(sel)
    return [sel]


# (name, k, legit, fraud, ratio, fraud_all, diversify)
VARIANTS = [
    ("g2_randL_randF_r05",   2, "random", "random", 0.05, False, 1),  # per-grp random ctrl
    ("g32_randL_randF_r05", 32, "random", "random", 0.05, False, 1),  # fine random ctrl
    # granularity sweep, kNN both, r05
    ("g2_knnL_knnF_r05",     2, "knn", "knn", 0.05, False, 1),
    ("g8_knnL_knnF_r05",     8, "knn", "knn", 0.05, False, 1),
    ("g16_knnL_knnF_r05",   16, "knn", "knn", 0.05, False, 1),
    ("g32_knnL_knnF_r05",   32, "knn", "knn", 0.05, False, 1),
    ("g64_knnL_knnF_r05",   64, "knn", "knn", 0.05, False, 1),
    # global/random fraud + kNN-local legit (user request)
    ("g2_knnL_randF_r05",    2, "knn", "random", 0.05, False, 1),
    ("g32_knnL_randF_r05",  32, "knn", "random", 0.05, False, 1),
    # ALL train frauds (global fraud signal) + kNN-local legit (user request)
    ("g2_knnL_allF",         2, "knn", "knn", 0.05, True, 1),
    ("g32_knnL_allF",       32, "knn", "knn", 0.05, True, 1),
    # local fraud + random legit
    ("g32_randL_knnF_r05",  32, "random", "knn", 0.05, False, 1),
    # diversified retrieval (retrieve 5x, subsample) — locality + diversity
    ("g2_div5_knnL_knnF",    2, "knn", "knn", 0.05, False, 5),
    ("g32_div5_knnL_knnF",  32, "knn", "knn", 0.05, False, 5),
    # higher ratio at fine granularity
    ("g32_knnL_knnF_r20",   32, "knn", "knn", 0.20, False, 1),
    # extra-fine granularity + user's ideas at fine G (does the recovery ceiling beat random?)
    ("g64_knnL_knnF_r05",   64, "knn", "knn", 0.05, False, 1),
    ("g128_knnL_knnF_r05", 128, "knn", "knn", 0.05, False, 1),
    ("g64_knnL_allF",       64, "knn", "knn", 0.05, True, 1),
    ("g64_knnL_randF_r05",  64, "knn", "random", 0.05, False, 1),
]
# de-duplicate by name, keeping first occurrence (g64_knnL_knnF_r05 appears twice)
_seen = set()
VARIANTS = [v for v in VARIANTS if not (v[0] in _seen or _seen.add(v[0]))]

if args.chunk_probe:
    sel = global_ref("C1")[0]
    model.fit(Xtr[sel], ytr[sel])
    _predict_proba_batched(model, Xtes[:4096], batch_size=2048)  # warmup
    print(f"\n=== CHUNK PROBE (C1 context, predict {len(ytes)} test rows, warmed up) ===", flush=True)
    print(f"{'chunk':>8} {'wall_s':>8} {'peak_MB':>9} {'rows/s':>9}  note")
    for cz in (16384, 32768, 49152, 65536):
        _reset_vram()
        try:
            t0 = _time.perf_counter()
            _predict_proba_batched(model, Xtes, batch_size=cz)
            dt = _time.perf_counter() - t0
            note = "<=1 batch (capped@n_test)" if cz >= len(ytes) else ""
            print(f"{cz:>8} {dt:>8.1f} {_peak_vram_mb():>9.0f} {len(ytes)/dt:>9.0f}  {note}", flush=True)
        except RuntimeError as e:
            print(f"{cz:>8}  OOM/ERROR: {str(e)[:70]}", flush=True)
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
    sys.exit(0)

print("=" * 96)
print(f"RAP DESIGN-SPACE SWEEP — {DATASET} seed={SEED} {MODEL} n_est={NEST}")
print("=" * 96)
evaluate("C1_ref", global_ref("C1"), np.zeros(len(ytes), int))
evaluate("C2_ref", global_ref("C2"), np.zeros(len(ytes), int))
for name, k, ls, fs, r, fa, dv in VARIANTS:
    if _ONLY and not any(s in name for s in _ONLY):
        continue
    lab, centres = grouping(k)
    idx, _ = build_ctx(lab, centres, ls, fs, r, fa, dv)
    evaluate(name, idx, lab)

df = pd.DataFrame(summary).sort_values("pr_auc", ascending=False)
df.to_parquet(OUT / f"designspace_{DATASET}_{MODEL}.parquet", index=False)
print("\n" + "=" * 96)
print(f"SUMMARY (sorted by PR-AUC) — {DATASET}  [C1/C2 are the bar to beat]")
print("=" * 96)
print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
best = df.iloc[0]
print(f"\nBEST: {best.variant}  PR-AUC={best.pr_auc:.4f}  (C1={df[df.variant=='C1_ref'].pr_auc.iloc[0]:.4f})")
