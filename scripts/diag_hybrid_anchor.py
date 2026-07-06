"""
Diagnostic 2: does mixing random "anchor" rows into the retrieved context fix C4?

Hypothesis from diag 1: localized retrieval strips the "obviously legit" anchor
rows, so legit test scores inflate (0.002 -> 0.417) and PR-AUC collapses. Fix
candidate: keep the kNN-retrieved rows but replace a fraction of the legit slot
with RANDOM legit rows (anchors).

Target cell: banksim seed=42, tabiclv2, n_est=4, ratio=0.05 (grid C1=0.908, C4=0.688).

Variants (all at ratio 0.05 unless noted):
  C1               random global context                      (reference 0.908)
  C2               stratified 50/50 global                    (reference 0.909)
  anchor_00        kNN legit + kNN fraud   (== C4_cos)         reproduce 0.688
  anchor_25/50/75  X% random legit anchors + rest kNN legit, kNN fraud
  anchor_100       100% random legit + kNN fraud               (legit fully random)
  pg_random        per-group random legit + random fraud       (per-group control)

If PR-AUC rises toward ~0.9 as anchor fraction grows -> anchors are the fix and
both C3/C4 get rerun with a hybrid context. Saves predictions to results/diag/.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data.loader import load_dataset
from src.data.schema import DATASETS
from src.data.splitter import split, split_info, subsample_test
from src.eval.metrics import compute_metrics
from src.experiment.runner import _PREDICT_CHUNK, _predict_proba_batched
from src.models.ftm import CONTEXT_LIMITS, build_ftm
from src.rap.context import RetrievalContextBuilder
from src.rap.retriever import KNNRetriever

DATASET, SEED, MODEL, NEST, RATIO = "banksim", 42, "tabiclv2", 4, 0.05
OUT = Path("results/diag")
OUT.mkdir(parents=True, exist_ok=True)

# ---- load + split (mirror runner: banksim grouped split by customer) ----
schema = DATASETS[DATASET]
X, y, names = load_dataset(DATASET)
gcol = schema.get("group_split")
groups = X[:, names.index(gcol)] if gcol else None
Xtr, Xte, ytr, yte = split(X, y, seed=SEED, groups=groups)
info = split_info(ytr, yte)
prevalence = info["n_fraud_test"] / info["n_test"]
negcap = schema.get("test_neg_cap")
Xtes, ytes = subsample_test(Xte, yte, seed=SEED, max_negatives=negcap)
subsampled = len(ytes) < info["n_test"]
ctx = min(len(ytr), int(CONTEXT_LIMITS[MODEL] * 1.0))
fidx = np.where(ytr == 1)[0]
lidx = np.where(ytr == 0)[0]
print(f"{DATASET} seed={SEED}: train={len(ytr)} (fraud {len(fidx)}) "
      f"test_scored={len(ytes)} ctx={ctx} ratio={RATIO}")

# ---- grouping (StandardScaler / euclidean KMeans, metric-independent) ----
b = RetrievalContextBuilder(Xtr, ytr, metric="cosine", seed=SEED)
labels, centres, G, sil = b.group_test(b.scale(Xtes))
print(f"grouping: G={G} silhouette={sil:.3f} sizes={np.bincount(labels).tolist()}\n")

scaler = StandardScaler().fit(Xtr)
Xs = scaler.transform(Xtr)
Xtes_s = scaler.transform(Xtes)
rf = KNNRetriever("cosine").fit(Xs[fidx])
rl = KNNRetriever("cosine").fit(Xs[lidx])

model = build_ftm(MODEL, device="cuda", seed=SEED, n_estimators=NEST)
model.ensure_loaded()
chunk = _PREDICT_CHUNK.get(MODEL, 512)
preds = {"y_true": ytes.astype(np.int8)}
summary = []


def metr(p):
    return compute_metrics(ytes, p, prevalence=prevalence if subsampled else None)


def evaluate(name, idx_per_group):
    probs = np.empty(len(ytes), dtype=np.float64)
    for g in range(G):
        idx = idx_per_group[g]
        model.fit(Xtr[idx], ytr[idx])
        m = labels == g
        probs[m] = _predict_proba_batched(model, Xtes[m], batch_size=chunk)
    mm = metr(probs)
    preds[f"p_{name}"] = probs
    yt = ytes.astype(bool)
    summary.append(dict(variant=name, pr_auc=mm["pr_auc"], r1=mm["recall_at_1fpr"],
                        r5=mm["recall_at_5fpr"], roc=mm["roc_auc"],
                        legit_score=float(probs[~yt].mean())))
    print(f"  [{name:10s}] PR-AUC={mm['pr_auc']:.4f}  R@1={mm['recall_at_1fpr']:.3f}  "
          f"R@5={mm['recall_at_5fpr']:.3f}  ROC={mm['roc_auc']:.4f}  "
          f"legit_score={probs[~yt].mean():.3f}", flush=True)


def global_ctx(kind):
    rng = np.random.default_rng(SEED)
    if kind == "C1":
        sel = rng.choice(len(ytr), size=min(ctx, len(ytr)), replace=False)
    else:
        nfh = ctx // 2
        cf = fidx if len(fidx) <= nfh else rng.choice(fidx, nfh, replace=False)
        cl = rng.choice(lidx, ctx - len(cf), replace=False)
        sel = np.concatenate([cf, cl]); rng.shuffle(sel)
    return [sel] * G


def hybrid(anchor_frac, fraud_source="knn"):
    """Per-group: kNN/random fraud + (anchor_frac random legit anchors, rest kNN legit)."""
    nf = int(round(RATIO * ctx)); nl = ctx - nf
    out = []
    for g in range(G):
        centre = Xtes_s[labels == g].mean(0)
        nfu = min(nf, len(fidx))
        if fraud_source == "knn":
            cf = fidx[rf.query(centre, nfu)]
        else:
            cf = np.random.default_rng([SEED, g, 0xF]).choice(fidx, nfu, replace=False)
        n_anchor = int(round(anchor_frac * nl))
        n_retr = nl - n_anchor
        retr_local = rl.query(centre, min(n_retr, len(lidx))) if n_retr > 0 else np.empty(0, np.int64)
        cl_retr = lidx[retr_local]
        if n_anchor > 0:
            mask = np.ones(len(lidx), bool); mask[retr_local] = False
            anchor_local = np.random.default_rng([SEED, g, 0xA]).choice(
                np.flatnonzero(mask), min(n_anchor, int(mask.sum())), replace=False)
            cl_anchor = lidx[anchor_local]
        else:
            cl_anchor = np.empty(0, np.int64)
        idx = np.concatenate([cf, cl_retr, cl_anchor]).astype(np.int64)
        np.random.default_rng([SEED, g]).shuffle(idx)
        out.append(idx)
    return out


print("=" * 90)
print(f"HYBRID-ANCHOR SWEEP — {DATASET} seed={SEED} {MODEL} n_est={NEST} ratio={RATIO}")
print("legit_score = mean predicted prob on TRUE legits (C1 ref ~0.002; high = anchor loss)")
print("=" * 90)
evaluate("C1", global_ctx("C1"))
evaluate("C2", global_ctx("C2"))
evaluate("anchor_00", hybrid(0.0))
evaluate("anchor_25", hybrid(0.25))
evaluate("anchor_50", hybrid(0.50))
evaluate("anchor_75", hybrid(0.75))
evaluate("anchor_100", hybrid(1.0))
evaluate("pg_random", hybrid(1.0, fraud_source="random"))

pred_path = OUT / f"{DATASET}_seed{SEED}_{MODEL}_hybrid_preds.parquet"
pd.DataFrame(preds).to_parquet(pred_path, index=False)
print("\n" + "=" * 90)
print("SUMMARY (PR-AUC vs anchor fraction)")
print("=" * 90)
print(pd.DataFrame(summary).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print(f"\nsaved predictions -> {pred_path}")
