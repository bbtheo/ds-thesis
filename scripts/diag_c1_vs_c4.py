"""
Diagnostic: why does C4 (RAP) collapse vs C1 on banksim?

Target cell: banksim seed=42, tabiclv2, n_estimators=4 (G=2, silhouette 0.857;
grid stored C1=0.908, C4 r0.05=0.688). One FTM is built once and reused; each
variant builds a per-group context, fits, predicts the scored test set, and we
SAVE the per-row predicted probabilities so C1 vs C4 can be compared directly.

Variants (all per-group except C1/C2 which are single global contexts):
  C1            random subsample            (reproduce baseline)
  C2            stratified 50/50            (reproduce balanced baseline)
  C4_cos        cosine retrieval, StandardScaler   (reproduce grid ~0.688)
  C4_euc        euclidean retrieval, StandardScaler (does the metric fix it?)
  C4_cos_rndF   cosine legit kNN + RANDOM frauds    (control: isolate fraud retrieval)
  C4_euc_robust euclidean retrieval, RobustScaler   (does normalisation fix it?)

The controls separate "kNN fraud selection is genuinely bad" (real finding) from
"wiring/metric/scaling artefact" (fixable). Saves predictions to results/diag/.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler

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

# ---------------------------------------------------------------- load + split
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
      f"test_scored={len(ytes)} (fraud {int(ytes.sum())}) ctx={ctx}")


def metr(p):
    m = compute_metrics(ytes, p, prevalence=prevalence if subsampled else None)
    return m


# ------------------------------------------------- grouping (StandardScaler/euclidean KMeans)
b_cos = RetrievalContextBuilder(Xtr, ytr, metric="cosine", seed=SEED)
b_euc = RetrievalContextBuilder(Xtr, ytr, metric="euclidean", seed=SEED)
Xte_std = b_cos.scale(Xtes)
labels, centres, G, sil = b_cos.group_test(Xte_std)
print(f"grouping: G={G} silhouette={sil:.3f} sizes={np.bincount(labels).tolist()}\n")

# ================================================================ CPU retrieval diagnostics
print("=" * 92)
print("RETRIEVAL QUALITY — group 0 frauds: are kNN-retrieved frauds a degenerate clump?")
print("  d_to_fraudmean: dist of retrieved frauds to GLOBAL fraud centroid (random≈baseline)")
print("  spread:         mean dist of retrieved frauds to THEIR OWN centroid (low = clumped)")
print("=" * 92)
nf = int(round(RATIO * ctx))
diag_rows = []
fraud_sets = {}
for sname, scaler in [("Standard", StandardScaler()), ("Robust", RobustScaler())]:
    Xs = scaler.fit(Xtr).transform(Xtr)
    Xtes_s = scaler.transform(Xtes)
    centre0 = Xtes_s[labels == 0].mean(0)
    gfmean = Xs[fidx].mean(0)
    spread_all = np.linalg.norm(Xs[fidx] - gfmean, axis=1).mean()
    d_rand = np.linalg.norm(Xs[fidx] - gfmean, axis=1).mean()  # random frauds ~ this
    for metric in ("cosine", "euclidean"):
        rf = KNNRetriever(metric).fit(Xs[fidx])
        fr = fidx[rf.query(centre0, min(nf, len(fidx)))]
        fraud_sets[(sname, metric)] = set(fr.tolist())
        d_retr = np.linalg.norm(Xs[fr] - gfmean, axis=1).mean()
        spread_retr = np.linalg.norm(Xs[fr] - Xs[fr].mean(0), axis=1).mean()
        diag_rows.append(dict(scaler=sname, metric=metric,
                              d_retr=d_retr, d_rand=d_rand,
                              spread_retr=spread_retr, spread_all=spread_all,
                              clump_ratio=spread_retr / spread_all))
dd = pd.DataFrame(diag_rows)
print(dd.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


def jac(a, b):
    return len(a & b) / len(a | b) if (a | b) else 1.0


print("\nfraud-set Jaccard overlap (which rows each setting pulls):")
print(f"  Standard cosine vs euclidean : {jac(fraud_sets[('Standard','cosine')], fraud_sets[('Standard','euclidean')]):.3f}")
print(f"  Standard vs Robust (euclidean): {jac(fraud_sets[('Standard','euclidean')], fraud_sets[('Robust','euclidean')]):.3f}")

# Where do the scored-test rows sit vs the context? distance-to-centre distributions.
print("\nCONTEXT vs TEST coverage (StandardScaler, group 0): is the context a pinpoint vs spread test?")
Xs = StandardScaler().fit(Xtr).transform(Xtr)
Xtes_s = StandardScaler().fit(Xtr).transform(Xtes)
centre0 = Xtes_s[labels == 0].mean(0)
rl = KNNRetriever("cosine").fit(Xs[lidx])
nl = ctx - nf
ctx_legit = lidx[rl.query(centre0, min(nl, len(lidx)))]
d_ctx = np.linalg.norm(Xs[ctx_legit] - centre0, axis=1)
d_test = np.linalg.norm(Xtes_s[labels == 0] - centre0, axis=1)
print(f"  context legit dist-to-centre: median {np.median(d_ctx):.2f}  p95 {np.percentile(d_ctx,95):.2f}")
print(f"  test   rows  dist-to-centre: median {np.median(d_test):.2f}  p95 {np.percentile(d_test,95):.2f}")
print(f"  -> {(d_test > np.percentile(d_ctx,95)).mean()*100:.0f}% of test rows lie beyond the context's p95 radius")


# ================================================================ FTM variants (save preds)
model = build_ftm(MODEL, device="cuda", seed=SEED, n_estimators=NEST)
model.ensure_loaded()
chunk = _PREDICT_CHUNK.get(MODEL, 512)
preds = {"y_true": ytes.astype(np.int8)}
summary = []


def run_variant(name, idx_per_group):
    """idx_per_group: list of global-train index arrays, one per group g=0..G-1."""
    probs = np.empty(len(ytes), dtype=np.float64)
    for g in range(G):
        idx = idx_per_group[g]
        model.fit(Xtr[idx], ytr[idx])
        m = labels == g
        probs[m] = _predict_proba_batched(model, Xtes[m], batch_size=chunk)
    mm = metr(probs)
    preds[f"p_{name}"] = probs
    summary.append(dict(variant=name, pr_auc=mm["pr_auc"], r1=mm["recall_at_1fpr"],
                        r5=mm["recall_at_5fpr"], roc=mm["roc_auc"]))
    print(f"  [{name:14s}] PR-AUC={mm['pr_auc']:.4f}  R@1={mm['recall_at_1fpr']:.3f}  "
          f"R@5={mm['recall_at_5fpr']:.3f}  ROC={mm['roc_auc']:.4f}", flush=True)
    return probs


def global_context(kind):
    """C1/C2 single global context (same for every group)."""
    rng = np.random.default_rng(SEED)
    if kind == "C1":
        sel = rng.choice(len(ytr), size=min(ctx, len(ytr)), replace=False)
    else:  # C2 stratified 50/50
        nfh = ctx // 2
        cf = fidx if len(fidx) <= nfh else rng.choice(fidx, nfh, replace=False)
        cl = rng.choice(lidx, ctx - len(cf), replace=False)
        sel = np.concatenate([cf, cl]); rng.shuffle(sel)
    return [sel for _ in range(G)]


def c4_context(metric, scaler, fraud_source):
    """Per-group class-conditional context. scaler/metric/fraud_source configurable."""
    Xs = scaler.fit(Xtr).transform(Xtr)
    Xtes_s = scaler.transform(Xtes)
    rf = KNNRetriever(metric).fit(Xs[fidx])
    rl = KNNRetriever(metric).fit(Xs[lidx])
    out = []
    nfl = int(round(RATIO * ctx)); nll = ctx - nfl
    for g in range(G):
        centre = Xtes_s[labels == g].mean(0)
        nfu = min(nfl, len(fidx))
        if fraud_source == "knn":
            cf = fidx[rf.query(centre, nfu)]
        else:
            rng = np.random.default_rng([SEED, g, 0xF])
            cf = rng.choice(fidx, nfu, replace=False)
        cl = lidx[rl.query(centre, min(nll, len(lidx)))]
        idx = np.concatenate([cf, cl]).astype(np.int64)
        np.random.default_rng([SEED, g]).shuffle(idx)
        out.append(idx)
    return out


print("\n" + "=" * 92)
print(f"FTM VARIANTS — {DATASET} seed={SEED} {MODEL} n_est={NEST} ratio={RATIO} (grid ref: C1=0.908 C4cos=0.688)")
print("=" * 92)
run_variant("C1", global_context("C1"))
run_variant("C2", global_context("C2"))
run_variant("C4_cos", c4_context("cosine", StandardScaler(), "knn"))
run_variant("C4_euc", c4_context("euclidean", StandardScaler(), "knn"))
run_variant("C4_cos_rndF", c4_context("cosine", StandardScaler(), "random"))
run_variant("C4_euc_robust", c4_context("euclidean", RobustScaler(), "knn"))

# -------------------------------------------------------------------- save + compare
pred_df = pd.DataFrame(preds)
pred_path = OUT / f"{DATASET}_seed{SEED}_{MODEL}_preds.parquet"
pred_df.to_parquet(pred_path, index=False)
sum_df = pd.DataFrame(summary)
print("\n" + "=" * 92)
print("SUMMARY")
print("=" * 92)
print(sum_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# C1 vs C4_cos prediction-level comparison
print("\nC1 vs C4_cos prediction comparison:")
c1, c4 = preds["p_C1"], preds["p_C4_cos"]
print(f"  score correlation (spearman-ish via rank): "
      f"{np.corrcoef(pd.Series(c1).rank(), pd.Series(c4).rank())[0,1]:.3f}")
yt = preds["y_true"].astype(bool)
print(f"  mean score on TRUE frauds:  C1={c1[yt].mean():.4f}  C4_cos={c4[yt].mean():.4f}")
print(f"  mean score on TRUE legits:  C1={c1[~yt].mean():.4f}  C4_cos={c4[~yt].mean():.4f}")
# fraud-legit score separation (a calibration-free signal of ranking quality)
print(f"  fraud-minus-legit mean gap: C1={c1[yt].mean()-c1[~yt].mean():+.4f}  "
      f"C4_cos={c4[yt].mean()-c4[~yt].mean():+.4f}")
print(f"\nsaved predictions -> {pred_path}")
