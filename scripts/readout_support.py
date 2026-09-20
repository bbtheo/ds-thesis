"""
Support tables for the decoder-readout case study. Rebuilds the C2 and
C4 contexts of scripts/run_readout.py (same samplers, same RNG streams; no model
needed) and writes:

  results/readout/nn_coverage.parquet   per readout test row: cosine distance (scaled
                                        space, as the retriever sees it) to its nearest
                                        legit and nearest fraud context row, under C2 and C4
  results/readout/example_rows.parquet  feature values of the worked-example test row
                                        and its top-10 context rows under C2 and C4
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd
from sklearn.neighbors import NearestNeighbors
from src.data.loader import load_dataset
from src.data.schema import DATASETS
from src.experiment.split_util import load_and_subsample_test
from src.models.ftm import CONTEXT_LIMITS
from src.rap.context import RetrievalContextBuilder
from src.rap.sampler import sample_c4
from scripts.run_readout import build_c2_plateau, SEED, C2_RATIO, C4_RATIO, C4_METRIC

OUT = Path("results/readout")
cov_rows, ex_rows = [], []
for ds in ["eu_cc", "banksim"]:
    Xtr, ytr, Xte, yte, *_ = load_and_subsample_test(ds, SEED)
    _, _, fnames = load_dataset(ds)
    limit = CONTEXT_LIMITS["tabpfn_3"]
    ro = {c: pd.read_parquet(OUT / f"{ds}_{c}.parquet") for c in ["C2", "C4"]}
    sub = ro["C2"].test_row_id.to_numpy()
    b = RetrievalContextBuilder(Xtr, ytr, metric=C4_METRIC, seed=SEED)
    Xte_s = b.scale(Xte)
    labels, centres, G, sil = b.group_test(Xte_s)
    ctx = {"C2": {0: build_c2_plateau(Xtr, ytr, limit, limit, SEED, C2_RATIO)}, "C4": {}}
    for g in np.unique(labels):
        idx, _ = sample_c4(b.retr_fraud, b.retr_legit, b.fraud_idx, b.legit_idx,
                           centres[g], min(len(ytr), limit), C4_RATIO, np.random.default_rng([SEED, int(g)]))
        ctx["C4"][int(g)] = idx
    Xtr_s = b.scaler.transform(Xtr)
    for cond in ["C2", "C4"]:
        grp = labels[sub] if cond == "C4" else np.zeros(len(sub), dtype=int)
        d_leg = np.full(len(sub), np.nan); d_fr = np.full(len(sub), np.nan)
        for g in np.unique(grp):
            idx = ctx[cond][int(g)]
            q = Xte_s[sub[grp == g]]
            for lab, arr in [(0, d_leg), (1, d_fr)]:
                pool = np.unique(idx[ytr[idx] == lab])
                nn = NearestNeighbors(n_neighbors=1, metric="cosine", algorithm="brute").fit(Xtr_s[pool])
                arr[grp == g] = nn.kneighbors(q, return_distance=True)[0][:, 0]
        cov_rows.append(pd.DataFrame({"dataset": ds, "condition": cond, "test_row_id": sub,
                                      "y_true": yte[sub], "d_nearest_legit": d_leg, "d_nearest_fraud": d_fr}))
    # worked example: banksim legit row with the largest C4 - C2 score gap
    if ds == "banksim":
        j = ro["C2"].merge(ro["C4"], on="test_row_id", suffixes=("_c2", "_c4"))
        j = j[j.y_true_c2 == 0]
        r = j.loc[(j.y_score_c4 - j.y_score_c2).idxmax()]
        tid = int(r.test_row_id)
        def feat(x, **k):
            return {**dict(zip(fnames, x)), **k}
        ex_rows.append(feat(Xte[tid], role="test row", condition="both", label=int(yte[tid]), rank=0, weight=np.nan,
                            score_c2=r.y_score_c2, score_c4=r.y_score_c4, test_row_id=tid))
        for cond in ["C2", "C4"]:
            rr = ro[cond].set_index("test_row_id").loc[tid]
            for k, (ti, lab, w) in enumerate(zip(rr.top10_ctx_train_idx, rr.top10_ctx_label, rr.top10_ctx_weight), 1):
                ex_rows.append(feat(Xtr[ti], role="context", condition=cond, label=int(lab), rank=k, weight=float(w),
                                    score_c2=np.nan, score_c4=np.nan, test_row_id=tid))
        print("banksim features:", fnames, "categoricals:", DATASETS[ds]["categoricals"])
pd.concat(cov_rows).to_parquet(OUT / "nn_coverage.parquet", index=False)
pd.DataFrame(ex_rows).to_parquet(OUT / "example_rows.parquet", index=False)
print("done")
