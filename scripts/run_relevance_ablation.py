"""
Relevance ablation (ratio-matched): isolate the PURE relevance effect by the 2x2
of selection mechanisms {legit: random|kNN} x {fraud: random|kNN}, holding fraud
ratio, context size, and granularity G identical.

This driver runs only the two CROSS arms (the others are reused):
  KR = kNN legit + random fraud   (isolates legit-relevance)
  RK = random legit + kNN fraud   (isolates fraud-relevance)
Reused: RR (random both) = imported from the C2 grid (results/runs_c2grid/,
run_c2_grid.py) — that grid is architecturally G=1 (one global context, one
fit; no per-group refit at all), NOT a matched-G arm. KK (kNN both) = the
design-space sweep (results/runs_rapds/, run_rap_designspace.py, G32
full-context) — that grid, like KR/RK here, runs per-group at G=32.

So the 4 arms do NOT form a clean 2x2 at matched (ratio, context, G): RR sits
at G=1 while KR/RK/KK all sit at G=32. The narrated RR->KR contrast ("flipping
only the legit selector") also silently flips granularity (1 fit -> 32
refits), confounding the comparison. The clean matched-G contrast is RK vs KK
(both G=32, differing only in the legit selector); RR vs RK is a fraud-selector
check across the G confound, not a controlled isolation. A true RR@G=32 arm
(per-group random-both) would close this gap and is an open control — it is
not run by this script.

Grid (configurable via CLI): G {1,32} x ratio {0.05,0.10} x context {10k,50k}
x {tabiclv2,tabpfn_3} x {eu_cc,banksim,paysim} x 3 seeds. metric=cosine (the
favourable metric from the design-space sweep). With-replacement fraud oversample
(matches C2/C4). Idempotent (config hash -> parquet) -> results/runs_ablation/.

Usage:
  # batch 1 (core, completes the 2x2 via reuse): G=32, ctx=50k
  TABPFN_TOKEN=... uv run python scripts/run_relevance_ablation.py --granularities 32 --contexts 50000
  # full grid
  TABPFN_TOKEN=... uv run python scripts/run_relevance_ablation.py
  uv run python scripts/run_relevance_ablation.py --dry-run
"""
from __future__ import annotations

import argparse, hashlib, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.eval.metrics import compute_metrics
from src.experiment.runner import _PREDICT_CHUNK, _predict_proba_batched
from src.experiment.split_util import load_and_subsample_test
from src.models.ftm import CONTEXT_LIMITS, build_ftm
from src.rap.retriever import KNNRetriever

try:
    import torch
except Exception:
    torch = None

ABL_VERSION = 1
UNITS = {"eu_cc": [42, 123, 7], "banksim": [42, 123, 7], "paysim": [42, 123, 7]}
MODELS = ["tabiclv2", "tabpfn_3"]
ARMS = ["KR", "RK"]            # KR=kNN legit+random fraud; RK=random legit+kNN fraud
GRANS = [1, 32]
RATIOS = [0.05, 0.10]
CONTEXTS = [10000, 50000]
METRIC = "cosine"
N_ESTIMATORS = 4
PREDICT_CHUNK = 16384
OUT = Path("results/runs_ablation")


def cell_hash(c):
    s = json.dumps({"study": "relevance_ablation", "v": ABL_VERSION, "dataset": c["dataset"],
                    "seed": int(c["seed"]), "model": c["model"], "arm": c["arm"], "G": int(c["G"]),
                    "fraud_ratio": c["fraud_ratio"], "context_size": int(c["context_size"]),
                    "metric": METRIC, "n_estimators": N_ESTIMATORS},
                   sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(s.encode()).hexdigest()[:12]


def configs(ds, seed, model, grans, ratios, contexts):
    return [dict(dataset=ds, seed=seed, model=model, arm=a, G=g, fraud_ratio=r, context_size=c)
            for a in ARMS for g in grans for r in ratios for c in contexts]


def done_set():
    return {f.stem for f in OUT.glob("*.parquet")} | {f.stem for f in OUT.glob("*.failed")}


def grouping(Xs, G, seed):
    if G <= 1:
        return np.zeros(len(Xs), int), Xs.mean(0, keepdims=True)
    raw = KMeans(n_clusters=G, random_state=seed, n_init=10).fit_predict(Xs)
    order = sorted(np.unique(raw), key=lambda g: int(np.flatnonzero(raw == g).min()))
    remap = {o: i for i, o in enumerate(order)}
    lab = np.array([remap[g] for g in raw])
    centres = np.vstack([Xs[lab == i].mean(0) for i in range(len(order))])
    return lab, centres


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--granularities", nargs="*", type=int, default=GRANS)
    ap.add_argument("--ratios", nargs="*", type=float, default=RATIOS)
    ap.add_argument("--contexts", nargs="*", type=int, default=CONTEXTS)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    grans, ratios, contexts = args.granularities, args.ratios, args.contexts

    units = [(d, s) for d, ss in UNITS.items() for s in ss if (args.datasets is None or d in args.datasets)]
    allc = [(d, s, m, c) for (d, s) in units for m in MODELS for c in configs(d, s, m, grans, ratios, contexts)]
    done = done_set()
    pending = [x for x in allc if cell_hash(x[3]) not in done]
    print(f"Total: {len(allc)}  pending: {len(pending)}  done: {len(allc)-len(pending)} "
          f"| G={grans} ratios={ratios} ctx={contexts}")
    if args.dry_run:
        return

    from importlib.metadata import PackageNotFoundError, version
    libs = {}
    for p in ("tabpfn", "tabicl"):
        try: libs[p] = version(p)
        except PackageNotFoundError: libs[p] = None

    nrun = 0
    for (ds, seed) in units:
        ucells = [(m, c) for m in MODELS for c in configs(ds, seed, m, grans, ratios, contexts)]
        done = done_set()
        if all(cell_hash(c) in done for _, c in ucells):
            continue
        print(f"\n### {ds} seed={seed} ###", flush=True)
        Xtr, ytr, Xtes, ytes, info, prev, subsampled, _negcap = load_and_subsample_test(ds, seed)
        scaler = StandardScaler().fit(Xtr); Xs = scaler.transform(Xtr); Xtes_s = scaler.transform(Xtes)
        fidx = np.where(ytr == 1)[0]; lidx = np.where(ytr == 0)[0]
        rf = KNNRetriever(METRIC).fit(Xs[fidx]); rl = KNNRetriever(METRIC).fit(Xs[lidx])
        groups = {g: grouping(Xtes_s, g, seed) for g in grans}

        for model in MODELS:
            mc = [c for c in configs(ds, seed, model, grans, ratios, contexts) if cell_hash(c) not in done]
            if not mc:
                continue
            limit = CONTEXT_LIMITS[model]
            print(f"  -- {model}: {len(mc)} cells --", flush=True)
            mdl = build_ftm(model, device="cuda", seed=seed, n_estimators=N_ESTIMATORS); mdl.ensure_loaded()
            for cfg in mc:
                rid = cell_hash(cfg); out = OUT / f"{rid}.parquet"
                if out.exists():
                    continue
                arm, G, ratio = cfg["arm"], cfg["G"], cfg["fraud_ratio"]
                ctx = min(cfg["context_size"], len(ytr), limit)
                lab, centres = groups[G]
                nf = int(round(ratio * ctx)); nl = ctx - nf; nfu = min(nf, len(fidx))
                probs = np.empty(len(ytes), np.float64); fit_s = pred_s = 0.0; fn_acc = []
                for gi in range(len(centres)):
                    c = centres[gi]; rng = np.random.default_rng([seed, gi, int(round(ratio*1000)), cfg["context_size"]])
                    # fraud: KR=random, RK=kNN
                    if arm == "RK":
                        cf = fidx[rf.query(c, nfu)]
                    else:
                        cf = rng.choice(fidx, nfu, replace=False)
                    if nf > nfu and nfu > 0:
                        cf = np.concatenate([cf, rng.choice(cf, nf - nfu, replace=True)])
                    # legit: KR=kNN, RK=random
                    nlt = min(nl, len(lidx))
                    if arm == "KR":
                        cl = lidx[rl.query(c, nlt)]
                    else:
                        cl = rng.choice(lidx, nlt, replace=False)
                    idx = np.concatenate([cf, cl]).astype(np.int64); rng.shuffle(idx)
                    m = lab == gi
                    t0 = time.perf_counter(); mdl.fit(Xtr[idx], ytr[idx]); fit_s += time.perf_counter() - t0
                    if m.any():
                        t0 = time.perf_counter()
                        probs[m] = _predict_proba_batched(mdl, Xtes[m], batch_size=PREDICT_CHUNK)
                        pred_s += time.perf_counter() - t0
                    fn_acc.append(nf)
                    if torch is not None and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                if not np.all(np.isfinite(probs)):
                    (OUT / f"{rid}.failed").write_text("non_finite\n"); print(f"  !! non-finite {rid}"); continue
                mm = compute_metrics(ytes, probs, prevalence=prev if subsampled else None)
                row = {"run_id": rid, "study": "relevance_ablation", "v": ABL_VERSION, "dataset": ds,
                       "seed": int(seed), "model": model, "arm": arm, "G": int(G), "metric": METRIC,
                       "fraud_ratio": ratio, "context_size_req": int(cfg["context_size"]), "context_size_eff": int(ctx),
                       "n_estimators": N_ESTIMATORS, "pr_auc": mm["pr_auc"], "recall_at_1fpr": mm["recall_at_1fpr"],
                       "recall_at_5fpr": mm["recall_at_5fpr"], "roc_auc": mm["roc_auc"], "f1": mm["f1"],
                       "context_fraud_n": int(nf), "context_fraud_unique": int(nfu), "n_groups": int(len(centres)),
                       "n_test_scored": int(len(ytes)), "prevalence": prev, "fit_s": fit_s, "predict_s": pred_s,
                       "tabpfn_lib": libs["tabpfn"], "tabicl_lib": libs["tabicl"]}
                pd.DataFrame([row]).to_parquet(out, index=False); nrun += 1
                print(f"  [{nrun}/{len(pending)}] {ds} s{seed} {model} {arm} G{G} r{ratio} "
                      f"ctx{cfg['context_size']//1000}k: PR-AUC={mm['pr_auc']:.4f} ({fit_s+pred_s:.0f}s)", flush=True)
    print("\nAll done.")


if __name__ == "__main__":
    main()
