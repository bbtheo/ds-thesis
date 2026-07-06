"""
Dev inspection for the RAP retrieval pipeline (C3/C4) on the ai_banking dev set.

NOT an experiment run — a diagnostic that prints the k-means/silhouette grouping,
the C3/C4 retrieved-context composition, the with-replacement duplication factor,
and a cosine-vs-euclidean retrieval comparison, so the metric choice and Kmax can
be eyeballed before committing the formal grid.

ai_banking has only ~8k train rows, which fit inside every real FTM context limit
(10k–50k), so at the real limit C3 would retrieve the WHOLE pool and selective
retrieval is never exercised. To make selection visible we inspect at a reduced
context (``--context-size``, default 2000). The formal grid always uses the real
``min(len(train), context_limit)``.

Usage
-----
    uv run python scripts/dev_rap_validation.py
    uv run python scripts/dev_rap_validation.py --context-size 1500 --seed 123
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.data.loader import load_dataset
from src.data.splitter import split
from src.rap.context import RetrievalContextBuilder

RATIO_SWEEP = (0.05, 0.10, 0.20, 0.30, 0.50)


def _jaccard(a: np.ndarray, b: np.ndarray) -> float:
    sa, sb = set(a.tolist()), set(b.tolist())
    return len(sa & sb) / len(sa | sb) if (sa or sb) else 1.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--context-size", type=int, default=2000,
                    help="reduced inspection context to force selective retrieval (dev only)")
    args = ap.parse_args()

    X, y, names = load_dataset("ai_banking")
    Xtr, Xte, ytr, yte = split(X, y, seed=args.seed)
    ctx = min(args.context_size, len(ytr))
    nat = ytr.mean()
    print(f"ai_banking seed={args.seed}: train={len(ytr)} (fraud {int(ytr.sum())}, "
          f"{nat:.1%}) test={len(yte)} (fraud {int(yte.sum())})")
    print(f"inspection context_size={ctx} (real grid would use "
          f"min(train, context_limit)={len(ytr)})\n")

    builders = {}
    for metric in ("cosine", "euclidean"):
        print(f"================  metric = {metric}  ================")
        b = RetrievalContextBuilder(Xtr, ytr, metric=metric, seed=args.seed)
        builders[metric] = b
        labels, centres, G, sil = b.group_test(b.scale(Xte))
        sizes = np.bincount(labels).tolist()
        print(f"grouping: G={G}  silhouette={sil:.4f}  group sizes={sizes}")

        # --- C3: mixed retrieval, natural-as-found ratio + guard ---
        c3_ratios, guards = [], 0
        for g in range(G):
            _, y_ctx, st = b.build("C3", centres[g], ctx)
            c3_ratios.append(st["fraud_n"] / len(y_ctx))
            guards += int(st["guard_fired"])
        print(f"C3: realised fraud ratio  mean={np.mean(c3_ratios):.3f} "
              f"min={np.min(c3_ratios):.3f} max={np.max(c3_ratios):.3f}  "
              f"(natural={nat:.3f})  single-class guard fired in {guards}/{G} groups")

        # --- C4: class-conditional, ratio-controlled + with-replacement top-up ---
        print("C4 (group 0): target -> realised fraud ratio, unique frauds, duplication")
        for r in RATIO_SWEEP:
            rng = np.random.default_rng([args.seed, 0])
            _, y_ctx, st = b.build("C4", centres[0], ctx, fraud_ratio=r, rng=rng)
            dup = st["fraud_n"] / max(st["fraud_unique"], 1)
            print(f"  r={r:>4}: ctx={len(y_ctx):>5} realised={int(y_ctx.sum())/len(y_ctx):.3f} "
                  f"fraud_n={st['fraud_n']:>4} unique={st['fraud_unique']:>4} dup={dup:.2f}x")
        print()

    # --- cosine vs euclidean: how much does the metric change WHICH rows are pulled? ---
    print("================  cosine vs euclidean retrieval overlap  ================")
    bc, be = builders["cosine"], builders["euclidean"]
    # Grouping is KMeans (euclidean) regardless of metric, so centres match — compare
    # retrieved sets group-by-group at a ratio that actually selects a fraud subset.
    lc, cc, Gc, _ = bc.group_test(bc.scale(Xte))
    r = 0.20  # n_fraud = round(.2*ctx) < train frauds -> a real fraud subset is chosen
    jacc_f, jacc_l = [], []
    for g in range(Gc):
        nf = int(round(r * ctx))
        idx_cf = bc.fraud_idx[bc.retr_fraud.query(cc[g], min(nf, len(bc.fraud_idx)))]
        idx_ef = be.fraud_idx[be.retr_fraud.query(cc[g], min(nf, len(be.fraud_idx)))]
        idx_cl = bc.legit_idx[bc.retr_legit.query(cc[g], min(ctx - nf, len(bc.legit_idx)))]
        idx_el = be.legit_idx[be.retr_legit.query(cc[g], min(ctx - nf, len(be.legit_idx)))]
        jacc_f.append(_jaccard(idx_cf, idx_ef))
        jacc_l.append(_jaccard(idx_cl, idx_el))
    print(f"at r={r}: mean Jaccard(retrieved fraud sets)={np.mean(jacc_f):.3f}  "
          f"mean Jaccard(retrieved legit sets)={np.mean(jacc_l):.3f}")
    print("(1.0 = metrics pull identical rows; lower = the metric materially changes "
          "retrieval -> worth A/B-ing. Default per plan: cosine.)")


if __name__ == "__main__":
    main()
