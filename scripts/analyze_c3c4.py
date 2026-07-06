"""
Ad-hoc analysis of the C3/C4 (RAP) grid vs the C1/C2 baselines.

Loads all pipeline_version==4 rows for the two RAP FTMs (tabiclv2, tabpfn_3),
averages over seeds, and reports the relevance effect (C3 vs C1), the ratio
sweep (C4), the best ratio per cell, and the RAP lift over the C1 (random) and
C2 (stratified-balance) baselines. Also flags with-replacement duplication.
"""
import glob
import numpy as np
import pandas as pd

pd.set_option('display.width', 200)
pd.set_option('display.max_rows', 200)

COLS = ['dataset', 'model', 'condition', 'seed', 'fraud_ratio', 'metric',
        'n_estimators', 'pipeline_version', 'pr_auc', 'recall_at_1fpr',
        'recall_at_5fpr', 'roc_auc', 'f1', 'context_fraud_n',
        'context_fraud_unique', 'n_groups', 'group_silhouette']

rows = []
for f in glob.glob('results/runs/*.parquet'):
    try:
        d = pd.read_parquet(f)
        rows.append(d[[c for c in COLS if c in d.columns]])
    except Exception:
        pass
df = pd.concat(rows, ignore_index=True)

RAP_FTMS = ['tabiclv2', 'tabpfn_3']
DS_ORDER = ['eu_cc', 'banksim', 'paysim', 'baf', 'fifar']
RATIOS = [0.05, 0.10, 0.20, 0.30, 0.50]

# v4 grid only, RAP FTMs, drop dev-only ai_banking.
m = (df.pipeline_version == 4) & (df.model.isin(RAP_FTMS)) & (df.dataset != 'ai_banking')
g = df[m].copy()

# Mean over seeds for each (dataset, model, condition, fraud_ratio) cell.
def agg(frame):
    keys = ['dataset', 'model', 'condition', 'fraud_ratio']
    a = (frame.groupby(keys, dropna=False)
              .agg(pr_auc=('pr_auc', 'mean'),
                   r1=('recall_at_1fpr', 'mean'),
                   r5=('recall_at_5fpr', 'mean'),
                   roc=('roc_auc', 'mean'),
                   cf_n=('context_fraud_n', 'mean'),
                   cf_u=('context_fraud_unique', 'mean'),
                   ngrp=('n_groups', 'mean'),
                   nseed=('pr_auc', 'size'))
              .reset_index())
    return a

A = agg(g)

def cell(ds, model, cond, ratio=None):
    q = (A.dataset == ds) & (A.model == model) & (A.condition == cond)
    if cond == 'C4':
        q &= np.isclose(A.fraud_ratio.astype(float), ratio)
    r = A[q]
    return r.iloc[0] if len(r) else None

print('=' * 100)
print('BASELINES + C3 + best-C4 — PR-AUC (mean over seeds), per dataset × model')
print('=' * 100)
recs = []
for ds in DS_ORDER:
    for model in RAP_FTMS:
        c1 = cell(ds, model, 'C1'); c2 = cell(ds, model, 'C2'); c3 = cell(ds, model, 'C3')
        c4s = [(r, cell(ds, model, 'C4', r)) for r in RATIOS]
        c4s = [(r, c) for r, c in c4s if c is not None]
        best_r, best_c4 = max(c4s, key=lambda rc: rc[1].pr_auc) if c4s else (None, None)
        rec = {
            'dataset': ds, 'model': model,
            'C1': c1.pr_auc if c1 is not None else np.nan,
            'C2': c2.pr_auc if c2 is not None else np.nan,
            'C3': c3.pr_auc if c3 is not None else np.nan,
            'C4*': best_c4.pr_auc if best_c4 is not None else np.nan,
            'C4*_r': best_r,
            'C4*-C1': (best_c4.pr_auc - c1.pr_auc) if (best_c4 is not None and c1 is not None) else np.nan,
            'C4*-C2': (best_c4.pr_auc - c2.pr_auc) if (best_c4 is not None and c2 is not None) else np.nan,
            'C3-C1': (c3.pr_auc - c1.pr_auc) if (c3 is not None and c1 is not None) else np.nan,
        }
        recs.append(rec)
T = pd.DataFrame(recs)
print(T.to_string(index=False, float_format=lambda x: f'{x:.3f}'))

print()
print('=' * 100)
print('C4 RATIO SWEEP — PR-AUC by ratio (mean over seeds). Best ratio per row in brackets.')
print('=' * 100)
recs = []
for ds in DS_ORDER:
    for model in RAP_FTMS:
        row = {'dataset': ds, 'model': model}
        c1 = cell(ds, model, 'C1')
        row['C1'] = c1.pr_auc if c1 is not None else np.nan
        for r in RATIOS:
            c = cell(ds, model, 'C4', r)
            row[f'r{int(r*100):02d}'] = c.pr_auc if c is not None else np.nan
        recs.append(row)
S = pd.DataFrame(recs)
print(S.to_string(index=False, float_format=lambda x: f'{x:.3f}'))

print()
print('=' * 100)
print('AGGREGATE — mean PR-AUC by condition/ratio across all 10 dataset×model cells')
print('=' * 100)
agg_rows = []
for cond, r in [('C1', None), ('C2', None), ('C3', None)] + [('C4', x) for x in RATIOS]:
    vals = []
    for ds in DS_ORDER:
        for model in RAP_FTMS:
            c = cell(ds, model, cond, r)
            if c is not None:
                vals.append(c.pr_auc)
    label = cond if r is None else f'C4 r={r}'
    agg_rows.append({'condition': label, 'mean_pr_auc': np.mean(vals), 'n_cells': len(vals)})
G = pd.DataFrame(agg_rows)
print(G.to_string(index=False, float_format=lambda x: f'{x:.3f}'))

print()
print('=' * 100)
print('RAP WIN RATE — best-C4 vs C1 and vs C2 (PR-AUC), per cell')
print('=' * 100)
win_c1 = (T['C4*-C1'] > 0).sum(); win_c2 = (T['C4*-C2'] > 0).sum(); n = len(T)
print(f'best-C4 beats C1 (random):     {win_c1}/{n} cells   mean lift {T["C4*-C1"].mean():+.3f}')
print(f'best-C4 beats C2 (balanced):   {win_c2}/{n} cells   mean lift {T["C4*-C2"].mean():+.3f}')
print(f'C3 beats C1 (relevance only):  {(T["C3-C1"]>0).sum()}/{n} cells   mean lift {T["C3-C1"].mean():+.3f}')

print()
print('=' * 100)
print('DUPLICATION FLAG — C4 cells where context_fraud_n / context_fraud_unique is high')
print('(with-replacement oversampling; dup>~3 means the context is mostly repeated frauds)')
print('=' * 100)
c4 = A[A.condition == 'C4'].copy()
c4['dup'] = c4['cf_n'] / c4['cf_u'].clip(lower=1)
heavy = c4[c4.dup > 3.0].sort_values('dup', ascending=False)
print(f'{len(heavy)}/{len(c4)} C4 cells have dup>3.0')
print(heavy[['dataset', 'model', 'fraud_ratio', 'pr_auc', 'cf_n', 'cf_u', 'dup']]
      .to_string(index=False, float_format=lambda x: f'{x:.2f}'))

print()
print('=' * 100)
print('SECONDARY METRICS — recall@1%FPR / recall@5%FPR / ROC-AUC, best-C4 vs C1 (mean over cells)')
print('=' * 100)
for metric, label in [('r1', 'recall@1%FPR'), ('r5', 'recall@5%FPR'), ('roc', 'ROC-AUC')]:
    c1v, c4v = [], []
    for ds in DS_ORDER:
        for model in RAP_FTMS:
            c1 = cell(ds, model, 'C1')
            c4s = [cell(ds, model, 'C4', r) for r in RATIOS]
            c4s = [c for c in c4s if c is not None]
            if c1 is not None and c4s:
                best = max(c4s, key=lambda c: c.pr_auc)  # ratio chosen by PR-AUC
                c1v.append(getattr(c1, metric)); c4v.append(getattr(best, metric))
    print(f'{label:14s}: C1 {np.mean(c1v):.3f}  ->  C4* {np.mean(c4v):.3f}  ({np.mean(c4v)-np.mean(c1v):+.3f})')

print()
print('grouping: mean n_groups per dataset (silhouette-chosen):')
ng = g[g.condition.isin(['C3', 'C4'])].groupby('dataset')['n_groups'].mean()
print(ng.to_string(float_format=lambda x: f'{x:.1f}'))
