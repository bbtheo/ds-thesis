# External figure-source data

Small, tracked datasets used only for thesis figures (not for model training).

## eda_*.parquet — data-chapter EDA summaries

Full-data (exact, unsampled) summary statistics over the six formal-grid
datasets, computed inside DuckDB by `scripts/export_data_summaries.R` and
consumed by the data-chapter figures so that `quarto render` never needs the
(unversioned) DuckDB file. Regenerate with:

```bash
Rscript scripts/export_data_summaries.R
```

| File | Grain | Contents |
|---|---|---|
| `eda_feature_auc.parquet` | dataset × numeric feature | tie-aware single-feature ROC-AUC (`auc`), orientation-free `auc_abs = max(auc, 1-auc)`, class counts |
| `eda_feature_stats.parquet` | dataset × feature × class | n, mean, sd, min, max |
| `eda_feature_quantiles.parquet` | dataset × feature × class × p | class-conditional quantiles at p ∈ {.01,.05,.1,.25,.5,.75,.9,.95,.99} |
| `eda_time_fraud_rate.parquet` | dataset × time bin | n, n_fraud per time bin (eu_cc hour, banksim/paysim step, baf/fifar month) |
| `eda_categorical_fraud_rate.parquet` | dataset × categorical feature × level | n, n_fraud for the top 30 levels by volume (`n_levels_total` records the untruncated level count) |

`class` is `fraud`/`legit`; features follow `src/data/schema.py` (label, ID,
and leakage columns excluded; fifar = train ∪ test with labels unified).

## statfin_fraud_offences_13ex.csv

Police-recorded fraud offences in Finland, 1980–2025, whole country, annual.

- **Source:** Statistics Finland (Tilastokeskus), statistic *Offences and coercive
  measures* (Rikos- ja pakkokeinotilasto), StatFin database table **13ex**
  "Offences recorded and their solving by offence category according to the
  municipality of offence and year of reporting, 1980–2025".
- **Browse URL:** https://pxdata.stat.fi/PxWeb/pxweb/en/StatFin/StatFin__rpk/statfin_rpk_pxt_13ex.px/
- **Downloaded:** 2026-07-07 via the PxWeb API (table last updated 2026-07-01).
- **License:** CC BY 4.0 (attribution: Statistics Finland).
- **Columns:** Year; Municipality (WHOLE COUNTRY only); Offence group; Offences
  known to the authorities (number).
- **Offence categories included** (Criminal Code chapter:section):
  - 1116 Fraud, petty fraud 36:1,3
  - 1117 Aggravated fraud 36:2
  - 1118 Means of payment fraud, petty means of payment fraud, preparation thereof 37:8,10,11
  - 1119 Aggravated means of payment fraud 37:9
- **Caveat:** categories 1117–1119 are zero before 1991 because the offence
  categories were introduced by the 1991 Criminal Code reform (structural zeros,
  not absence of the behaviour). Filter them out before 1991 when plotting.

Reproduce the download:

```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"query":[{"code":"alue_23_20230101","selection":{"filter":"item","values":["SSS"]}},{"code":"rikokset_74_20211209","selection":{"filter":"item","values":["141","142","143","144"]}},{"code":"contentscode","selection":{"filter":"item","values":["rikokset_lkm"]}}],"response":{"format":"csv"}}' \
  https://pxdata.stat.fi/PxWeb/api/v1/en/StatFin/rpk/13ex.px \
  -o statfin_fraud_offences_13ex.csv
```
