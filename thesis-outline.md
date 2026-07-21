# Thesis Outline — Context Construction for Tabular Foundation Models under Heavy Class Imbalance: a relevance-vs-ratio decomposition (Retrieval-Augmented Prediction is a negative result; stratified balancing wins)

> **Status & framing (revised 2026-06-22 per supervisor review `scratchpad/thesis-outline-review.md`).** Distilled from the full literature + the *complete* C1–C4 grid (pipeline_version==4, n_estimators=4), with every Chapter-6/7 number recomputed from the parquet result sets and every load-bearing citation checked against the source.
> **Central result — honest reframing:** RAP's retrieval-relevance lever is a **negative result** — C4 (RAP) < C2 (stratified) on every dataset and C3 (kNN) < C1 (random) on all but baf (a marginal +0.005 exception at the separability floor, within the tie band), with per-dataset attributed causes (lost negative diversity where retrieval collapses; duplication degeneracy on eu_cc — §7.2). The *positive*, salvageable finding is that **stratified context balancing (C2) is the best context-construction strategy and is competitive with sound external SOTA at no per-dataset tuning** (the effective optimum is a broad plateau near r≈0.10, not literal 50/50). The thesis is an honest relevance-vs-ratio decomposition, not a "RAP wins" story.
>
> **Conventions for finding-pointers below**
> - `[finding: results/runs …]` = main grid (`results/runs/*.parquet`, v4, n_est=4, FTMs `tabpfn_3`+`tabiclv2` for C3/C4).
> - `[finding: results/runs_c2grid]` = C2 fraud-ratio grid (810 cells); `[finding: results/runs_rapds]` = RAP design-space sweep (720 cells); `[finding: results/diag]` = mechanism diagnostics.
> - `[finding: eda §N]` = `scratchpad/eda/eda-results-v4.md`; figures `scratchpad/eda/fig1–fig5`.
> - **PR-AUC pivot rule:** values are the **better of {tabiclv2, tabpfn_3}**, mean over seeds, C4 at its best ratio — both models shown where they diverge (the divergence is itself a finding, §6.7).
> - All `[@key]` verified present in `thesis-refs.bib`. Two lit-review keys (`shukor2024multimodal_icl`, `mccarter2024what_tabpfn_learned`) were **not** in the bib and are dropped.

---

# Chapter 1. Introduction

> **Ch.1/Ch.2 division of labor (drafting rule):** the introduction gives each technical concept **at most one plain-language sentence + forward reference**; every mechanism, definition, and derivation appears **once, in Ch. 2**. Ch. 1 is organized around the *argument* (problem → idea → contributions → RQs), Ch. 2 around the *concepts* needed for Methods/Discussion. Test: an expert reader can skip Ch. 2 entirely and still follow Ch. 1 → Ch. 4; a novice never sees an intro claim re-justified from scratch in Ch. 2 without the full treatment.

## 1.1 Fraud detection as an extreme-imbalance learning problem
- societal stakes (already drafted in `sections/introduction.qmd`): Finnish police-recorded fraud up sharply since 2010 (StatFin 13ex figure); FFI: EUR 148M attempted / 72.5M realised losses in 2025; EBA/ECB: EUR 4.2bn EEA payment fraud in 2024, +17% YoY [@statfin2026_fraud_offences_13ex; @finanssiala2025_huijaukset_2024; @finanssiala2026_huijaukset_2025; @ebaecb2025_payment_fraud_report]
- scope narrowing (drafted): first-party vs third-party fraud; thesis = third-party (fraudster- or victim-initiated); adversarial drift (fraudsters adapting to the detector) explicitly out of scope
- ML framing: supervised binary classification at 0.1–1.5% prevalence — and the EBA-reported real-world rates (0.001–0.033%) mean public datasets are already oversampled by ≥1 order of magnitude [@he2009learning_imbalanced_data; @ebaecb2025_payment_fraud_report]
- one-sentence assertions only (no argument here): accuracy is uninformative at this skew, and a fixed investigation budget fixes the tolerable false-positive rate → evaluation must center the minority class [@he2009learning_imbalanced_data; @leborgne2022fraud_detection_handbook]
  - *(defers to §2.2: the full evaluation argument — ROC dilution, PR-AUC primacy, Recall@FPR as the budget metric, and the ranking-vs-calibration canonical statement. He 2009's remedy families defer to §2.1.)*
- *(planned figure, TODO already in draft: fraud-prevalence bar chart across the 6 formal datasets vs the EBA real-world band)*

## 1.2 From gradient-boosted trees to tabular foundation models
- GBDTs are the entrenched tabular baseline — one sentence on why (robust on heterogeneous features, cheap, tuned-SOTA) [@grinsztajn2022why_tree_based; @chen2016xgboost_scalable_tree_boosting; @dorogush2018catboost]
- ICL tabular FTMs (TabPFN line, TabICL): pretrained once on synthetic tasks, then classify a *new* table in a single forward pass by conditioning on a context of labeled rows — no gradient training, competitive with tuned GBDTs [@hollmann2025accurate_tabular_foundation_model; @qu2026tabiclv2]
- the one plain-language consequence the reader must carry: **whatever sits in the context window is the model's entire effective training set for that prediction** — stated without mechanism
  - *(defers to §2.4–2.5: attention/ICL, the PFN Bayesian posterior-predictive framing — the [@muller2022transformers_bayesian_inference] citation lives THERE, not here — version lineage, context limits)*

## 1.3 The problem: a fixed context window wasted under imbalance
- FTMs have a bounded context budget (soft, tied to pretraining-scale row counts) [@hollmann2025accurate_tabular_foundation_model]
- the hook — the π·B arithmetic lives HERE, once (canonical statement): expected fraud rows in a random context = π·B → a 10k window at 0.17% holds ~17 frauds; many subsamples hold none
- two deficiencies of the random context: (i) irrelevant rows, (ii) class-imbalanced rows — this pair is the C1–C4 design in embryo (each condition toggles one deficiency)
  - *(defers to §2.6: why flooding is fatal rather than a nuisance — because the context is the effective training set (§2.4). §2.6 back-references this arithmetic instead of restating it.)*

## 1.4 The idea: Retrieval-Augmented Prediction (the hypothesis under test)
- RAP = test-time context construction on a **frozen** model, no fine-tuning [@thomas2024localpfn_retrieval_finetuning]
- two levers, mapping 1:1 onto §1.3's two deficiencies: relevance (kNN retrieval around the test region) and class balance (controlled fraud-ratio oversampling)
- HYPOTHESIS (stated, later falsified): a context both relevant and minority-populated sharpens the posterior [@nagler2023statistical_foundations_pfn; @saerens2002adjusting_priors]
  - honesty note: PFN guarantees are tied to the synthetic prior + pretraining-scale contexts → out-of-design benefit is an empirical question [@nagler2023statistical_foundations_pfn]
  - *(defers to §3.1: the theory that makes the hypothesis plausible — Nagler localization, kernel-regression lens, in-context prior. Here it is one paragraph of claim, not argument.)*

## 1.5 Contributions (honest framing)
- **(primary) a clean relevance-vs-ratio decomposition** of test-time context construction (C1→C2→C3→C4) on frozen FTMs at fraud-grade (<1.5%) imbalance, with each lever isolated: C1→C2 (ratio, clean); C1→C3 (relevance) backed by a **ratio-matched 2×2 relevance ablation** that removes the residual ratio drift [finding: results/runs_ablation]
- **(primary) a NEGATIVE result with a quantified symptom and an attributed cause:** retrieval-relevance does not help, and usually hurts, ICL tabular FTMs under heavy imbalance (C3<C1, C4<C2) [finding: results/runs] — attributed to *lost negative diversity* on the ratio-matched relevance ablation (3 datasets: eu_cc, banksim, paysim), which is the **dominant driver where retrieval collapses** (paysim, banksim), while **duplication degeneracy dominates on eu_cc** (where the same locality is mildly *positive* for legit selection, +0.01–0.02 PR-AUC) and baf/fifar sit at a separability floor; the anchor-injection diagnostic is a banksim/tabiclv2 case study, not general causal evidence [finding: results/diag; results/runs_ablation; results/runs_rapds]. A clean, per-dataset-attributed null on a published hypothesis (Nagler localization / LoCalPFN frozen retrieval) is itself a contribution
- **(practical) context rebalancing wins:** stratified balancing of the in-context fraud ratio (C2) is the robust best strategy on every dataset and is competitive with methodologically-sound external SOTA at **no per-dataset hyperparameter tuning** — above sound external results on paysim and banksim; at par (statistical tie) on eu_cc; at par on baf, fifar [finding: eda §2; sota-comparison]
  - novelty vs prior art: this partly replicates Tanna et al. 2026 (balanced context helps tabular FTMs on imbalanced credit data) but extends it to (i) far heavier imbalance (0.13–1.5% fraud vs ~8–22%), (ii) threshold-free *ranking* metrics (PR-AUC, Recall@FPR) Tanna does not report, and (iii) the full C2 ratio×context sweep + the C1–C4 decomposition [@tanna2026data_presentation_resampling]
- a prevalence-corrected, threshold-free benchmark filling a Recall@FPR reporting gap (no resampling-before-split leakage; the random-split time-feature caveat is stated in §5.5/§8.1) [finding: sota-comparison]

## 1.6 Research questions
- RQ1 — FTMs vs GBDTs under naive random subsampling [→ C1: FTM-C1 vs GBDT-C1]
- RQ2 — where is a random context most deficient (which size/prevalence regime)? [→ C1 across size/prevalence tiers + where C1→C2 gain concentrates]
- RQ3 — does retrieval-relevance improve over a random context? [→ C3 vs C1]
- RQ4 — does controlled in-context fraud ratio help, and is there an optimum? [→ C2 vs C1 (ratio in a random context); within-C4 ratio sweep (ratio on a fixed retrieval sampler)]
  - cross-check (multi-axis, hedged — relevance/ratio/granularity all differ at once, §6.8/§7.2): **C4 vs C2** — does retrieval add anything beyond rebalancing? single-factor evidence for this question lives in §6.7/§7.2, not this comparison alone [→ §6.8 / §7.1]

## 1.7 Thesis outline
- chapter roadmap

---

# Chapter 2. Background

## 2.1 Supervised fraud detection under extreme class imbalance
- imbalance-remedy families: SMOTE interpolation [@chawla2002smote]; focal/cost-sensitive loss [@lin2017_focal_loss]; survey [@he2009learning_imbalanced_data]
- under extreme imbalance, objective/balance matters more than architecture [@sun2025extreme_imbalance_baf]

## 2.2 Evaluation under imbalance
- *(division of labor: §1.1 asserts "accuracy is uninformative" in one sentence; the full argument is made HERE, once — do not re-motivate from fraud statistics)*
- accuracy and ROC-AUC are optimistic at heavy skew (FPR diluted by huge negatives) [@fawcett2006roc_analysis; @saito2015pr_more_informative_imbalanced]
- PR-AUC / average precision is the right primary metric [@davis2006relationship_pr_roc; @saito2015pr_more_informative_imbalanced]
- Recall@fixed-FPR matches a fixed review budget [@leborgne2022fraud_detection_handbook; @mcclish1989partial_roc_curve; @jesus2022baf_arxiv]
- **canonical statement — ranking vs calibration** (the load-bearing defense, stated once here, referenced later):
  - threshold-free ranking metrics (PR-AUC, Recall@FPR, ROC-AUC) are invariant to any monotone re-scaling of scores — an elementary property of rank-based metrics [@fawcett2006roc_analysis; @davis2006relationship_pr_roc]
  - post-hoc prior correction (the SLD adjustment) is exactly such a monotone rescale of a *fixed* scorer's outputs, so it cannot move these metrics [@saerens2002adjusting_priors]; the same holds empirically for in-weight resampling, which shifts calibration but not discrimination [@vandengoorbergh2022harm_imbalance_corrections] — van den Goorbergh's result concerns well-specified, low-capacity models where rebalancing is approximately an intercept shift (monotone, discrimination-invariant); the analogy does not carry over unmodified to an ICL FTM, whose nonparametric posterior conditions directly on the context, so rebalancing the context changes the *effective training set*, not just a prior term
  - **therefore a C2 gain on these metrics cannot be reproduced by thresholding/prior-correction.** Because the in-context set *is* the training data, C2 re-fits a *different* scoring function (not a monotone rescale of C1's scores), so the gain reflects a genuinely different posterior. *(This in-context-vs-in-weight distinction, and the reason the van den Goorbergh analogy breaks, is our interpretive argument, not established consensus; the invariance shield is exact for a single global monotone map — satisfied by C1 and C2 only. C3 mixes groups with different base rates (non-monotone across groups) and C4's per-group retrieval also changes the scorer per group, so **both C3 and C4 sit outside the shield**, §8.1.)*
  - this is the direct rebuttal to the McDowell "just threshold" critique (§3.5) [@mcdowell2026correcting_imbalance_pfn]

## 2.3 Tabular ML and gradient-boosted decision trees
- trees dominate tabular DL [@grinsztajn2022why_tree_based]; XGBoost [@chen2016xgboost_scalable_tree_boosting]; CatBoost [@dorogush2018catboost]; LightGBM [@ke2017lightgbm]

## 2.4 In-context learning and Prior-Data Fitted Networks (mechanism)
- transformers / self-attention [@vaswani2017attention_is_all_you_need]; LLM few-shot ICL [@brown2020language_models_few_shot]
- PFNs = amortized Bayesian posterior-predictive in one forward pass; CE meta-training ≈ minimizing KL to the true PPD [@muller2022transformers_bayesian_inference]
- the in-context set **is** the data the PPD conditions on → context composition determines the prediction [@muller2022transformers_bayesian_inference]
- ICL as implicit Bayesian inference; demonstrations concentrate a posterior over a latent concept (analogy, not tabular proof) [@xie2022explanation_icl_bayesian]
- permutation invariance of TabPFN/TabICL → ordering pathologies scoped out (architectural, not by curation)
  - *(division of labor: mechanism here; the kNN-as-Nagler's-remedy argument + SCM-prior caveat go to §3.1)*

## 2.5 Tabular foundation models
- TabPFN [@hollmann2023tabpfn; @hollmann2022tabpfn_arxiv]; v2 matches/beats tuned GBDTs [@hollmann2025accurate_tabular_foundation_model]
- 2.5 [@grinsztajn2025tabpfn25_arxiv]; 2.6 [@priorlabs2026tabpfn26_model_card]; v3 [@grinsztajn2026tabpfn3_technical_report; @priorlabs2026tabpfn_release_v800]; TabICLv2 [@qu2026tabiclv2]
- analyses/extensions: closer look at v2 [@ye2025closer_look_tabpfnv2]; scaling via sketching [@feuer2023scaling_tabpfn]; ensembling as a context choice [@lakshminarayanan2017deep_ensembles]
- FTMs in finance/anomaly [@olusegun2024ifs_tabpfn_ethereum; @priorlabs_taktile_case_study; @djilani2025robustness_tabular_foundation_models]

## 2.6 The context-window bottleneck under imbalance (bridge into Ch. 3)
- fixed budget + attention cost grows with rows [@vaswani2017attention_is_all_you_need]
- the mechanistic reading of §1.3's arithmetic (**back-reference it, do not restate it**): because the PPD conditions directly on the context (§2.4), a majority-flooded window is not "too little fraud data" — the context is the model's *entire world* for that prediction, so the minority class must be characterized from a handful of rows with no training-time memory to fall back on
- context is a controllable test-time lever (shown adversarially) [@djilani2025robustness_tabular_foundation_models]
- → hands off to Ch. 3: if composition determines the prediction, what does the literature propose for choosing it — and what does it predict for imbalance?

---

# Chapter 3. Related Work

## 3.1 Why context composition determines the prediction (theory)
- relevance reduces approximation bias: PFN variance vanishes from the architecture, but bias vanishes only if the context localizes around the test point; Nagler proposes a **simple post-hoc kNN context restriction** ("Localized PFNs", his §6.4) as one remedy — RAP's C3 mechanism is precisely this published method (Nagler frames it as one illustrative localization, not the unique fix) [@nagler2023statistical_foundations_pfn]
- kernel-regression lens: ICL attention ≈ Nadaraya–Watson similarity-weighted label average — an LLM-ICL result; transfer to tabular FTMs is analogy, not proof [@han2025kernel_regression_icl]
- in-context label frequency = effective class prior; posterior scales by P_ctx(y)/P_test(y) [@saerens2002adjusting_priors]
- caveat: PFN guarantees tied to the SCM prior; SCM-to-fraud distance unknown → empirical study warranted [@nagler2023statistical_foundations_pfn]

## 3.2 The TabPFN lineage and the scaling of context
- accuracy rises past the 10k pretraining limit (2.5→~50k) [@grinsztajn2025tabpfn25_arxiv]; random sketching matches informed selection up to Feuer's own tested cap (n_max=3,000 — a "~100k" sketching figure is not in that paper and is dropped here; the surviving half of the claim is §3.5's "adverse prior 1") [@feuer2023scaling_tabpfn]
- context optimization as a paradigm: prompt-tuned soft contexts [@feuer2024tunetables]
- but under <1.5% fraud, even a 50k random window holds few fraud rows → curation, not "use all data"

## 3.3 Retrieval and context-selection for ICL (C3 is confirmatory, not novel)
- tabular retrieval precedent: LoCalPFN — its **frozen retrieval-only ablation (TabPFN-kNN) already helps** on general tabular benchmarks (IQM AUC 0.943 vs 0.917 vanilla), with fine-tuning adding the remainder (→0.958) [@thomas2024localpfn_retrieval_finetuning]; TabDPT [@ma2024tabdpt]; TabR neighbor attention [@gorishniy2024tabr_nearest_neighbors]
- feature/instance-subset construction (incl. KMeans prototypes) lets TabPFN v2 scale to oversized data and can beat the vanilla model in that regime → composition matters, not only size (a scaling result; Ye does **not** test test-point kNN retrieval) [@ye2025closer_look_tabpfnv2]
- LLM-ICL: similarity-based demo selection helps [@liu2022good_in_context_examples; @rubin2022learning_to_retrieve_prompts]; order sensitivity [@lu2022fantastically_ordered_prompts; @zhao2021calibrate_before_use]
- kNN foundations + standardize before distance [@cover1967nearest_neighbor_pattern_classification; @hastie2009elements_statistical_learning]
- the gap RAP tests: frozen kNN retrieval is **shown to help on general/balanced tabular data**, but has not been tested under fraud-grade (<1.5%) imbalance — our C3 finds it fails there (a regime/domain transfer gap, not an unexplored niche)

## 3.4 Class balance and ratio control in the context
- balanced/resampled context lifts AUC/MCC ("data presentation over architecture") for tabular FTMs on imbalanced credit data — the closest prior result to our C2 finding (differentiation in §1.5/§3.5) [@tanna2026data_presentation_resampling]
- inverse-density context for imbalanced regression [@nejjar2024im_context]
- imbalance-aware retrieval gains *grow* with imbalance — but only with a **learned** retriever; naive distance can **degrade** (shown on high-dimensional clinical EHR data) [@pham2026aware]
- in LLM-text ICL, naive class-weight-only rebalancing is ineffective; Gao's RCB couples class weight with a within-class conditional-bias correction and *works* on accuracy — an analogue **in intent** of coupling C3+C4 (but RCB is post-hoc score reweighting, not in-context oversampling) [@gao2025imbalanced_annotations]

## 3.5 The direct competitor and the adverse priors
- McDowell et al. 2026 — closest competitor: same inference-time design space for **TabPFN-2.5** under imbalance, NO relevance/kNN, threshold-dependent metrics (balanced/worst-class accuracy), *milder* imbalance (π₁≥0.05, balanced test set); concludes post-hoc thresholding/downsampling win and context oversampling *hurts* via a "spiky posterior" [@mcdowell2026correcting_imbalance_pfn] — note their *downsampling-to-50/50* is exactly our C2 mechanism, so that half **corroborates the direction** of our positive result, not the operating point (§6.4 finds ratios ≥0.20 hurt, and 50/50 sits past the eu_cc duplication cliff)
- competing prior for the positive result: Tanna et al. 2026 reports balanced-context-helps-TFMs on (milder) imbalanced credit data; our C2 partly replicates it and extends to heavy imbalance + ranking metrics + the decomposition (§1.5) [@tanna2026data_presentation_resampling]
- DistPFN — analytic post-hoc counterpart to RAP: it adjusts the same in-context class-prior term *analytically* (downweighting it) where RAP manipulates it *in-context* (opposite directions — DistPFN removes an imposed prior, RAP imposes one) [@lee2026distpfn_label_shift_tabpfn]
- adverse prior 1: random sketching matches informed sketching (k-means/CoreSet) for TabPFN in all but one case — a strong baseline [@feuer2023scaling_tabpfn]
- adverse prior 2: post-hoc prior-correction is a monotone rescale and cannot move ranking metrics (§2.2), so a real C2 ranking gain must be discrimination [@saerens2002adjusting_priors; @vandengoorbergh2022harm_imbalance_corrections]
- our heavy-imbalance tabular test of these *is* the open question

## 3.6 Why retrieval can fail under imbalance (predicted failure modes)
- a relevant neighborhood is still majority-dominated → collapses toward a similarity-weighted "legitimate" vote [@pham2026aware]
- single-class neighborhoods, and — by analogy only, since the published evidence is high-dimensional image data — minority "bad hubs" in imbalanced data (a speculative, out-of-domain mechanism; see §8.1) [@tomasev2013minority_hubs]
- with-replacement duplication shifts the posterior without adding information (our own empirical observation; consistent with the "spiky-posterior" account) [@mcdowell2026correcting_imbalance_pfn]

## 3.7 Taxonomy of test-time context construction (positioning)
- five axes: **relevance** (random / kNN) × **balance** (natural / enforced ratio) × **granularity** (global / per-group) × **replacement** (unique / with-replacement) × **selection mode** (mixed / class-conditional)
  - C1 = random·natural·global·unique·mixed; C2 = random·enforced·global·unique·mixed; C3 = kNN·natural·per-group·unique·**mixed**; C4/RAP = kNN·enforced·per-group·with-replacement·**class-conditional** — C3's mixed retrieval vs C4's class-conditional retrieval is itself a factor the C3→C4 step changes (§5.2)
  - LoCalPFN/TabDPT = kNN·natural·per-query + fine-tuning (LoCalPFN also has a frozen retrieval-only arm that helps, §3.3) [@thomas2024localpfn_retrieval_finetuning; @ma2024tabdpt]; AWARE = learned-relevance [@pham2026aware]; IM-Context = inverse-density balance [@nejjar2024im_context]; McDowell = random·post-hoc-threshold [@mcdowell2026correcting_imbalance_pfn]; DistPFN = post-hoc analytic [@lee2026distpfn_label_shift_tabpfn]
- RAP's cell (kNN relevance + enforced swept ratio + per-cluster local + with-replacement, on a frozen FTM at <1.5% fraud) is unoccupied → the gap the thesis tests (the verdict is delivered in §6–§7, not here)

---

# Chapter 4. Data

## 4.1 Datasets and the imbalance axis
- 6 formal public transaction datasets; task fixed, varied on size (~285k–6.3M) and prevalence (0.13–1.5%) [finding: methods §datasets]
- eu_cc (ULB) 284,807×30, 0.17%, **real transaction data** [@dalpozzolo2015calibrating]; cc_2025 500k×11, 1.5%, synthetic [@rajak2025credit_card_fraud_2025]; banksim 594,643×7, 1.2%, **agent-based simulation** [@lopezrojas2014banksim; @patil2024graphdb_banksim]
- paysim 6,362,620×7, 0.13%, **agent-based simulation** [@lopezrojas2016_paysim; @alumona2025paysim_ml]; fifar 602,961×29, 1.2%, **synthetic (BAF-derived)** [@alves2023fifar_learning_to_defer]; baf 1,000,000×30, 1.1%, **semi-synthetic generative suite** [@jesus2022baf_arxiv; @feedzai_baf_repo]
- ai_banking 10k, 28.4% — dev-only, excluded (near-balanced, unsuitable as an imbalance benchmark) [@talha2025ai_powered_banking_fraud]
- **provenance caveat:** only eu_cc is real transaction data; the other 4 formal-grid datasets are agent-based simulations (banksim, paysim), synthetic (fifar), or semi-synthetic (baf) — load-bearing for any "negative manifold of real fraud data" mechanism language (§7.2/§8.1)

## 4.2 cc_2025 as a degenerate control
- fraud labels statistically independent of every feature [finding: MEMORY.md cc_2025]
- all models score PR-AUC ≈ fraud rate (0.015), ROC-AUC ≈ 0.50 [finding: results/runs, cc_2025]
- retained as a cautionary example of a poorly-constructed synthetic dataset; no further runs queued — so the formal evaluation is substantively a **5-dataset** study (cc_2025 is a one-section data-quality control, not a result-bearing dataset) [finding: CLAUDE.md]

## 4.3 Difficulty is separability, not imbalance (preview)
- best PR-AUC (pivot) ordering: paysim 0.983 > banksim 0.909 > eu_cc 0.869 > fifar 0.224 > baf 0.183 — difficulty tracks imbalance in **neither** direction (eu_cc is 3rd, behind ~7×-less-imbalanced banksim), which *strengthens* the section's point rather than illustrating a simple imbalance/difficulty trend [finding: eda §5]
- baf/fifar built with deliberately overlapping fraud/legit distributions [@jesus2022baf_arxiv; @alves2023fifar_learning_to_defer]; paysim fraud is rule-based [@lopezrojas2016_paysim]
- caveat: the paysim/eu_cc separability may be partly inflated by retaining time as a feature under a random split (§5.5) — a validity threat to be bounded, not a settled property [finding: §8.1]

---

# Chapter 5. Methods

## 5.1 Problem formulation
- frozen ICL FTM; test-time context construction only; pre-trained weights untouched [@thomas2024localpfn_retrieval_finetuning]
- RQ-to-condition decomposition is the design spine: C1→C2 (ratio), C1→C3 (relevance), C3→C4 (ratio on retrieval), C4 vs C2 (retrieval beyond rebalancing)

## 5.2 Experimental conditions C1–C4
- C1 Random: uniform subsample to context limit, natural ratio — baseline
- C2 Stratified: balance toward 50/50 (all available fraud if scarce — true 50/50 only on the 10k model; ~11–18% on the 48–50k models, ~0.8% on eu_cc); C2 vs C1 isolates the ratio effect with relevance fixed [finding: eda §7]
- C3 kNN control: retrieval around test clusters, natural ratio. C3 vs C1 *varies relevance* but does **not** hold the ratio fixed (the realized fraud fraction drifts with the local neighbourhood — depletes eu_cc, enriches banksim); relevance is isolated cleanly only by the ratio-matched ablation (§5.3, §7.2) [finding: results/runs, C3 context_fraud_n; results/runs_ablation]
- C4 RAP: class-conditional kNN + enforced fraud ratio swept {0.05,0.10,0.20,0.30,0.50}. The **within-C4 ratio sweep** isolates ratio tuning on a fixed retrieval sampler; the C3→C4 step additionally swaps mixed for class-conditional retrieval, so it is not single-factor [finding: phase3-plan decision 4]
- GBDTs run C1 only on the full training set (the context limit is a hard architectural bound for tabpfn_v2 (10k); for the larger-context FTMs the 50k figure used here is a chosen experiment budget for cross-model comparability, not a universal model property — §5.4)

## 5.3 The RAP pipeline (per-group)
- per-GROUP design (CLAUDE.md / src authoritative; the draft's per-batch "Algorithm 1" is stale) [finding: src/rap/context.py]
- cluster the scored test set (k-means; G chosen by silhouette over k=2..20 on a seeded ~5k subsample); G recorded as output `n_groups`, `group_silhouette` = selection score [finding: phase3-plan decision 2]
- one retrieval + one FTM refit per group around its cluster center. Cost argument = **context reuse / cacheability**, not fit-count: ICL "fit" is near-free preprocessing (fit_s ≤1 s vs predict_s 100–600 s; no transformer at fit in the default modes, verified from tabpfn 8.0.8 / tabicl 2.0.3 source); the expensive operation is encoding the ~50k context, paid inside every predict pass when uncached (~10.5 s/pass constant at 50k ctx; predict_s linear in pass count) and amortizable/cacheable only across predictions that share the context. Per-group keeps distinct contexts at G (2–32), each serving thousands of rows → predict is condition-invariant (§6.10) and KV-cache-compatible (opt-in `fit_with_cache`/`kv_cache=True`; **substantial benefit**, ~6× at our shape; unsupported on tabpfn_26, memory ∝ n_estimators, tabpfn_3 cache int8-quantized ≠ bit-identical; our runs uncached throughout). Finer granularity (per-batch/per-query, at any batch size) trades reuse for locality: per-prediction encoding cost grows as context/batch and cacheability vanishes in the small-batch limit, which is the objection to the draft's per-batch "Algorithm 1". Deployment: each context build is also a data-store query (§7.4) [finding: phase3-plan decision 1; src/experiment/runner.py `_PREDICT_CHUNK`; tabpfn inference.py inference engines]
- retriever: brute NearestNeighbors on StandardScaler-standardized train features (fit on train only); cosine default; retrieval depth derived from `context_size` [finding: src/rap/retriever.py]
- C3 mixed retrieval + single-class guard (inject nearest global fraud/legit when a group is single-class; logged) [finding: src/rap/sampler.py]
- C4 class-conditional retrieval; fill fraud quota **with replacement** when pool < quota — oversampling *is* the method; log `context_fraud_n` and `context_fraud_unique` → duplication factor recoverable [finding: src/rap/sampler.py]
- C3/C4 restricted to `tabpfn_3` + `tabiclv2` (their C1/C2 baselines exist → contrasts intact) [finding: phase3-plan decision 5]
- C3/C4 hashing: `batch_size` and `k` dropped; `metric` + `fraud_ratio` are the real factors; C1/C2/GBDT hashes unchanged → no pipeline_version bump
- **ratio-matched relevance ablation** (the clean isolation of relevance): a 2×2 of {legit: random|kNN} × {fraud: random|kNN} at matched fraud ratio and context, run on 3 datasets (eu_cc, banksim, paysim) — RR = random both (imported from the C2 grid, architecturally **G=1**, one global context/fit), KK = kNN both (= the C3/C4 path, G=32), KR = kNN-legit + random-fraud (G=32), RK = random-legit + kNN-fraud (G=32); RR is therefore **not** granularity-matched to the other three. The clean within-G32 isolation of *which* selector drives the collapse is the **RK-vs-KK contrast** (same G=32, differs only in legit selector): flipping the legit selector from random to kNN collapses paysim ~0.97→~0.48, is mild on banksim (~−0.03), and is mildly *positive* on eu_cc (~+0.01–0.02); RR-vs-RK is the fraud-selector check (does almost nothing); RK≈RR empirically suggests the per-group G=32 architecture itself is roughly neutral with random legits. **The 2026-07-18 extension closes this confound:** a true RR@G=32 arm (per-group random both, ratios {0.05, 0.10}, 50k) matches both RK@G=32 and the global RR within seed noise on all 3 datasets (banksim RR 0.906 vs RK 0.898–0.906; eu_cc RR 0.817–0.851 vs RK 0.830–0.846, max |Δ| 0.013 ≪ eu_cc seed-sd; paysim RR 0.975–0.979 vs RK 0.977–0.979), so per-group granularity with random contexts is directly confirmed neutral and the RK-vs-KK legit-selector contrast is granularity-clean end to end. A companion **natural-ratio pair** ({RR, KK} at ratio = train prevalence, G=32, 50k) supplies the granularity- and ratio-matched analogue of C3-vs-C1 (results in §6.5) [finding: results/runs_ablation; scripts/run_relevance_ablation.py]

## 5.4 Models
- 5 FTMs: tabpfn_v2 (10k), tabpfn_25/26/3 (~50k experiment context budget), tabiclv2 (~48k) [@hollmann2025accurate_tabular_foundation_model; @grinsztajn2025tabpfn25_arxiv; @priorlabs2026tabpfn26_model_card; @grinsztajn2026tabpfn3_technical_report; @qu2026tabiclv2]
- the 50k figure is a design choice for cross-model comparability, not an inherent property at that number for every model: the installed tabpfn_3 checkpoint's own `MAX_NUMBER_OF_SAMPLES` is 1,000,000 and tabpfn_26's is 100,000 (verified 2026-07-06 from the inference config, tabpfn 8.0.8) — both far above the 50k budget used here; tabpfn_v2 (10,000) and tabpfn_25 (50,000) do run at their checkpoints' own caps
- common sklearn fit/predict_proba wrapper; `fit` loads context not weights; all GPU (RTX 5070)
- library versions recorded per-run, not hashed (`tabpfn` 8.0.8) — v4 full rerun keeps one library, no silent mix
- GBDT baselines class-weighted (`scale_pos_weight` set from train labels), fixed **strong defaults**: 2,000 trees, depth 6, lr 0.05, subsample/colsample 0.8, PR-AUC early stopping (patience 50, 10% stratified val split) — *not* library defaults. Neither GBDTs nor FTMs are tuned per dataset, so the comparison is **zero-shot FTM vs fixed-default GBDT** (a tuned-GBDT comparison is not performed, §8.1) [@chen2016xgboost_scalable_tree_boosting; @dorogush2018catboost]
- CatBoost `thread_count=1` to avoid a TBB-race SIGSEGV on the paysim ~5M-row fit (CPU thread-invariant → bit-identical) [finding: catboost-paysim-segfault]

## 5.5 Splitting protocol
- seeded stratified 80/20; banksim grouped by `customer` (entity-ID leakage guard) [@lopezrojas2014banksim]; fifar fixed published split (single seed)
- time columns kept as features deliberately (eu_cc `Time`, banksim/paysim `step`) — stated explicitly; under a random split this is a **temporal-leakage threat** that may inflate the separable datasets' ranking metrics (paysim/eu_cc), bounded as a limitation (§8.1) and partially checked by the BAF-by-month ablation (§8.2) [finding: CLAUDE.md]
- temporal generalization = a separate BAF-by-`month` ablation, not the main grid [@jesus2022baf_arxiv]

## 5.6 Test-set negative subsampling and prevalence correction (v4)
- keep ALL positives, cap negatives at `test_neg_cap`=30,000; eu_cc scored full (~98 frauds) [finding: CLAUDE.md results schema]
- PR-AUC prevalence-corrected (`ap_at_prevalence`); = standard AP to machine precision when not subsampled [finding: metrics.py]
- Recall@FPR / ROC-AUC invariant to negative subsampling; dominant speedup (paysim ~42×); a cap change ⇒ version bump

## 5.7 Evaluation metrics
- primary: prevalence-corrected PR-AUC, Recall@1%FPR, Recall@5%FPR
- Recall@FPR = conservative step value (highest TPR with FPR ≤ target); no interpolation (interpolated points are not achievable thresholds) [finding: CLAUDE.md pipeline v3]
- tiny downward bias 1/N_neg, negligible at N≈30k; secondary: ROC-AUC, F1@0.5

## 5.8 Implementation, reproducibility, compute
- Python = train/infer/RAP; R+Quarto+Typst = analysis; communicate only via Parquet (no reticulate)
- all randomness via `numpy.random.default_rng(seed)`; test-subsample stream offset `seed ^ 0x5CA1E`; versions pinned via uv.lock
- idempotent runner: config → SHA-256 run_id → output path; `_PIPELINE_VERSION`=4 in the hash; analysis filters to v4 (v2/v3 stale, except cc_2025) [finding: CLAUDE.md results schema]
- **all reported FTM grid results use `n_estimators=4`** for one-GPU tractability; runner default 8 is NOT comparable [finding: CLAUDE.md]
- 3-way timing split (setup_s / fit_s / predict_s) so model-load cost never contaminates fit/predict comparisons
- **reproducibility caveat:** k-means grouping is deterministic only at a fixed thread configuration — thread-count changes can alter the label partition (the recorded C3/C4 grid predates the 2026-07-06 threadpool pin), and GPU inference carries the usual CUDA nondeterminism jitter. Seeds fix all sampling decisions, but bit-exact rerun is not claimed across different thread/CUDA configurations

## 5.9 Statistical treatment
- 3 seeds per dataset (fifar = 1 fixed split); results are descriptive point estimates with seed-spread (±sd or min/max) attached to every per-dataset value
- paired comparisons (e.g. the C2−C1 ΔPR-AUC over 65 paired runs) reported with a confidence interval, not a bare mean — the 65 deltas share only **13** dataset-seed splits across 5 models (not 65 independent observations), so the CI is a **cluster (split-level) bootstrap**: resample the 13 units, keeping all models' deltas together, rather than a naive per-run paired bootstrap/Wilcoxon; where clustering cannot be respected, deltas are reported as descriptive only
- tie band |Δ| < 0.005 stated explicitly for all win/tie/loss tallies; any effect within one seed-sd is flagged "directional, not significant at n=3"
- **fifar** is excluded from any variance-dependent claim (fixed split): qualitative "consistent-with" support only; even at 10 seeds (extension slice) its seeds vary context subsampling, not split composition, so its CIs quantify context-sampling variance only
- **CI seed extension (complete, verified 2026-07-19):** the two SOTA FTMs (tabpfn_3, tabiclv2) × {C1, C2} × 5 formal datasets at a fixed **a-priori** 10-seed list ({0–6, 42, 123, 7}, identical for every dataset; all seeds reported, none dropped), n_estimators=4 through the unchanged v4 pipeline; driver `scripts/run_seed_extension.py` (idempotent, deadline-aware) [finding: scripts/run_seed_extension.py; results/runs]
- extension-slice treatment: per dataset×model, seed-paired C2−C1 deltas with a seeded 10,000-resample bootstrap 95 % percentile CI over the 10 seed-level deltas; pooled per model with a cluster bootstrap that resamples seeds, keeping each seed's five per-dataset deltas together; resulting numbers reported in Ch. 6, not here
- everything outside the extension slice (C3/C4, the other 3 FTMs) stays under the n=3 rules above

## 5.10 The experiment grid
- designed 16 dataset-seed units (5×3 + fifar×1) including cc_2025; **executed 13** (cc_2025 dropped as degenerate after its C1/C2 cells); seeds 42,123,7; report mean±sd (fifar single point)
- executed v4 analysis grid (verified complete, zero duplicates/missing cells): C1/C2 FTM 5 models×2 conditions×13=130; GBDT 2 models×13=26; C3 2 FTMs×13=26; C4 2 FTMs×5 ratios×13=130 (net-new beyond the shared default-ratio cells: 104); batch-size sweep DROPPED (per-group grouping replaced fixed batching) [finding: phase3-plan decision 2]
- cc_2025 cells beyond the already-collected C1/C2 dropped (degenerate)
- **seed-extension tier (complete 2026-07-19):** 2 SOTA FTMs × {C1, C2} × 5 datasets × 10 seeds = 200 runs, 52 shared with the core grid, 148 net-new; `results/runs` totals 464 v4 parquets (312 core + 4 ai_banking dev + 148 extension) [finding: scripts/run_seed_extension.py; results/runs, verified 2026-07-19]

---

# Chapter 6. Results

## 6.1 RQ1 — FTMs vs GBDTs under random subsampling (C1)
- under a random context, context-limited FTMs and full-data trees are roughly on par; best-FTM-C1 vs best-GBDT win/tie/loss = 2/1/2; family means GBDT 0.591 > FTM 0.569 PR-AUC [finding: eda §3]
- the comparison is intentionally asymmetric: GBDTs see all 228k–5.1M rows; each FTM sees only a 10k–50k subsample. The **same full-data GBDT advantage persists under C2** (C2 adds no data, only rebalances labels), so the C2 FTM win (§6.3) is achieved *despite* this handicap, not by removing it. RQ1 therefore answers "are context-limited FTMs competitive?" (yes, roughly); the fair-comparison verdict lives at C2/RQ4(a) [finding: eda §3]
- [finding: scratchpad/eda/fig4_metric_panel.png; fig2_ftm_vs_gbdt.png]

## 6.2 RQ2 — where is a random context most deficient?
- the standalone deficiency of a random context is its in-context fraud count (C1 holds ~0.7% fraud regardless of size — §6.3); where that deficiency is worst, the C1→C2 balancing gain is largest, so the gain concentration is read as a *proxy* for deficiency (the two coincide only because balancing is the effective remedy — itself a finding, not a definition)
- C2-over-C1 gain concentrates where the context is most fraud-starved: smallest-context tabpfn_v2 +0.110 (~3× the 48–50k group); most-imbalanced paysim +0.142 [finding: eda §2]
- canonical case: tabpfn_v2/paysim C1 ≈11 frauds in a 10k window → PR-AUC 0.576; C2 ≈5,000 frauds → 0.955 (+0.380) [finding: eda §2,§4]
- mechanism: C1 holds ~0.7% fraud in every context regardless of size; C2 lifts fraud count 16–58× [finding: eda §2]
- *(planned figure: C1 PR-AUC vs n_rows / context-fraud-count panel — does not yet exist)*

## 6.3 RQ4(a) — the ratio effect: C2 vs C1 (the positive result)
- stratified context balancing robustly helps: mean ΔPR-AUC +0.057 (cluster/split-level bootstrap CI to be reported, §5.9); **57/60 (95%) of non-fifar paired runs improve** (fifar reported separately per the §5.9 n=1 rule: 5/5 consistent-with); zero recall regressions [finding: eda §2]
- the only 3 dips are banksim, all ≤0.014 (C1 already near-ceiling) [finding: eda §2]
- C2 is the BEST context-construction strategy on PR-AUC across all 5 formal datasets (banksim only marginally above C1, +0.001 pivot-rule delta) [finding: results/runs, C2 best per dataset]
- the gain is on threshold-free ranking metrics → genuine discrimination, not calibration, **on our interpretive reading of the in-context-vs-in-weight distinction** (see §2.2 canonical statement) — the load-bearing rebuttal to the McDowell critique [@saerens2002adjusting_priors; @mcdowell2026correcting_imbalance_pfn]
- **what "balanced" means here:** C2 reaches true 50/50 only on tabpfn_v2 (10k); the 48–50k models that produce the SOTA numbers cap at ~11–18% (eu_cc ~0.8%) — and §6.4 shows the *optimum* is a broad plateau near r≈0.10, with ratios ≥0.20 hurting. So the lever is "include enough fraud (~5–10%), don't over-duplicate", not literal 50/50 [finding: eda §7; results/runs_c2grid]
- [finding: scratchpad/eda/fig1_condition_effect.png; fig3_delta_heatmap.png]

## 6.4 The C2 ratio×context grid (how much balancing, how big a context)
- design: random stratified C2 swept over fraud_ratio {0.02–0.40}×9 × context_size {10–50k}×5 × 2 FTM (tabiclv2, tabpfn_3) × {eu_cc, banksim, paysim} × 3 seeds = 810 cells; with-replacement fraud oversampling (matches C4 policy), `dup_factor` logged [finding: results/runs_c2grid, study=c2_grid]
- aggregation note: the by-ratio and by-context marginals below are simple means over models+seeds, whereas the best-*single-cell* numbers use the better-of-FTM pivot rule — stated to avoid apparent self-contradiction (under better-of-FTM the eu_cc by-ratio peak shifts to r0.02)
- the target ratio is ALWAYS realized (`realized_fraud_ratio` == requested to 3 dp) — with-replacement always fills the quota; on few-fraud data the cost of a high ratio is duplication, not under-filling [finding: results/runs_c2grid, realized vs requested]
- optimal fraud ratio is dataset-dependent but a BROAD (near-)PLATEAU, not a sharp optimum [finding: results/runs_c2grid, PR-AUC by ratio]
  - eu_cc peak r0.10 (0.834); r0.04/r0.05/r0.10 ≈0.831–0.834 with shallow interior dips at r0.06/r0.08 (~0.823–0.825 — not a perfectly flat band); sharp decline after r0.20 (0.801→0.781→0.777 at r0.40)
  - banksim peak r0.04 ≈ r0.10 (0.902); plateau r0.04–0.20 ≥0.90; mild decline r0.30/0.40 (0.886/0.879)
  - paysim peak r0.30 (0.980); monotone rise 0.02→0.30 then flat — tolerates high ratio (abundant unique fraud, low dup)
- NOT a fixed multiple of prevalence (peaks ≈ 59× / 3× / 231× the natural rate) → no "set ratio = k·prevalence" rule [finding: results/runs_c2grid]
- **r≈0.10 is within ~0.006 of every dataset's peak → safe universal default** (lower also minimizes duplication) — 0.006 sits below seed-sd on every dataset checked (§5.9), which *strengthens* rather than undercuts the default claim; note the ratio itself is chosen by scoring every sweep cell on the same fixed test splits (implicit multiple testing / test-set selection, an exploratory-search limitation carried to §5.9/§8.1/§7.4) [finding: results/runs_c2grid]
- context size: monotone but small gains, largest where fraud is scarcest — eu_cc 10k→50k +0.045 (0.784→0.829); paysim +0.015; banksim flat (~0.896, peaks 30k) [finding: results/runs_c2grid, PR-AUC by context_size]
- context helps mainly at LOW ratio; best single cell on the imbalanced sets = lowest ratio (r0.02) + biggest context (50k): eu_cc 0.864, banksim 0.910; paysim = mid-ratio (r0.30/20k) 0.983 [finding: results/runs_c2grid, best cell]
- Recall@1%FPR agrees with PR-AUC: flat-to-slightly-declining in ratio (eu_cc 0.906→0.893; banksim peak 0.939 @r0.10 → 0.919 @r0.40; paysim ~0.999 flat) → conclusions are not metric-specific [finding: results/runs_c2grid, recall_at_1fpr by ratio]
- duplication is the high-ratio failure mechanism — the C4-collapse confound isolated here in a RANDOM context:
  - dup_factor reaches ~50× on eu_cc (mean 10.5×) vs ≤3.5× on banksim/paysim → eu_cc is the duplication-stressed case [finding: results/runs_c2grid, dup_factor range]
  - heavy duplication degrades eu_cc PR-AUC: >25× dup → 0.784 vs 0.833 at the 3–10× plateau (a duplication cliff, not a retrieval effect) [finding: results/runs_c2grid, PR-AUC by dup bin]
- NET claim: the imbalance win is modest re-balancing of a RANDOM context (include frauds, ~5–10%, don't over-duplicate) — not relevance, not a large context, not precise tuning; this closes the ratio-vs-relevance and duplication-threshold caveats [finding: results/runs_c2grid]
- scope caveat: grid covers only the 3 separable datasets (eu_cc, banksim, paysim); baf/fifar (separability floor) not swept — their C2 sits at the difficulty ceiling regardless [finding: results/runs_c2grid datasets; eda §5]
- *(planned figure: C2 PR-AUC ratio×context heatmap per dataset + the duplication-cliff curve — does not yet exist)*

## 6.5 RQ3 — the relevance effect: C3 vs C1 (negative)
- C3 (kNN, natural ratio) is WORSE than C1 on 4/5 datasets, sometimes catastrophic (baf is the lone exception — separability floor) [finding: results/runs, C3 vs C1, n_est=4]
  - paysim C3 ~0.10 vs C1 ~0.91; eu_cc C3 0.33 (tabiclv2) / 0.56 (tabpfn_3) vs C1 ~0.82, with huge seed variance (tabiclv2 seed123 = 0.087) [finding: results/runs]
  - banksim C3 ~0.86 vs ~0.90 (mild); baf/fifar C3 ≈ C1 (separability floor) [finding: results/runs]
- the collapse is **not primarily** fraud-count starvation: C3 keeps ~39 frauds/group on eu_cc and ~276 on paysim *on average* — but the per-run spread is extreme (4/6 paysim C3 runs hold only ~1 fraud, so on those runs starvation is confounded); on the fraud-rich runs the driver is the local context's *legitimate* composition, isolated cleanly by the ratio-matched ablation (§5.3, §7.2-H1) [finding: results/runs, C3 context_fraud_n; results/runs_ablation]
- the silhouette picks a COARSE grouping (G=2 on banksim/paysim/fifar, ~4 on eu_cc, ~10 on baf). The harm is **not** that the grouping is *too coarse*: kNN retrieval around any centroid concentrates the context in the dense fraud/legit overlap — but the effect is **dataset-dependent, not universal locality-harm**: the ratio-matched ablation (§5.3, §7.2) shows the legit-selector delta is catastrophic on paysim (≈−0.72), mild on banksim (≈−0.03), and mildly *positive* on eu_cc (≈+0.01–0.02); forcing finer G still loses to C2 everywhere (§6.7) [finding: results/runs, n_groups/group_silhouette; results/runs_ablation]
  - cluster quality is weak outside banksim (group_silhouette: banksim 0.87, paysim 0.66, eu_cc 0.62, fifar 0.19, baf 0.07) [finding: results/runs, group_silhouette]
- the **natural-ratio ablation arms** (2026-07-18: {RR, KK} × ratio = train prevalence × G=32 × 50k, 3 datasets) give the granularity- and ratio-matched version of this contrast: kNN selection at natural ratio collapses paysim (RR 0.897 → KK 0.033, both models) and cuts banksim (0.901 → 0.785), but mildly *raises* eu_cc (0.834 → 0.845) — so eu_cc's main-grid C3 collapse (0.33/0.56) is carried by its drifted/depleted realized fraud ratio and the coarse silhouette grouping, not by locality itself, which independently sharpens the §7.2 per-dataset attribution [finding: results/runs_ablation, natural arms]
- the negative is *stronger* under the per-model (un-pivoted) view; the §6.8 better-of-two pivot understates it
- this is a regime/domain transfer gap from the LoCalPFN frozen-retrieval precedent (which helps on balanced general tabular data); it confirms the strong-random-baseline prior + AWARE's naive-distance caveat [@thomas2024localpfn_retrieval_finetuning; @feuer2023scaling_tabpfn; @pham2026aware]
- [finding: scratchpad/eda/fig6_condition_pivot.png — the C1/C2/C3/C4 pivot heatmap (created 2026-07-06, cells verified against §6.8)]

## 6.6 RQ4(b) — ratio tuning on retrieval: the C4 sweep (negative)
- sweep over fraud_ratio {0.05,0.10,0.20,0.30,0.50}; higher ratio HURTS on 4/5 datasets [finding: results/runs, C4 sweep, n_est=4]
  - eu_cc monotone mild decline: tabiclv2 0.74→0.72→0.67→0.65→0.62; tabpfn_3 0.71→0.72→0.69→0.68→0.63
  - banksim collapses past r0.05: tabiclv2 0.67→0.43→0.18→0.19→0.20; tabpfn_3 0.58→0.22→0.12→0.12→0.11
  - fifar (n=1) slow decline 0.052→0.036; baf flat ≈ fraud rate (0.015→0.013)
  - paysim splits by model: tabiclv2 is U-shaped, RISES at high ratio 0.48→0.43→0.42→0.49→0.56 (local frauds offset lost negatives); tabpfn_3 collapses immediately 0.073→0.010 [finding: results/runs]
- C4-best worse than C2 on EVERY dataset (eu_cc .74<.87, banksim .67<.91, paysim .56<.98, baf .016<.18, fifar .05<.22) [finding: results/runs, C4best vs C2]
- duplication is the high-ratio cliff ONLY on eu_cc: dup_factor 6.2×@r0.05 → 62×@r0.50 (extreme); banksim/paysim/baf/fifar stay ≤4.3× even at r0.50 → their collapse is the retrieval/locality, NOT duplication [finding: results/runs, context_fraud_n/unique]
  - corroborated independently in a RANDOM context (§6.4): heavy duplication alone degrades eu_cc PR-AUC (>25× → 0.784 vs 0.833 plateau) [finding: results/runs_c2grid, PR-AUC by dup bin]
- the relevance+ratio coupling that helps in Gao's LLM-text/accuracy setting does **not transfer** to frozen tabular FTMs under <1.5% fraud on ranking metrics (a failed transfer, not a refutation of Gao's own result); replicates McDowell's "context oversampling hurts" (TabPFN-2.5) and extends it to ranking metrics under heavier imbalance [@gao2025imbalanced_annotations; @mcdowell2026correcting_imbalance_pfn]
- *(planned figure: C4 ratio-sweep line plot per dataset — does not yet exist)*

## 6.7 The RAP design-space sweep (can any retrieval config win?)
- design: forced granularity G∈{8,16,32,64} × metric {cosine, euclidean} × fraud_ratio {0.05–0.50}×5 × 2 FTM × {eu_cc, banksim, paysim}×3 seeds = 720 cells, local-fraud strategy; self-contained schema [finding: results/runs_rapds, study=rap_designspace]
- motivation: the main grid's silhouette picks a coarse G (=2) → "local" ≈ global; force finer G to test whether better locality rescues retrieval
- finer G recovers but ASYMPTOTES BELOW C2 on every dataset [finding: results/runs_rapds, PR-AUC by n_groups]
  - banksim G8→16→32→64: 0.30→0.52→0.72→0.80 (C2 ≈ 0.91 — never reached)
  - eu_cc G8→64: 0.50→0.61→0.76→0.74 (peaks at G32; C2 ≈ 0.87)
  - paysim G8→64: 0.24→0.27→0.28→0.31 (C2 ≈ 0.98 — collapse persists at every G)
- cosine > euclidean (banksim 0.65 vs 0.52; eu_cc 0.71 vs 0.60; paysim tie ~0.27) → settles the open metric A/B for cosine, but cosine still loses to C2 [finding: results/runs_rapds, metric]
- ratio within retrieval is dataset-split, no universal value: eu_cc lower-is-better (0.75→0.53), paysim higher-is-better (0.24→0.33), banksim flat-then-declining [finding: results/runs_rapds, fraud_ratio]
- mechanism smoking gun — `legit_score` (mean predicted score on legit rows; lower = better) falls as G refines: banksim 0.070→0.005, eu_cc ~0.02 mean (0.051 worst cell)→0.001 → coarse retrieval INFLATES legit scores (lost easy-legit anchors = §7.2-H1); paysim stays high (0.535→0.233) → retrieval cannot separate paysim at any G [finding: results/runs_rapds, legit_score by n_groups]
- best cherry-picked RAP config TIES C2 only on eu_cc, loses elsewhere: eu_cc tabiclv2 cosine G64 r0.05 = 0.866 ≈ C2 0.87; banksim cosine G64 r0.05 = 0.898 < C2 0.91; paysim cosine G16 r0.50 = 0.693 ≪ C2 0.98 [finding: results/runs_rapds, best cell]
- the win requires fine G + cosine + a near-random-balanced regime — LOW ratio on the two near-separable sets (eu_cc/banksim), but HIGH ratio on paysim (its best cell is G16 r0.50); in every case the helpful limit *approaches a relevance-free balanced random context* → RAP is not salvageable by tuning [finding: results/runs_rapds]
- scope: 3 separable datasets only (baf/fifar at the separability floor excluded) → confirms the negative where retrieval *could* have helped [finding: results/runs_rapds datasets]
- [finding: scratchpad/eda/fig7_mechanism_panel.png — legit_score-vs-G panel + the banksim/tabiclv2 anchor-injection case study (created 2026-07-06)]; *(still planned: the G-recovery curve vs C1/C2 baselines)*

## 6.8 The PR-AUC pivot (consolidated)
- aggregation: better of {tabiclv2, tabpfn_3}, mean over seeds, C4 at best ratio, v4, n_est=4 [finding: results/runs]
- multi-axis contrast **C4 vs C2** (hedged: relevance, ratio, and granularity all differ at once — this is not a single-factor comparison; see §6.7/§7.2 for the single-factor evidence): retrieval adds nothing beyond rebalancing — it subtracts on every dataset

  | dataset | C1 | C2 | C3 | C4-best | best strategy |
  |---|---|---|---|---|---|
  | eu_cc | .82 | **.87** | .56 | .74 | C2 |
  | banksim | .91 | **.91** | .86 | .67 | C2≈C1 |
  | paysim | .91 | **.98** | .13 | .56 | C2 |
  | baf | .16 | **.18** | .17 | .016 | C2 |
  | fifar | .22 | **.22** | .20 | .05 | C2≈C1 |
- observed ordering: C2 ≥ C1 on all 5 datasets; both retrieval conditions (C3, C4) fall below C1/C2 everywhere **except baf-C3** (0.168 vs C1 0.163); the C3-vs-C4 order is **dataset-dependent**, not a fixed C3>C4 — C4-best beats C3 on eu_cc (0.739 vs 0.555) and paysim (0.563 vs 0.128), while C3 beats C4 on baf/fifar; the full C2≥C1>C3>C4 chain holds only on banksim and fifar. Worst condition is dataset-specific: **C3** on paysim (0.128), C4 on baf/fifar — this **inverts** the draft's expected C1<C2≈C3<C4-best [finding: results/runs]
- note: the pivot uses the better FTM, so it *understates* the negative result — the un-pivoted per-model view is more strongly negative. Worst eu_cc C3 cell: tabiclv2 seed123 = 0.087 (this is **not** the global C3 minimum: four paysim C3 cells score lower, min paysim tabpfn_3 seed42 = 0.0225). Largest hidden C4 divergence: paysim C4 (tabpfn_3 = 0.073 total collapse vs tabiclv2 = 0.563, a 0.49 gap masked by the better-of-two cell). The better-of-two pivot rule used throughout also mildly inflates the C2/SOTA numbers exactly as it does the C4 cells above (§6.11) [finding: results/runs]

## 6.9 Best FTM and version progression
- best FTM = tabiclv2 ≈ tabpfn_3 (tied within seed noise) > tabpfn_26 ≈ tabpfn_25 > tabpfn_v2; context size dominates ranking, not architecture/version [finding: eda §4]
- v2→2.5→2.6→3 not monotone: v2→2.5 the only big step (mostly 10k→50k); 2.5→2.6 a wash; 2.6→3 regresses on hardest **baf** (holds at n=3 seeds: C1 0.1624→0.1591, C2 0.1810→0.1787), with **fifar (n=1)** consistent in direction (0.2169→0.2121, 0.2237→0.2194) [finding: eda §4]
- fragility is a retrieval artifact: tabpfn_3 ≈ tabiclv2 under C1/C2 but diverges sharply under C4 — 42/65 tabpfn_3 C4 runs hit recall@1%FPR=0 vs 0/65 tabiclv2 (formal grid; probability ranking more sensitive to duplicated/degenerate context) [finding: results/runs, C4 recall_at_1fpr==0 by model]
  - the failures concentrate on the hard/duplicated cells: tabpfn_3 recall@1%FPR=0 on baf 100%, fifar 100%, paysim 93%, banksim 53%, eu_cc 0%; tabiclv2 0% everywhere [finding: results/runs, C4 recall_at_1fpr==0 by dataset×model]

## 6.10 Computational cost
- paradigms invert: GBDTs pay one-time fit, score free; FTMs fit near-free, inference dominates (tabpfn_25 ~1,159s ≈ 86× xgboost; tabiclv2 105s) [finding: eda §6]
- FTM predict_s scales with rows×context×feature-dim, not dataset size; ~30-feat ≈ 3× low-dim per row [finding: eda §6]
- RAP overhead modest: setup_s ~6.7s + fit_s ~2.7s; predict dominates and is condition-invariant — but **retrieval/context-build time (kNN query + sampling) was not separately timed into any recorded column** (not setup_s, fit_s, or predict_s) in this grid, so these are **lower bounds** on RAP overhead; wall-clock gaps between consecutive C3/C4 parquets exceed the recorded sums, confirming the under-count. The code now folds `builder.build()` into `setup_s` going forward (fixed 2026-07-06, re-timing only, no rerun/version bump needed) [finding: results/runs, setup_s/fit_s/predict_s by condition]
- [finding: scratchpad/eda/fig5_efficiency.png]

## 6.11 External SOTA comparison
- FTM+C2 matches or exceeds methodologically-sound external SOTA with **no per-dataset hyperparameter tuning** (label-supervised context, frozen weights, n_estimators=4 — a deliberately reduced ensemble, §5.8) [finding: sota-comparison]
  - clean like-for-like: eu_cc 3-seed mean **0.869** (tabpfn_3: 0.897/0.888/0.822; tabiclv2 mean 0.861) vs external 0.867 — a **statistical tie** (seed-sd far exceeds the 0.002 gap); the single best run reaches 0.897 (tabpfn_3, seed 7), but a best-seed value is not a fair comparator, so the verdict for eu_cc is **at par with sound SOTA**, not above it; fifar R@5%FPR 0.568 ≈ 0.579 (at-par) [@alves2023fifar_learning_to_defer]
  - **10-seed update (extension slice, §5.9):** eu_cc best-C2 mean revises 0.869 → **0.866** (tabpfn_3, n=10, sd 0.032), 0.001 below external 0.867 — verdict unchanged, a statistical tie; most 10-seed cell means sit slightly below the 3-seed means (the anticipated lucky-seed regression), eu_cc C1 is the main exception (rises ~0.02–0.03); paired C2−C1 deltas positive on all 5 datasets for both models, CI excluding zero everywhere except banksim×tabiclv2 [finding: results/runs, seed-extension slice, bootstrap analysis 2026-07-19]
  - **above** sound SOTA: banksim 0.910 vs 0.896 (our grouped split is harder → conservative, so if anything an understatement); paysim 0.986 vs 0.854 (note: account-level comparator, different task, so not fully like-for-like); baf R@5%FPR 0.553 ≈ 0.50–0.55 (figure-read, at-par) [@jesus2022baf_arxiv]
  - the better-of-{tabiclv2, tabpfn_3} pivot rule used throughout also mildly inflates these numbers relative to either single model (§6.8) — both eu_cc model means are shown above for transparency
  - all SOTA-table values are 3-seed means; the 10-seed CI extension (§5.9, in progress 2026-07-18) will restate them as 10-seed means with bootstrap CIs — verdicts (above / at-par) to be re-checked against the CIs, not the point deltas
- much published imbalance SOTA is unsound: ubiquitous ~0.99 AUPRC = SMOTE-before-split leakage [@chawla2002smote] [finding: sota-comparison]
- Recall@k%FPR essentially never reported on eu_cc/paysim/banksim → we fill a genuine gap [finding: sota-comparison]

---

# Chapter 7. Discussion

## 7.1 Headline: ratio control wins, retrieval-relevance does not
- the win is RATIO CONTROL delivered by stratified balancing of a random context, not relevance; retrieval under heavy imbalance subtracts [finding: c3c4-grid-running]
- a clean, mechanistically-explained null on a published hypothesis (Nagler localization / LoCalPFN frozen retrieval) is itself the contribution: negative answer to RQ3; positive answer to the ratio half of RQ4 (C2); negative answer to the retrieval half (C4)
- even a cherry-picked, finer-grained RAP config only **ties** C2 on eu_cc and loses everywhere else (eu_cc RAP-best 0.866 ≈ C2 0.869; banksim 0.898 < 0.909; paysim 0.693 ≪ 0.983) → RAP is not salvageable by tuning [finding: results/runs_rapds]

## 7.2 Why retrieval collapses (dominant mechanism: lost negative diversity — attributed per dataset)
- **per-dataset attribution (not one universal principle):** on the 3-dataset ratio-matched ablation (eu_cc, banksim, paysim), lost negative diversity — kNN retrieval concentrating the context in the dense fraud/legit overlap and stripping the diverse easy-legitimate background — is the **dominant driver where retrieval collapses** (paysim ≈−0.72, banksim ≈−0.03 legit-flip delta); on eu_cc the same locality is mildly **positive** for legit selection (≈+0.01–0.02) and the C3/C4 collapse there is instead carried by **duplication degeneracy** (H3); baf/fifar sit at a separability floor where neither mechanism moves much. Finer G recovers some ground (each tiny cluster's retrieval re-approaches a near-random balanced context, whose limit is C2 itself) but the recovery pattern also differs by dataset (§6.7) [finding: results/diag; results/runs_rapds; results/runs_ablation]
- **causal isolation (H1 — the RK-vs-KK contrast, both G=32):** flipping only the *legit* selector from random to kNN, holding G=32 and ratio/context fixed, collapses paysim from ~0.97 (RK) to ~0.48 (KK); mild on banksim; mildly positive on eu_cc (per-dataset detail above). RR-vs-RK (the fraud-selector check) does almost nothing. The granularity confound is **closed** (2026-07-18): the RR@G=32 arm matches both the global RR and RK@G=32 within seed noise on all 3 datasets (§5.3), so per-group granularity with random contexts is directly confirmed neutral and RK-vs-KK is a clean single-factor isolation of the legit selector. The natural-ratio arms corroborate the attribution on an independent contrast (§6.5): kNN selection at matched natural ratio still collapses paysim (0.897→0.033) and hurts banksim (0.901→0.785) but mildly helps eu_cc (0.834→0.845) [finding: results/runs_ablation]
- **direct quantification:** `legit_score` falls monotonically as grouping refines (banksim 0.070→0.005; eu_cc ~0.02 mean→0.001 from G8→G64) [finding: results/runs_rapds]; banksim tabiclv2 C4 inflates 46.6% of legits >0.5, dropping AP 0.96→0.81 while ROC only 0.998→0.966 (loss concentrated in top-rank precision = genuine ranking damage); injecting just 25% random-legit *anchors* into the retrieved context drops legit_score 0.42→0.004 and lifts AP 0.69→0.82 — direct causal evidence; C2 (also balanced) does NOT inflate (0.77% of legits >0.5 vs C4's 46.6%) → the win is real, not a balance artifact [finding: results/diag]
- **H2 — not fraud starvation (n=2):** C3 keeps ~39 frauds/group (eu_cc) and ~276 (paysim) *on average*, yet collapses. The 6 paysim C3 runs split 2 fraud-rich vs 4 fraud-starved: the 2 fraud-rich runs — seed 7 tabpfn_3 PR-AUC 0.2978 at ~832 mean frauds/context (run `931d287e9c22`) and tabiclv2 0.2110 at ~820 (run `ee59299a0dcb`) — still collapse relative to C1 (~0.91), versus the 4 fraud-starved runs at 0.0225–0.0629 (~1 fraud/context). Even fraud-rich retrieved contexts stay collapsed, supporting H2, but the fraud-rich evidence is explicitly **n=2** [finding: results/runs, C3 context_fraud_n]
- **the residual gap:** even at G64, where legit inflation is essentially gone (banksim 0.005), PR-AUC still asymptotes ~0.11 below C2 (0.80 vs 0.91) → loss of negative *diversity/coverage* (not just mean legit score) carries the remainder; lost-negative-diversity is the unifying cause, not the sole scalar [finding: results/runs_rapds]
- **H3 duplication degeneracy (eu_cc-specific):** with-replacement quota-fill injects many identical fraud rows (eu_cc dup 6.2×→62× over the ratio sweep), independently confirmed by the random-C2 duplication cliff (§6.4); banksim/paysim collapse at dup ≤4× so duplication is not their cause [finding: results/runs; results/runs_c2grid]
- **per-dataset symptom, not one cause:** banksim/eu_cc show legit inflation; on paysim retrieval "cannot separate it at any G" despite its high *global* separability under C1/C2 — we describe this descriptively as a *local/retrieval* separability failure, but the "trapped neighbourhood" framing is **speculative** (no backing statistic isolates it as a distinct fourth mechanism; it may simply be the same lost-negative-diversity effect at its most severe on this dataset); baf/fifar sit at a global separability floor (~0.16/0.20 under all conditions) where neither random nor retrieval separates [finding: results/runs_rapds; eda §5]
- the paysim C4 ratio-rise is **model-specific** (it lifts tabiclv2 0.48→0.56 but tabpfn_3 collapses 0.073→0.010 on the *identical* context) → governed by tabpfn_3 ranking fragility (§6.9), not a dataset-level legit-dilution effect
- forcing finer G recovers but asymptotes BELOW C2: banksim G8→64 0.30→0.80 (C2 0.91); eu_cc 0.50→0.74 (peak 0.76 at G32; C2 0.87); paysim →0.31 (C2 0.98) [finding: results/runs_rapds, PR-AUC by n_groups]
- killed alternative designs: euclidean<cosine; random-fraud<kNN-fraud; RobustScaler no help; diversified retrieval only ties random; global-all-fraud + local-legit among the worst [finding: c3c4-grid-running; results/runs_rapds]
- literature-consistent (as analogy, scoped in §8.1): similarity-weighted majority vote, and — borrowed out-of-domain — minority "bad hubs"; the directly-evidenced legit-inflation account above is the primary support, not these [@tomasev2013minority_hubs; @pham2026aware]

## 7.3 Reconciling with theory and prior work
- Nagler's localization theory predicts relevance reduces bias, but validates post-hoc kNN on *balanced synthetic* (SCM) data at n≈1,000; on heavy-imbalance fraud it is dominated by lost-negative-diversity / legit-inflation (and, on eu_cc, duplication) — the out-of-design gap is the point [@nagler2023statistical_foundations_pfn; @pham2026aware]
- vindicates McDowell's caution against context oversampling (and his downsampling-to-50/50 corroborates the *direction* of our C2 result, not the operating point, §3.5); the direct C2-vs-thresholding experiment was never run (§8.2). The §2.2 invariance argument covers the thresholding half of their design (a fixed-prior correction is provably monotone and cannot move ranking metrics — confirmed from Saerens' Eq. 4), but their downsampling arm re-fits and *can* move ranking metrics; our design covers the axes their study lacked (relevance, ranking metrics, heavier imbalance) — a same-protocol head-to-head remains future work (§8.2) [@mcdowell2026correcting_imbalance_pfn]
- shows Gao's relevance+ratio coupling does **not transfer** from LLM-text/accuracy to the tabular heavy-imbalance frozen-FTM/ranking-metric regime (a failed transfer, not a refutation of Gao's own result) [@gao2025imbalanced_annotations]
- C2's gain is on ranking metrics, so by the §2.2 argument it cannot be the calibration artifact post-hoc prior-correction would predict — it reflects a re-fit posterior (our interpretive synthesis) [@saerens2002adjusting_priors]

## 7.4 Practical guidance
- under heavy imbalance, cheap stratified context balancing is robust and SOTA-competitive at no per-dataset tuning — relevance is unnecessary; "objective/balance > elaborate context construction" (Tanna corroborates this on milder-imbalance ROC; we extend to fraud-grade imbalance on ranking metrics) [@sun2025extreme_imbalance_baf; @tanna2026data_presentation_resampling]
- the optimal in-context fraud ratio is a broad (near-)plateau (full sweep in §6.4); r≈0.10 is within ~0.006 of every dataset's peak → a safe universal default (lower also minimizes duplication; 0.006 is below seed-sd, §6.4), and ratios ≥0.20 hurt; context size helps only at low ratio. Aggregation caveat carried from §6.4: the by-ratio marginal is a mean over models+seeds — under the better-of-FTM pivot the eu_cc peak shifts to r0.02, and the default's own selection is subject to the same test-set-selection caveat as the ratio sweep generally (§5.9/§8.1) [finding: results/runs_c2grid, 810 cells]
- **deployment cost note:** our retrieval is the cheapest possible implementation (brute-force in-memory kNN over preloaded, pre-scaled arrays), so the benchmark *understates* RAP-style construction cost in production, where every context build is a round trip against a feature store / database / data lake. This compounds the accuracy verdict: fine-grained (per-batch or per-query) context construction multiplies retrieval round trips and forfeits the context-encoding amortization (§5.3), so it is defensible only where per-decision accuracy dominates and latency/throughput are secondary concerns — and our results indicate the accuracy gain is absent in this regime anyway. C2's single global balanced context, by contrast, is built once per model refresh, adds no per-prediction data-access cost, and is the ideal KV-cache workload (encode once per refresh; every transaction thereafter pays only the query cross-attention term, §5.3), so the serving-cost gap between C2 and fine-grained retrieval widens further under a cached deployment

---

# Chapter 8. Limitations and Future Work

## 8.1 Limitations
- the HEADLINE is a negative result for the RAP relevance hypothesis — bounded to this design space (frozen FTMs, kNN/cosine retrieval, <1.5% fraud, ranking metrics), not absolute [finding: c3c4-grid-running]
- **statistical power:** 3 seeds per dataset (fifar = 1) — effects are reported with seed-spread and cluster-level CIs (§5.9), but small per-dataset deltas (banksim C2 +0.001; fifar) are not significant at this n and are read as directional only
- C3 does not hold the fraud ratio fixed vs C1 (sign-varying: depletes eu_cc, enriches banksim); we **close this** with a ratio-matched relevance ablation (§5.3, §7.2) that isolates the legit-selector as the cause, so the relevance reading no longer rests on the confounded C3-vs-C1 contrast alone [finding: results/runs_ablation]
- C4 ratio sweep confounds duplication, context size, and locality jointly at high ratios → collapse cannot be cleanly attributed from C4 alone (the ablation and the §6.4/§6.7 sweeps disentangle these) [finding: results/runs, C4 sweep design]
  - eu_cc declines with dup 6×→61× (duplication harm) BUT baf/fifar collapse already at dup=1 → duplication is not the sole cause [finding: results/runs]
- per-group cross-calibration caveat: multi-group C3 mixes groups with different base rates (a non-monotone score map, so the §2.2 ranking-invariance shield does not cover cross-group C3); **C4 is not immune either** — equal per-group *ratio* does not make the pooled score map a single global monotone rescale, since per-group *retrieval* (the mechanism §7.2 blames) changes the scorer per group. The §2.2 shield is exact only for the global constructions, C1 and C2; both C3 and C4 sit outside it (costless to note, since C4's result is negative anyway) [finding: c3c4 design]
- **C2 is partly a replication** of Tanna et al. 2026 (balanced context helps TFMs); the surviving novelty is the heavy-imbalance regime, the ranking-metric (discrimination-not-calibration) argument, and the C1–C4 decomposition [@tanna2026data_presentation_resampling]
- **temporal leakage:** time kept as a feature under a random split (§5.5) may give an optimistic upper bound on ranking metrics for paysim/eu_cc vs a temporal deployment; the BAF-by-month ablation (§8.2) is the partial check and should bound the SOTA claim
- **baseline fairness:** GBDTs use fixed strong defaults with early stopping, not per-dataset tuning (FTMs are zero-shot) — a tuned-GBDT comparison is not performed, so the FTM-vs-GBDT result is "vs fixed-default trees," not "vs tuned SOTA trees" (§5.4)
- **exploratory-search / test-set-selection:** the ratio sweeps (§6.4, §6.7) score every configuration on the same fixed test splits, so the recommended r≈0.10 default and the RAP design-space's best cherry-picked configuration are both **selected on test data** (implicit multiple testing); the eu_cc RAP-best-ties-C2 comparison (§6.7/§7.1) is not distinguishable under this shared search-induced optimism — the "cherry-picked" caveat applies to the C2 side too, not only to RAP
- **dataset provenance:** only eu_cc is real transaction data; banksim and paysim are agent-based simulations, fifar is synthetic (BAF-derived), and baf is a semi-synthetic generative suite (§4.1) — mechanism claims about "the negative manifold of real fraud data" (§7.2) rest on 4 of 5 formal-grid datasets being synthetic or semi-synthetic
- minority-hubs is borrowed out-of-domain (high-dimensional image data; intrinsic, not nominal, dimensionality governs onset) → used as a speculative analogy, not a mechanism; the directly-evidenced legit-inflation account (§7.2) carries the weight [@tomasev2013minority_hubs]
- formal theoretical weight rests on Nagler + LoCalPFN (directly tabular); LLM-ICL Bayesian/kernel accounts are mechanistic analogy, not tabular proof [@nagler2023statistical_foundations_pfn; @xie2022explanation_icl_bayesian; @han2025kernel_regression_icl]
- single GPU → n_estimators=4 (not the library default 8), so absolute FTM numbers (incl. the SOTA comparison) are at a reduced ensemble; ad-hoc runs at the default are not comparable [finding: CLAUDE.md]
- the cosine-vs-euclidean metric A/B is **settled** by the design-space sweep (cosine > euclidean, §6.7), not left open; broader learned-metric retrieval remains future work
- C2 reaches true 50/50 only on tabpfn_v2; 48–50k models cap at available unique fraud (C1/C2 do not oversample), so cross-model C2 comparisons are at different effective ratios [finding: eda §7]
- silhouette-chosen G is mostly degenerate (G=2) on big imbalanced sets → per-group retrieval has weak geometric justification; yet G=2 still hurts and forcing finer G (§6.7) still loses to C2 → the harm is not over- or under-fragmentation of the grouping but the retrieved context itself — locality of the legit pool where retrieval collapses (paysim, banksim), duplication on eu_cc (§7.2 per-dataset attribution) [finding: results/runs, n_groups; results/runs_rapds]
- evidential-base hygiene: several load-bearing sources are recent/unrefereed 2026 preprints (McDowell workshop; DistPFN withdrawn-ICLR; AWARE/Tanna preprints); cc_2025 excluded as degenerate (the formal evaluation is substantively 5 datasets); CV/time-series direction out of scope; `% VERIFY` bib items [@dalpozzolo2015calibrating; @lopezrojas2014banksim; @grinsztajn2026tabpfn3_technical_report; @priorlabs2026tabpfn26_model_card] need metadata confirmation [finding: thesis-draft]

## 8.2 Future work
- learned imbalance-aware retrieval embedding (naive distance ≠ predictive relevance under imbalance) [@pham2026aware]
- interpolation-based (SMOTE-style) minority synthesis vs with-replacement duplication, to break the duplication-degeneracy on the scarcest-fraud set (eu_cc) [@chawla2002smote]
- head-to-head vs post-hoc thresholding (McDowell) and analytic label-shift (DistPFN) on the same ranking metrics [@mcdowell2026correcting_imbalance_pfn; @lee2026distpfn_label_shift_tabpfn]
- temporal-split robustness via BAF-by-month (also bounds the time-as-feature leakage caveat, §5.5); a tuned-GBDT comparison; shared retrieval index across ratios (deferred optimization)

---

# Chapter 9. Conclusion
- RAP's retrieval-relevance lever is a NEGATIVE result: it does not help and usually hurts ICL tabular FTMs under heavy class imbalance (C3<C1, C4<C2), with a quantified symptom (legit-score inflation) and an attributed, per-dataset cause — lost negative diversity dominates where retrieval collapses (paysim, banksim); duplication degeneracy dominates on eu_cc [finding: results/runs; results/runs_ablation; results/diag]
- the salvage is robust and practical: stratified context balancing (C2) is the best strategy on every dataset and is competitive with sound external SOTA at no per-dataset tuning — above sound external results on paysim and banksim, at par on eu_cc (statistical tie), baf, and fifar (the effective optimum is r≈0.10, not literal 50/50) [finding: results/runs; results/runs_c2grid; sota-comparison]
- the study confirms the strong-random-baseline and ICL-rebalancing-caution priors, shows the relevance+ratio coupling does not transfer to this regime, and shows the balancing gain is genuine discrimination — a clean, honest contribution to context construction for tabular foundation models [@feuer2023scaling_tabpfn; @gao2025imbalanced_annotations; @mcdowell2026correcting_imbalance_pfn; @saerens2002adjusting_priors]

---

## Appendix — open items flagged during synthesis
- **Figures for the negative-result core — the two blocking ones now EXIST (created 2026-07-06 via `scratchpad/eda/make_figs_review.py`):** (1) `fig6_condition_pivot.png` — the C1/C2/C3/C4 pivot heatmap (§6.8; cells verified against the table to 3dp); (2) `fig7_mechanism_panel.png` — the legit_score-vs-G panel + the banksim/tabiclv2 anchor-injection diagnostic (§7.2). Still to plot: the C4 ratio-sweep line plot, the G-recovery curve vs C1/C2 baselines (§6.7), and the tabpfn_v2/paysim 0.576→0.955 motivation panel. Data all exist (results/runs, runs_rapds, runs_ablation, diag).
- **Now folded in (was listed as missing):** `results/runs_ablation` (the ratio-matched relevance ablation) — promoted into §5.3/§7.2; the corresponding §8.1/§8.2 "still open" items were removed. **Extended 2026-07-18** with the closing batch (72 new cells, dir now 196 rows): RR@G=32 × {0.05, 0.10, natural} + KK@natural × G=32 × 50k — closes the RR-granularity confound (review BLK-3) and resolves the methods natural-ratio TODO; results folded into §5.3/§6.5/§7.2.
- **In progress (2026-07-18):** the 10-seed CI extension (§5.9; 148 runs, tabpfn_3/tabiclv2 × C1/C2 × 5 datasets). On completion: compute per-dataset bootstrap CIs + seed-paired C2−C1 deltas, restate §6.11 verdicts, and update §6.3's "CI to be reported".
- **In progress (2026-07-20): two C2-sweep appendix extensions**, both running through `scripts/run_c2_grid.py` (systemd user unit `c2grid-resume`) into `results/runs_c2grid/`; grand total 1092 cells when complete. (1) *Large-context extension:* context {100k, 200k} × ratio {0.02, 0.05, 0.10, 0.20} × {tabiclv2, tabpfn_3} × {eu_cc, banksim, paysim} × 3 seeds (144 cells). Purpose: turn the "50k is a design budget, not the model cap" claim into a measured curve (does C2 PR-AUC keep climbing past 50k, and does the ratio plateau shift with size). Caveats for the writeup: tabpfn_3 stays in-spec (checkpoint cap 1M), but tabiclv2 beyond its 48k pre-training bound is extrapolation and must be flagged; eu_cc large-context cells are duplication-heavy (dup ≥ 25× at 100k/r=0.10, ~102× at 200k/r=0.20), which makes the duplication cliff visible by design. (2) *n_estimators sweep:* n_est {1, 2, 4, 8, 10, 12} at the fixed plateau recipe (r=0.10, 50k) × 2 models × all 5 formal datasets, 13 units (156 cells; the 18 n_est=4 cells are shared with the existing grid, context rows identical across n_est by construction). Purpose: audit the settled n_estimators=4 grid decision against the package default 8 and locate the saturation shoulder. On completion: write both up as appendix sections with figures, and update §6.4's "810 cells" count to reflect the extended grid.
- **Citations narrowed/corrected (per supervisor review):** Saerens (don't carry the metric-invariance fact — cite Fawcett/Davis; flag the in-context-vs-in-weight step as our synthesis); Gao ("failed transfer", not "refutes"); Ye 2025 (a scaling result, not kNN retrieval); LoCalPFN (frozen retrieval helps — a regime gap, not an empty niche); Tomašev hubness (out-of-domain analogy). See `scratchpad/thesis-outline-review.md`.
- **Numeric fixes applied:** paysim C4-best .65→.56 (§6.6/§6.8); fifar C1 .21→.22 (§6.8); fragility tabiclv2 0/66→0/65 (§6.9); eu_cc C2 "band ≥0.831" softened (§6.4); legit_score eu_cc 0.051 labelled as worst-cell (§6.7); §7.2 eu_cc G64 endpoint 0.74 (peak 0.76 at G32); study label ds_designspace→rap_designspace (§6.7).
- **Dropped (keys not in `thesis-refs.bib`):** `shukor2024multimodal_icl`, `mccarter2024what_tabpfn_learned` — the duplication/vote-collapse mechanism is stated as our own observation (or add the keys to the bib to cite the literature).
- **Bib metadata:** clear every `% VERIFY` flag (dalpozzolo, lopezrojas, the 2026 arXiv ids, page ranges) before submission; record DistPFN as withdrawn-ICLR and McDowell as a workshop preprint.
- **Bib gaps worth filling** (would strengthen §2.2/§3.5 label-shift argument; not currently in bib): label-shift estimators (BBSE / unified label shift). Add entries before citing.
