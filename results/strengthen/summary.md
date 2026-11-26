# Strengthening experiments (23 Nov 2026)

Code: `code/scripts/strengthen/` (common.py, s0–s6). Raw results: CSVs in this folder.
All probes were rebuilt with the exact procedure of `06_linear_probe.py`. The rebuilt predictions match
the paper's prediction files to <2e-7 for every encoder, direction and seed (s0_cache_and_check.py).
Main direction unless stated: HMU (slide-grouped) -> GasHisSDB. BA = balanced accuracy of the ensemble-mean
probability at 0.5 (as in Table III). BA@80 = BA on the 80% least uncertain patches. Mean ± SD over 3 seeds.

## 0. Bug found: ResNet50 "MC dropout" had no effect
timm applies head dropout with `training=self.training`; in eval mode this is off, so all 20 passes were
identical (max SD across passes = 0 in every CNN prediction file). The CNN uncertainty reported in the paper
(error-detection AUROC 0.66–0.68) is therefore the single-pass softmax entropy, not MC dropout.
Fixed in `code/src/gcx/uncertainty.py`. Re-run with `Fix CNN uncertainty.command` (Mac, ~1–2 h).
Point predictions (prob_det), AUC, BA and all foundation-model results are unaffected.

## 1. Tumour prevalence of the target set (n = 4,000 per subsample, 5 repeats x 3 seeds)
UNI, BA:
| method | 10% | 30% | 50% | 70% | 90% | natural (39%) |
|---|---|---|---|---|---|---|
| none | 0.633 | 0.634 | 0.633 | 0.633 | 0.634 | 0.633 |
| target std | 0.777 | 0.830 | 0.846 | 0.824 | 0.778 | 0.842 |
| centre only | 0.767 | 0.827 | 0.845 | 0.822 | 0.776 | 0.840 |
| class-balanced std | 0.798 | 0.846 | 0.844 | 0.828 | 0.784 | 0.848 |
UNI, BA@80 (20% referred): none 0.57–0.61; target std 0.794 / 0.878 / 0.906 / 0.866 / 0.790 / 0.899.
Phikon, BA: none 0.69; target std 0.750 / 0.780 / 0.788 / 0.779 / 0.754 / 0.787.
- Target standardisation beat no adaptation in 36/36 (encoder, prevalence, seed) cells, but loses up to
  ~0.07 BA at 10% or 90% tumour, because the target mean then moves towards one class.
- A class-balanced variant (pseudo-labels, 2 rounds) recovers part of this at low prevalence (0.777 -> 0.798 at 10%).
- Reverse direction (GasHisSDB -> HMU): no method helps much (BA ~0.69–0.72); standardisation slightly hurts at 90%.

## 2. How many unlabelled target images are needed (stats from n images, evaluated on all 33,284)
UNI BA: n=10 0.812, 25 0.833, 50 0.834, 100 0.839, 200 0.841, 1,000 0.841, all 0.842.
UNI BA@80: n=10 0.859, 25 0.888, 100 0.895, 200 0.898, all 0.899. Phikon converges the same way (0.769 -> 0.787).
=> ~100–200 images suffice (within 0.003 of using all), supporting GastroMap's 100-tile minimum.

## 3. Oracle threshold (uses target labels) vs label-free methods, full target set
| | UNI H->G | Phikon H->G | UNI G->H | Phikon G->H |
|---|---|---|---|---|
| AUC (none) | 0.920 | 0.862 | 0.777 | 0.768 |
| BA none | 0.634 | 0.688 | 0.691 | 0.708 |
| BA centre only | 0.839 | 0.783 | 0.715 | 0.709 |
| BA target std | 0.842 | 0.787 | 0.714 | 0.703 |
| BA class-balanced std | 0.845 | 0.788 | 0.709 | 0.702 |
| BA CORAL | 0.728 | 0.715 | 0.654 | 0.651 |
| BA oracle threshold (none) | 0.844 | 0.781 | 0.723 | 0.712 |
- HMU->GasHisSDB: label-free standardisation reaches the oracle-threshold BA (0.842 vs 0.844 for UNI;
  0.787 vs 0.781 for Phikon). The failure without adaptation is almost entirely a misplaced threshold.
- GasHisSDB->HMU: even the oracle threshold gives only ~0.72, so the limit there is ranking (AUC ~0.77),
  which no threshold or standardisation can fix. This explains the small reverse-direction gain.
- Most of the benefit comes from centring alone (shift), scaling adds little.

## 5. Significance (bootstrap; GasHisSDB has no slide IDs, so patch-level CIs are optimistic)
| comparison | diff | 95% CI | p |
|---|---|---|---|
| leakage gap ResNet50 (block bootstrap, 2000) | +0.025 | 0.001–0.049 | 0.038 |
| leakage gap Phikon | +0.046 | 0.018–0.076 | 0.001 |
| leakage gap UNI | +0.036 | 0.007–0.068 | 0.014 |
| cross AUC UNI − ResNet50 | +0.228 | 0.223–0.233 | <0.001 |
| cross AUC UNI − Phikon | +0.058 | 0.055–0.061 | <0.001 |
| BA UNI target std − none | +0.208 | 0.204–0.211 | <0.001 |
| BA@80 UNI target std − none | +0.310 | 0.306–0.314 | <0.001 |
| BA UNI target std − CORAL | +0.114 | 0.110–0.118 | <0.001 |
| UNI, no adaptation: BA@80 − BA (referral effect) | −0.046 | −0.048 to −0.044 | <0.001 |
Leakage gap positive in every seed for every model (ResNet50 0.013/0.030/0.031; Phikon 0.033/0.062/0.042;
UNI 0.015/0.049/0.045). Without adaptation, referral significantly LOWERS balanced accuracy.

## 6. Tool: per-slide standardisation erases the slide's tumour fraction (fraction_bias.csv)
UNI, subsets of 2,000 GasHisSDB patches. Predicted tumour share (ensemble mean >= 0.5):
| true share | 10% | 30% | 50% | 70% | 90% |
|---|---|---|---|---|---|
| per-slide standardisation | 46% | 44% | 47% | 51% | 51% |
| no standardisation | 3% | 9% | 14% | 19% | 24% |
=> GastroMap's earlier "58.1% tumour" for the TCGA slide is not interpretable.

## 7. Fix: hospital-level calibration (site_calibration.csv)
Statistics from a separate pool of the same site (n_cal = 500 or 2,000; 20/40/60% tumour), applied to test sets of
10–90% tumour. With a 40%-tumour calibration pool: BA 0.84–0.85 at every test prevalence (per-slide: 0.78–0.84),
and predicted share 25/39/52/66/79% for true 10/30/50/70/90% (tracks the truth, compressed towards the middle).
The calibration pool's own composition shifts the estimate (20% pool over-calls, 60% under-calls), so the
calibration slides should be a representative mix. 500 calibration tiles are as good as 2,000.
Implemented in code/tool/app.py: hospital calibration (>= 3 slides, 300 tiles each, saved in data/calibration/);
per-slide fallback hides the tumour %; the slider shows a range in per-slide mode (per_slide_curves.npz:
BA@80% kept 0.79–0.90 depending on tumour share). Old version kept as code/tool/app_v1_backup.py.

## 8. One metric definition (23 Nov)
07_evaluate.py now uses the ensemble/MC mean ("prob") as the point prediction for every model (option --point).
Old results kept in results/_previous_prob_det/ and paper/_previous_prob_det/. Tables II–III and Figs 2–4 regenerated.
Changes vs the paper draft: Phikon HMU-grouped in-domain AUC 0.878 -> 0.881, cross AUC 0.862 -> 0.869;
UNI HMU-random in-domain 0.924 -> 0.926; GasHisSDB->HMU cross AUC 0.768/0.777 -> 0.772/0.783;
UNI cross BA (Table II) 0.654 -> 0.634, now equal to Table III. CNN numbers will change slightly after the MC-dropout rerun.
Note: AUC columns in prevalence_*.csv and sample_size_*.csv were computed from the single probe (not used in the paper).

## 9. TCGA "percent tumour cells" is not a usable check of the tool
The GDC records percent tumour cells only for the frozen top/bottom sections (TS/BS), not for the diagnostic
FFPE slide (DX) that we analyse, and the values are 70–100% for all six TCGA-HF cases. So it cannot validate the
tool's tumour fraction. TCGA-HF has 6 diagnostic slides in total (A5NB, 7131, 7132, 7133, 7134, 7136).

## 10. Few-label calibration (fewlabel_calibration.csv; s11 curve in per_slide_curves.npz)
No feature standardisation; the decision threshold is reset from k labelled tiles (k/2 tumour, k/2 normal).
BA at 10–90% tumour: k=10 0.81 (10th pct 0.74), k=20 0.82, k=50 0.83 (0.79), k=100 0.84 (0.82).
Predicted share for true 10/50/90%: 20/46/72% (k=50). With the threshold applied as a logit shift, referral works:
BA on the 80% kept 0.88–0.89 at every prevalence (10th percentile over label sets 0.86). Added to GastroMap as
"Few-label calibration" (>= 10 tumour + 10 normal tiles; 25 + 25 recommended).

## 11. TCGA-HF slides for the demo
Downloaded via the GDC API on 23 Nov 2026; MD5 checked against the GDC records:
TCGA-HF-7132 (225,387,000 B), TCGA-HF-7133 (341,809,264 B), TCGA-HF-7136 (355,459,472 B), plus TCGA-HF-A5NB.
Planned demo: hospital calibration from 7132/7133/7136, then analyse A5NB (leave-one-out).

## 12. CNN MC-dropout re-run done (23 Nov, ~04:35)
Fixed predictions written for all 6 ResNet50 runs; old files in runs/cnn_*/pre_mcfix/. prob_det identical to before
(max diff 0). MC passes now differ (mean max SD 0.055). Error detection on GasHisSDB: MC entropy 0.655 (grouped) /
0.676 (random), the same as single-pass softmax entropy (0.656 / 0.677); mutual information is worse (0.59).
=> The paper's CNN numbers stand; only the method wording changes (MC dropout now really applied, and it does not
beat plain softmax entropy). CNN AUC/BA from the MC mean are unchanged to 3 decimals.

## 13. Significance re-run with the unified metric (ensemble/MC mean)
Leakage gap: ResNet50 +0.025 (95% CI 0.001–0.049, p=0.040); Phikon +0.041 (0.011–0.072, p=0.004);
UNI +0.037 (0.007–0.069, p=0.018). Cross AUC UNI−ResNet50 +0.229; UNI−Phikon +0.051; Phikon−ResNet50 +0.178;
target-std vs none and vs CORAL as before; all p<0.001. Referral without adaptation still lowers BA (−0.046).

## 14. Calibrated demo (A5NB, hospital calibration from 7132/7133/7136, 900 tiles)
4,726 tiles, 418 s. Tumour fraction 89.3% (per-slide mode had given 58.1%). Normal-mucosa fragment still called
normal. 945 tiles referred; unfamiliar 39% (warning). TCGA reports 90% tumour cells for this patient's frozen top
section (not the analysed FFPE slide): consistent, not a validation. Per-slide outputs kept in results/gastromap/per_slide_mode_A5NB/.

## 15. Tier 2 set-up (23 Nov, 05:30)
Stain reference: data/stain_reference/hmu_train_s0_mosaic16.png (8 TUM + 8 NOR HMU training patches, seed 0;
sources listed beside it). Macenko on 300 GasHisSDB patches: 3.9 ms/patch, 10/300 near-empty patches left unchanged;
mean optical density moved from 0.04/0.20/0.04 to 0.36/0.61/0.36 (HMU reference 0.44/0.91/0.53).
AdaBN: code/scripts/15_adabn_cnn.py (BatchNorm stats re-estimated on 2,000 random unlabelled GasHisSDB images).
Launcher: "Run Tier 2 baselines.command" (resumable). Analysis: code/scripts/strengthen/s12_tier2.py -> tier2.csv.

## 16. Tier 2 results (HMU grouped -> GasHisSDB, 3 seeds; tier2.csv, significance_tier2.csv)
| model | setting | AUC | BA | BA@80% kept | error-detection AUROC |
|---|---|---|---|---|---|
| ResNet50 | none | 0.692 | 0.541 | 0.525 | 0.655 |
| ResNet50 | AdaBN | 0.812 | 0.729 | 0.776 | 0.708 |
| ResNet50 | Macenko | 0.650 | 0.587 | 0.591 | 0.609 |
| Phikon | none / target std | 0.869 / 0.878 | 0.688 / 0.787 | 0.682 / 0.841 | 0.747 / 0.760 |
| Phikon | Macenko / + target std | 0.675 / 0.684 | 0.622 / 0.621 | 0.646 / 0.647 | 0.624 / 0.632 |
| UNI | none / target std | 0.920 / 0.924 | 0.634 / 0.842 | 0.588 / 0.899 | 0.810 / 0.798 |
| UNI | Macenko / + target std | 0.754 / 0.780 | 0.605 / 0.698 | 0.583 / 0.738 | 0.665 / 0.687 |
- Macenko lowers AUC for every model (ResNet50 −0.04, Phikon −0.19, UNI −0.17; all p<0.001, all seeds).
- AdaBN rescues the CNN (+0.19 BA, p<0.001, every seed) but UNI + target std stays ahead (+0.11 BA, p<0.001).
- Message: re-estimating target normalisation statistics without labels helps both CNN (AdaBN) and foundation
  models (target std); image-level stain normalisation hurts here.

## 17. HiESD third-hospital set (23 Nov)
Source: Wang et al., Sci Data 12:1326 (2025), Figshare 10.6084/m9.figshare.28919840 (Figshare licence CC BY 4.0;
paper text says CC BY-NC-ND 4.0). Xi'an Jiaotong University; early gastric cancer (ESD specimens), 40x, 0.2458 um/px.
Selected the 14 smallest slides with >= 100 tiles each of tumour and normal mask (5.42 GB; data/raw/HiESD/meta/
selected_14_slides.csv); all MD5-verified. Tiles: code/scripts/16_hiesd_tiles.py, 224 px at 0.5 um/px aligned to 7x7
mask blocks; TUM = >=50% tub1/tub2/pap; NOR = >=50% normal gland or chronic gastritis (no tumour); >=30% tissue;
normal capped at 800/slide. Index data/index/hiesd.csv: 12,467 tiles (3,117 TUM, 9,350 NOR), 14 slides (real IDs).
No patient IDs in the release (44 patients, 104 slides), so slides of one patient may co-occur; HiESD is test-only.
Launcher: "Run HiESD features.command" (UNI, Phikon, ResNet50 +/- AdaBN).

## 18. HiESD results: HMU (grouped) -> third hospital (hiesd_main.csv, hiesd_significance.csv)
12,467 tiles, 14 slides; 95% CIs by bootstrap over SLIDES (statistic = mean over 3 seeds).
| model | setting | AUC [95% CI] | BA [95% CI] | BA@80% kept |
|---|---|---|---|---|
| UNI | none | 0.866 [0.807–0.911] | 0.636 [0.587–0.688] | 0.579 |
| UNI | target std | 0.871 [0.812–0.915] | 0.777 [0.729–0.819] | 0.822 |
| UNI | class-balanced std | 0.870 | 0.794 | 0.846 |
| Phikon | none / target std | 0.866 / 0.867 | 0.726 / 0.780 | 0.720 / 0.817 |
| ResNet50 | none / AdaBN | 0.790 / 0.784 | 0.569 / 0.709 | 0.546 / 0.752 |
Oracle-threshold BA (uses labels): UNI 0.80, Phikon 0.80, ResNet50 0.72.
Paired slide bootstrap: AUC UNI−ResNet50 +0.076 (p<0.001); UNI−Phikon +0.000 (p=0.99); BA UNI std−none +0.142
(p<0.001); BA@80 +0.243 (p<0.001); Phikon std−none +0.054 (p=0.03); ResNet50 AdaBN−none +0.140 (p<0.001);
UNI std − ResNet50 AdaBN +0.068 (p<0.001). Referral without adaptation again lowers BA (UNI 0.636 -> 0.579).
=> Replicates on a third hospital with real slide IDs and a different stage (early cancer).

## 19. Tool validation against pathologist masks (hiesd_tool_validation.csv, hiesd_fewlabel_validation.csv)
Per-slide true vs predicted tumour share of labelled tiles (leave-one-slide-out; UNI; 3 seeds):
| mode | Pearson r | MAE | mean pred (true 0.248) | pred range | median slide BA |
|---|---|---|---|---|---|
| no adaptation | 0.79 | 0.173 | 0.075 | 0.00–0.29 | 0.632 |
| per-slide standardisation | 0.22 | 0.184 | 0.430 | 0.38–0.50 | 0.789 |
| hospital calibration (13 other slides) | 0.59 | 0.168 | 0.412 | 0.17–0.80 | 0.756 |
| few-label (25+25 tiles from other slides) | 0.64 | 0.084 | 0.289 | – | 0.784 |
- Per-slide standardisation collapses the tumour share (confirms the earlier simulation).
- Hospital calibration restores between-slide differences but over-calls tumour by ~16 points here (calibration pool
  ~25% tumour; matches the simulation's over-call with a tumour-poor pool).
- Few-label calibration gives the most accurate tumour share (MAE 0.08) with good per-slide BA -> recommended mode
  for the tumour percentage; hospital calibration for unlabelled use.

## 20. Fair Macenko check (dataset-level stain matrix from 500 pooled GasHisSDB patches), UNI
- AUC: raw 0.920 | Macenko per patch 0.754 | Macenko dataset-level 0.876 (dataset - raw = -0.044, 95% CI -0.046 to -0.042, p<0.001; dataset - per-patch = +0.122)
- BA (none): raw 0.634 | dataset-level 0.705. BA after target std: raw 0.842 | dataset-level 0.795 (raw - dataset = +0.047, p<0.001); BA@80: 0.899 vs 0.851
- Conclusion: per-patch was unfair to Macenko, but even fairly applied it does not beat raw images; the paper's conclusion stands. Limitation sentence removed. Files: tier2.csv, significance_tier2.csv.
