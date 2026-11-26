# gastromap

Code and results for the paper:

**Label-Free Feature Standardisation Enables Reliable Uncertainty-Based Referral Under Domain Shift in Computational Pathology**
Aryan Misra, Department of Biotechnology, Delhi Technological University (2026).

The study trains gastric cancer patch classifiers (tumour vs normal mucosa) on HMU-GC-HE-30K, using leakage-controlled, slide-grouped splits. It then tests them without retraining on two other hospitals: GasHisSDB, and HiESD (14 whole slides with pathologist masks and real slide IDs). Three models are compared: a fine-tuned ResNet50 and two frozen pathology foundation models (UNI, Phikon) with a bootstrap linear-probe ensemble. Uncertainty-based referral is evaluated at a fixed threshold, with and without label-free standardisation of the target features (AdaBN for the CNN). GastroMap packages the result as a local whole-slide tool.

## Main result

| Setting (UNI, trained on HMU) | GasHisSDB BA | HiESD BA | BA with 20% referred (GasHisSDB / HiESD) |
|---|---|---|---|
| No adaptation | 0.634 | 0.636 | 0.588 / 0.579 |
| Target feature standardisation (no labels) | 0.842 | 0.777 | 0.899 / 0.822 |
| Oracle threshold (uses target labels) | 0.844 | 0.798 | – |

Foundation models keep their AUC at the new hospitals (UNI 0.920 and 0.866, vs ResNet50 0.692 and 0.790), but at threshold 0.5 their decisions are poor, and referring the most uncertain patches lowers balanced accuracy further. Label-free standardisation recovers almost all of the oracle-threshold accuracy, and referral then works. AdaBN does the same for ResNet50 (+0.19 and +0.14 BA). Macenko stain normalisation lowers AUC for every model. Random patch splitting inflates in-domain AUC by 0.025 to 0.041 compared with the slide-grouped split of the same images.

Tool validation on HiESD (leave-one-slide-out): few-label calibration gives the most accurate tumour share (MAE 8 points). Hospital calibration restores between-slide differences but over-calls tumour, and per-slide standardisation flattens the estimates. Full tables are in `results/`, and every number is traced in `docs/Analysis_log.md`.

## Repository layout

```
code/src/gcx/             library: datasets, models, stain normalisation, uncertainty
code/scripts/             numbered pipeline (run from the repository root)
code/scripts/strengthen/  robustness analyses, baselines, significance tests, HiESD, tool validation
code/tool/                GastroMap (Gradio app) and its trained probe bundle
data/index/               exact image lists and train/validation/test splits (seeds 0-2); hiesd.csv lists the 12,467 HiESD tiles
data/stain_reference/     Macenko reference mosaic (16 HMU training patches) and checks
data/calibration/         example hospital calibration (3 TCGA-HF slides)
results/                  all result tables; results/figures/ holds the paper figures
docs/                     data and split notes, full analysis log
```

## Reproducing the analysis

Python 3.10 or later. Install dependencies:

```
pip install -r requirements.txt
```

Download the public inputs into `data/raw/` as described in `data/README.md`. UNI is gated (request access at huggingface.co/MahmoodLab/UNI, then run `huggingface-cli login`); Phikon downloads automatically. Then run from the repository root (full commands in `code/README.md`):

| Step | Script | Output |
|---|---|---|
| 1. Check that HMU tile order follows slides | `code/scripts/00_check_tile_order.py` | `results/tile_order/` |
| 2. Index datasets and make splits | `01_build_index.py`, `02_make_splits.py` | `data/index/*.csv` |
| 3. ResNet50: train, MC-dropout predictions, AdaBN | `03_train_cnn.py`, `04_predict_cnn.py`, `15_adabn_cnn.py` | `runs/cnn_*` |
| 4. Foundation models: frozen features and probe ensemble | `05_extract_features.py`, `06_linear_probe.py` | `data/features/`, `runs/probe_*` |
| 5. Metrics | `07_evaluate.py` | `results/{cnn,phikon,uni}/` |
| 6. Leakage audit (exact and near duplicates) | `10_leakage_check.py` | `results/leakage_check/` |
| 7. HiESD tiles from pathologist masks | `16_hiesd_tiles.py`, `16b_hiesd_collect.py` | `data/index/hiesd.csv` |
| 8. Prevalence, sample size, oracle threshold, CORAL | `strengthen/s1`–`s3` | `results/strengthen/` |
| 9. Significance tests | `strengthen/s5_significance.py`, `s13_tier2_significance.py` | `significance_*.csv` |
| 10. CNN uncertainty, Macenko and AdaBN baselines | `strengthen/s6`, `s12_tier2.py` | `cnn_uncertainty.csv`, `tier2.csv` |
| 11. Tool calibration modes | `strengthen/s7`–`s11` | `fraction_bias.csv`, `site_calibration.csv`, `fewlabel_calibration.csv` |
| 12. HiESD evaluation and tool validation | `strengthen/s14`–`s16` | `hiesd_*.csv` |
| 13. Figures | `08`, `09`, `11`, `12`, `13`, `17` | `paper/figures/` (copies used in the paper: `results/figures/`) |

Scripts that check the rebuilt probes against the saved predictions are in `strengthen/s0_cache_and_check.py`.

## GastroMap tool

```
python code/tool/app.py
```

On macOS you can instead double-click `Install GastroMap.command` once, then `GastroMap.command`. Drag in a whole-slide image (.svs) or an image, choose the review budget, and click **Analyse**.

- **Few-label calibration** (recommended when a tumour percentage is needed): a pathologist labels about 25 tumour and 25 normal tiles, and the decision threshold is reset.
- **Hospital calibration**: features are standardised with statistics from 3 or more slides of the same hospital, without labels.
- **Per-slide standardisation** (fallback): no tumour percentage is reported.

GastroMap is a research tool and is not approved for clinical use.

## Data sources

- HMU-GC-HE-30K: Lou S. et al., Sci. Data 12:138 (2025), doi:10.1038/s41597-025-04489-9
- GasHisSDB: Hu W. et al., Comput. Biol. Med. 142:105207 (2022), https://gitee.com/neuhwm/GasHisSDB
- HiESD: Wang C. et al., Sci. Data 12:1326 (2025), Figshare doi:10.6084/m9.figshare.28919840
- TCGA-STAD diagnostic slides TCGA-HF-A5NB, -7132, -7133, -7136: NCI Genomic Data Commons
- UNI (MahmoodLab/UNI) and Phikon (owkin/phikon) on Hugging Face

## License

Code: MIT (see `LICENSE`). Result tables and figures: CC BY 4.0. Dataset and foundation-model licences are set by their providers.
