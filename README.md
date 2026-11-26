# GastroMap: label-free standardisation for uncertainty-based referral in gastric histopathology

Code for: A. Misra, "Label-Free Feature Standardisation Enables Reliable Uncertainty-Based Referral
Under Domain Shift in Computational Pathology."

Models are trained on HMU-GC-HE-30K using leakage-controlled, slide-grouped splits. They are then
tested on two other hospitals, GasHisSDB and HiESD. Frozen pathology foundation models (UNI, Phikon)
keep their AUC at the new hospitals, but at a fixed threshold they make wrong decisions, and
uncertainty-based referral makes this worse. Standardising the target features with their own
statistics (a known label-free method, used here as AdaBN for the CNN) fixes the threshold, and
referral then works. GastroMap packages this as a local whole-slide tool.

## Data

All data are public. Place them under `data/raw/` as described in `data/README.md` (they are not tracked):

| Dataset | Role | Source |
|---|---|---|
| HMU-GC-HE-30K | training, in-domain test | Lou et al., *Sci Data* 12:138 (2025), doi:10.1038/s41597-025-04489-9 |
| GasHisSDB (160 px) | second hospital; reverse-direction training | Hu et al., *Comput Biol Med* 142:105207 (2022), gitee.com/neuhwm/GasHisSDB |
| HiESD (14 of 104 slides) | third hospital, test only | Wang et al., *Sci Data* 12:1326 (2025), Figshare doi:10.6084/m9.figshare.28919840 |
| TCGA-STAD (4 slides: HF-A5NB, HF-7132, HF-7133, HF-7136) | tool demonstration | NCI GDC |

`data/index/` holds the exact image lists and train/validation/test splits (seeds 0–2) used in the
paper. `hiesd.csv` lists the 12,467 HiESD tiles with their slide IDs and coordinates.
`data/stain_reference/` holds the Macenko reference mosaic, and `data/calibration/` holds an example
hospital calibration built from 3 TCGA-HF slides.

Foundation-model weights are not included. UNI is gated (request access at
huggingface.co/MahmoodLab/UNI, then run `huggingface-cli login`). Phikon downloads automatically.

## Pipeline

Run from the repository root. Step-by-step commands are in `code/README.md`.

1. `00`–`02`: tile-order check (the slide proxy for HMU), index, and splits.
2. `03`–`04`, `15`: ResNet50 fine-tuning, MC-dropout prediction, AdaBN.
3. `05`–`06`: frozen UNI/Phikon features and the bootstrap linear-probe ensemble.
4. `07`: metrics. `10`–`11`: leakage audit (exact and near duplicates).
5. `16`, `16b`: HiESD tiling from the pathologist masks.
6. `code/scripts/strengthen/`: prevalence and sample-size limits, oracle threshold, CORAL and Macenko
   baselines, significance tests, HiESD evaluation, and tool validation.
7. `08`, `09`, `11`, `12`, `13`, `17`: figures.

## Results

`results/` holds every table behind the paper. `results/strengthen/summary.md` is the full analysis
log, with each number in the paper and the file it comes from.

## GastroMap tool

```bash
pip install -r requirements.txt
python code/tool/app.py
```

On macOS you can instead double-click `Install GastroMap.command` once, then `GastroMap.command`.
Drag in a whole-slide image (.svs) or an image, choose the review budget, and click **Analyse**.
The trained probe ensemble is in `code/tool/bundle/`.

Calibration modes:
- **Few-label calibration** (recommended when a tumour percentage is needed): a pathologist labels
  about 25 tumour and 25 normal tiles, and the decision threshold is reset. This was the closest to
  the HiESD pathologist masks (MAE 8 points).
- **Hospital calibration**: features are standardised with statistics from 3 or more slides from
  the same hospital, without labels. Use a representative mix of slides.
- **Per-slide standardisation** (fallback): each slide is adapted on its own, and no tumour
  percentage is reported.

GastroMap is a research tool and is not approved for clinical use.

## Requirements

`pip install -r requirements.txt` (Python 3.10+). Whole-slide images also need the OpenSlide system library.

## Licence

The code is released under the MIT licence. Dataset and foundation-model licences are set by their
providers (UNI weights are for non-commercial research use).
