# Code: leakage-aware, cross-dataset gastric cancer histopathology

**Paper question:** How accurate are gastric cancer AI models when tested properly, on unseen slides and on a different hospital's images? Can pathology foundation models and uncertainty estimation make them reliable?

| Contribution | Experiment | Scripts |
|---|---|---|
| 1. Leakage audit | HMU-GC-HE-30K: random patch split vs slide-grouped split | 00, 01, 02, 03/05-06, 07 |
| 2. Cross-dataset generalisation | Train on one dataset, test on the other (no retraining). CNNs vs foundation models, ± stain normalisation/augmentation | 03-07 |
| 3. Uncertainty-aware tool | MC dropout / probe ensembles; does flagging uncertain patches raise accuracy on the unseen dataset? | 04, 06, 07, `tool/app.py` |

All commands below are run from the **project root** (`gastric_cancer/`), so paths look like `data/...`, `runs/...`, `results/...`.

---

## 0. Setup

**Google Colab (recommended, free GPU):** upload the whole `gastric_cancer` folder to Google Drive, then in a Colab notebook (Runtime → GPU):

```python
from google.colab import drive; drive.mount('/content/drive')
%cd /content/drive/MyDrive/gastric_cancer
!pip -q install -r code/requirements.txt
```

Tip: copying the datasets to `/content/` (Colab's local disk) before training is much faster than reading from Drive.

**Foundation models:** Phikon downloads freely. UNI is gated: accept the terms at huggingface.co/MahmoodLab/UNI, then run `huggingface-cli login`.

## 1. Data (see `data/README.md`)

- `data/raw/GasHisSDB/160/{Abnormal,Normal}/`
- `data/raw/HMU-GC-HE-30K/{ADI,DEB,LYM,MUC,MUS,NOR,STR,TUM}/`

**Label mapping (binary task):** tumour = GasHisSDB *Abnormal* / HMU *TUM*; normal = GasHisSDB *Normal* / HMU *NOR*. The other HMU classes are excluded by default (`--hmu_mapping tum_vs_rest` gives the sensitivity analysis). All patches are resized to 224 px.

## 2. Pipeline

```bash
# 00 - CHECK FIRST: do consecutive HMU tile numbers come from the same slide?
python code/scripts/00_check_tile_order.py --hmu_root data/raw/HMU-GC-HE-30K --out results/tile_order
#   ratio < ~0.8 in most classes -> contiguous blocks are a valid slide proxy. Also look at the contact sheets.

# 01 - index both datasets (HMU blocks of 60 tiles; similarity fades by ~15 tiles, see notes/)
python code/scripts/01_build_index.py --gashis_root data/raw/GasHisSDB --gashis_subset 160 \
    --hmu_root data/raw/HMU-GC-HE-30K --hmu_mapping tum_vs_nor --hmu_block 60 --out data/index

# 02 - splits, seeds 0 1 2 (already done; files are in data/index/)
for s in 0 1 2; do
  python code/scripts/02_make_splits.py --index data/index/hmu.csv --mode grouped --gap 15 --seed $s
  python code/scripts/02_make_splits.py --index data/index/hmu.csv --mode random --seed $s \
      --pool_from data/index/hmu_grouped_s$s.csv --out data/index/hmu_random_s$s.csv
  python code/scripts/02_make_splits.py --index data/index/gashissdb.csv --mode random --seed $s
done
#   grouped must print groups_shared = 0; random is drawn from the SAME images as grouped (fair comparison)

# 03 - train a CNN on the source (example: HMU, grouped split, ResNet50, HED stain augmentation)
python code/scripts/03_train_cnn.py --index data/index/hmu_grouped_s0.csv --arch resnet50 --aug hed \
    --epochs 15 --out runs/cnn_hmu-grouped_resnet50_hed_s0

# 04 - predict with MC dropout: in-domain test, then the whole other dataset
python code/scripts/04_predict_cnn.py --ckpt runs/cnn_hmu-grouped_resnet50_hed_s0/best.pt \
    --index data/index/hmu_grouped_s0.csv --split test --name in_domain --out runs/cnn_hmu-grouped_resnet50_hed_s0
python code/scripts/04_predict_cnn.py --ckpt runs/cnn_hmu-grouped_resnet50_hed_s0/best.pt \
    --index data/index/gashissdb.csv --split all --name cross_gashis --out runs/cnn_hmu-grouped_resnet50_hed_s0
#   add --stain_ref <a source patch.png> for the Macenko-normalised variant (use a different --name)

# 05-06 - foundation model: frozen features + linear probe (bootstrap ensemble gives uncertainty)
python code/scripts/05_extract_features.py --index data/index/hmu_grouped_s0.csv --encoder phikon --out features/hmu-grouped_phikon.npz
python code/scripts/05_extract_features.py --index data/index/gashissdb.csv --encoder phikon --out features/gashis_phikon.npz
python code/scripts/06_linear_probe.py --source features/hmu-grouped_phikon.npz --target features/gashis_phikon.npz \
    --name hmu2gashis_phikon --out runs/probe_hmu-grouped_phikon_s0

# 07 - all result tables
python code/scripts/07_evaluate.py --preds "runs/*/pred_*.csv" --out results
```

## 3. Experiment grid (what to run for the paper)

| ID | Source → target | Split of source | Models | Fixes |
|---|---|---|---|---|
| E1 | HMU → HMU test | random vs grouped | ResNet50, EfficientNet-B0, Phikon, UNI | — |
| E2a | HMU (grouped) → GasHisSDB | grouped | same 4 | none / Macenko / HED aug |
| E2b | GasHisSDB → HMU | random (no IDs; state limitation) | same 4 | none / Macenko / HED aug |
| E3 | outputs of E2 | — | MC dropout (CNN), probe ensemble (FM) | flag top 10/20/30% uncertain |

Run 3 seeds (0, 1, 2) and report mean ± SD. Name runs `<model>_<source>-<split>_<arch/encoder>_<fix>_s<seed>`.

## 4. Result tables (`results/`)

- `metrics_all.csv`: AUC (95% CI, resampling whole slides where groups exist), accuracy, balanced accuracy, F1, sensitivity, specificity, ECE, Brier, per run and evaluation set.
- `uncertainty_all.csv`: AURC, error-detection AUROC, accuracy at 100/90/80/70% coverage.
- **Leakage gap** = metric(random) − metric(grouped), same model and seed.
- **Generalisation gap** = metric(in-domain test) − metric(cross-dataset).

## 5. Tool

```bash
python code/tool/app.py --ckpt runs/<best_run>/best.pt --share        # web app (public link on Colab)
python code/tool/app.py --ckpt runs/<best_run>/best.pt --cli data/raw/demo_slides/x.svs   # saves PNG maps
```

The tool outputs a tumour heatmap, an uncertainty map, the tumour fraction and the share of tiles flagged for pathologist review. Research use only.

## Known limitations to report

- Neither dataset gives patient IDs. HMU slide groups are inferred from tile order (check with script 00). GasHisSDB pseudo-groups are approximate clusters.
- GasHisSDB patch size (160 px) and magnification differ from HMU (224 px); this is part of the domain shift.
- A GasHisSDB "Abnormal" patch needs only more than 50% cancer area, while HMU TUM patches are cleaner; label definitions differ slightly.
