#!/usr/bin/env bash
# Overnight CNN run on the Mac (Apple GPU): ResNet50, HMU grouped vs random, seed 0.
# For each split: train -> predict on HMU test (in-domain) -> predict on all GasHisSDB (cross-dataset).
# Run from the gastric_cancer folder:   caffeinate -i bash code/run_cnn_overnight.sh
set -euo pipefail
SEED=${SEED:-0}
for SPLIT in grouped random; do
  RUN=runs/cnn_resnet50_hmu-${SPLIT}_hed_s${SEED}
  echo "=== $(date '+%H:%M') training $RUN"
  python code/scripts/03_train_cnn.py --index data/index/hmu_${SPLIT}_s${SEED}.csv --arch resnet50 \
      --aug hed --epochs 12 --patience 4 --bs 32 --seed $SEED --out $RUN
  echo "=== $(date '+%H:%M') predicting in-domain"
  python code/scripts/04_predict_cnn.py --ckpt $RUN/best.pt --index data/index/hmu_${SPLIT}_s${SEED}.csv \
      --split test --name hmu-${SPLIT}_to_gashis_source_test --T 20 --out $RUN
  echo "=== $(date '+%H:%M') predicting cross-dataset (GasHisSDB)"
  python code/scripts/04_predict_cnn.py --ckpt $RUN/best.pt --index data/index/gashissdb.csv \
      --split all --name hmu-${SPLIT}_to_gashis_target_all --T 20 --out $RUN
done
echo "=== $(date '+%H:%M') ALL DONE"
