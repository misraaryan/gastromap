# Data preparation and splits (22 Nov 2026)

## Datasets in data/raw/ (counts verified against published numbers)
- GasHisSDB: 245,196 patches total. Used: 160 px subset, Abnormal 13,124 / Normal 20,160 (33,284).
- HMU-GC-HE-30K: 31,096 patches, 3,887 per class x 8. Used for binary task: TUM (tumour) vs NOR (normal), 7,774.
  The zip's `all_image/` wrapper folder was removed; HMU-GC-Clinical.csv sits next to the images.

## Is HMU tile order a valid slide proxy? (results/tile_order/)
Colour/texture distance between consecutive tiles divided by distance between random tiles of the same class:
ADI 0.81, DEB 0.77, LYM 0.62, MUC 0.79, MUS 0.79, NOR 0.74, STR 0.68, TUM 0.61 (1.0 = no relation).
By lag (ratio to random): TUM lag1 0.61, lag5 0.72, lag10 0.85, lag13 0.92, then flat ~0.93.
NOR lag1 0.68, lag5 0.76, lag10 0.88, lag13 0.92, then flat ~0.95.
=> Consecutive tiles share a slide; a slide spans roughly 10-15 consecutive tiles.

## Split design (for the Methods section)
- HMU grouped: tiles cut into contiguous blocks of 60 (65 blocks per class); blocks assigned to
  train/val/test (70/15/15), stratified by class. Wherever a block borders (in tile order) a block of a
  different split, its first 15 tiles are excluded ("gap"), ~870-945 tiles per seed.
  Verified: closest train vs val/test tile is 16 numbers apart; 0 of 130 blocks shared.
- HMU random: stratified random patch split drawn from EXACTLY the same images as the grouped split
  of the same seed (only the splitting procedure differs) -> leakage gap is a fair comparison.
- GasHisSDB random: stratified random patch split (no slide IDs exist; state as a limitation).
- Seeds 0, 1, 2 for everything.

## Limitations to report
- HMU numbering is per class, so a slide's TUM tiles and NOR tiles cannot be linked: a slide's tumour
  tiles may be in train while its normal tiles are in test. Blocks are a proxy, not true slide IDs.
- GasHisSDB has no IDs at all; its within-dataset results cannot be made leakage-free.
- Patch size/magnification differ (GasHisSDB 160 px vs HMU 224 px at ~0.5 um/px); all resized to 224.
