# Exact duplicate images (MD5 of file bytes), 2026-11-23

| Dataset | Files | Unique | Duplicate groups | Files in duplicate groups |
|---|---|---|---|---|
| HMU-GC-HE-30K, TUM+NOR | 7,774 | 7,368 | 302 | 708 |
| HMU-GC-HE-30K, all 8 classes | 31,096 | 29,910 | – | 1,186 redundant copies |
| GasHisSDB 160 px | 33,284 | 33,284 | 0 | 0 |

- In HMU, duplicate copies sit close together in tile numbering (median distance 5 tiles), which supports the slide-ordered numbering assumption.
- Slide-grouped split (seeds 0–2): **0** duplicate groups cross train/val/test.
- Random split: 125–135 duplicate groups cross sets; 61–67 have identical images in both train and test.
- GasHisSDB has no byte-identical files, yet 57–60% of random-split test patches have a near-twin (UNI cosine ≥ τ) in train, which is consistent with rotated/overlapping crops of the same regions.
