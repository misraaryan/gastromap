"""Leakage checker: does a test image have a near-identical "twin" in the training set?

For every test image we find its most similar training image (cosine similarity of frozen
foundation-model features). Under a leakage-free split, test images come from unseen slides, so
their best match is a different slide. Under a leaky split, many test images have a twin from the
same slide. We calibrate a "same-slide" threshold on the HMU slide-grouped split (99th percentile
of test->train best-match similarity) and report, for any split, the share of test images above it.

Usage (project root):
  python code/scripts/10_leakage_check.py --features data/features/hmu_uni.npz \
      --splits data/index/hmu_grouped_s0.csv data/index/hmu_random_s0.csv \
      --target_features data/features/gashis_uni.npz --target_splits data/index/gashissdb_random_s0.csv
Generic use on your own data: give a feature file (.npz with X and path) and split CSV(s) with
columns path and split (train/val/test); the first split given defines the threshold unless --tau is set.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load(npz, csv):
    z = np.load(npz, allow_pickle=True)
    X = z["X"].astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    pos = {p: i for i, p in enumerate(z["path"])}
    sp = pd.read_csv(csv)
    sp = sp[sp.split.isin(["train", "test"])]
    tr = np.array([pos[p] for p in sp.loc[sp.split == "train", "path"] if p in pos])
    te = np.array([pos[p] for p in sp.loc[sp.split == "test", "path"] if p in pos])
    return X, tr, te


def best_match(X, tr, te, chunk=1024):
    T = X[tr].T
    return np.concatenate([(X[te[i:i + chunk]] @ T).max(1) for i in range(0, len(te), chunk)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--splits", nargs="+", required=True)
    ap.add_argument("--target_features", default=None)
    ap.add_argument("--target_splits", nargs="*", default=[])
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--out", default="results/leakage_check")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    rows, sims = [], {}
    for csv in a.splits:
        X, tr, te = load(a.features, csv)
        sims[Path(csv).stem] = best_match(X, tr, te)
    tau = a.tau if a.tau is not None else float(np.percentile(sims[Path(a.splits[0]).stem], 99))
    for csv in a.target_splits:
        X, tr, te = load(a.target_features, csv)
        sims[Path(csv).stem] = best_match(X, tr, te)
    for name, s in sims.items():
        rows.append(dict(split=name, n_test=len(s), median_best_sim=float(np.median(s)),
                         share_above_tau=float((s > tau).mean()), tau=tau))
    res = pd.DataFrame(rows)
    res.to_csv(out / "leakage_check.csv", index=False)
    np.savez_compressed(out / "best_match_similarities.npz", **sims)
    print(f"same-slide threshold tau = {tau:.4f} (99th pct of {Path(a.splits[0]).stem})")
    print(res.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
