"""Add a `split` column (train/val/test) to an index CSV.

Usage:
    python scripts/02_make_splits.py --index data/index/hmu.csv --mode grouped --seed 0
    python scripts/02_make_splits.py --index data/index/hmu.csv --mode random  --seed 0

Writes <index stem>_<mode>_s<seed>.csv next to the input and prints a leakage check.
For the cross-dataset experiments the TARGET dataset is used whole (all rows = test),
so only the source dataset needs a split.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gcx.data import random_split, grouped_split, apply_boundary_gap, check_no_group_overlap  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--mode", choices=["random", "grouped"], required=True)
    ap.add_argument("--fracs", type=float, nargs=3, default=[0.7, 0.15, 0.15])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gap", type=int, default=0,
                    help="HMU grouped only: drop this many tiles at block edges that border another split")
    ap.add_argument("--pool_from", default=None,
                    help="use only the images that are NOT 'gap' in this grouped-split CSV "
                         "(so random and grouped splits draw from identical images)")
    ap.add_argument("--out", default=None, help="output CSV path (default: <index>_<mode>_s<seed>.csv)")
    a = ap.parse_args()
    df = pd.read_csv(a.index)
    if a.pool_from:
        keep = pd.read_csv(a.pool_from)
        keep = set(keep.loc[keep.split != "gap", "path"])
        df = df[df.path.isin(keep)].reset_index(drop=True)
        print(f"pool restricted to {len(df)} images from {a.pool_from}")
    fn = random_split if a.mode == "random" else grouped_split
    df["split"] = fn(df, tuple(a.fracs), a.seed)
    if a.gap and a.mode == "grouped":
        df["split"] = apply_boundary_gap(df, df["split"], a.gap)
        print(f"boundary gap {a.gap}: {(df.split == 'gap').sum()} tiles left out")
    out = Path(a.out) if a.out else Path(a.index).with_name(f"{Path(a.index).stem}_{a.mode}_s{a.seed}.csv")
    df.to_csv(out, index=False)
    print(df.groupby(["split", "label"]).size().unstack(fill_value=0))
    print("leakage check:", check_no_group_overlap(df[df.split != "gap"]))
    print("wrote", out)


if __name__ == "__main__":
    main()
