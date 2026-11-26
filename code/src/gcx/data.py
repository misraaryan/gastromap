"""Dataset indexing, label mapping and train/val/test splitting.

Two public gastric H&E patch datasets are supported:

* GasHisSDB  (Hu et al., Comput Biol Med 2022) - folders like <root>/160/Abnormal, <root>/160/Normal
* HMU-GC-HE-30K (Lou et al., Sci Data 2025) - 8 class folders: ADI DEB LYM MUC MUS NOR STR TUM

Every image becomes one row of an index table with these columns:
    path, dataset, orig_class, label, tile_idx, group

label: 1 = tumour, 0 = normal, -1 = excluded from the binary task.
group: the unit that must never be split across train/test (a slide, or a proxy for it).
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

IMG_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

HMU_CLASSES = ["ADI", "DEB", "LYM", "MUC", "MUS", "NOR", "STR", "TUM"]


# --------------------------------------------------------------------------- indexing
def _list_images(root: Path) -> list[Path]:
    out = []
    for dirpath, _, files in os.walk(root):
        for f in files:
            if Path(f).suffix.lower() in IMG_EXT:
                out.append(Path(dirpath) / f)
    return sorted(out)


def _trailing_int(name: str) -> int:
    """Last run of digits in a file stem, or -1. Used as the tile order number."""
    nums = re.findall(r"\d+", Path(name).stem)
    return int(nums[-1]) if nums else -1


def index_gashissdb(root: str | Path, subset: str | None = "160") -> pd.DataFrame:
    """Index GasHisSDB. `subset` picks the 160/120/80 px sub-database folder (None = scan all)."""
    root = Path(root)
    if subset is not None:
        cands = [p for p in root.rglob("*") if p.is_dir() and p.name.lower().startswith(str(subset))]
        if not cands:
            raise FileNotFoundError(f"No sub-database folder starting with '{subset}' under {root}")
        root = cands[0]
    rows = []
    for p in _list_images(root):
        cls = p.parent.name.lower()
        if "abnormal" in cls:
            label = 1
        elif "normal" in cls:
            label = 0
        else:
            continue
        rows.append(dict(path=str(p), dataset="gashissdb", orig_class=p.parent.name,
                         label=label, tile_idx=_trailing_int(p.name), group=np.nan))
    if not rows:
        raise FileNotFoundError(f"No Normal/Abnormal images found under {root}")
    return pd.DataFrame(rows)


def index_hmu(root: str | Path, mapping: str = "tum_vs_nor") -> pd.DataFrame:
    """Index HMU-GC-HE-30K.

    mapping:
        'tum_vs_nor'  -> TUM=1, NOR=0, other classes excluded (matches GasHisSDB best; primary)
        'tum_vs_rest' -> TUM=1, every other class 0 (sensitivity analysis)
    """
    root = Path(root)
    rows = []
    for p in _list_images(root):
        cls = p.parent.name.upper()
        if cls not in HMU_CLASSES:
            continue
        if cls == "TUM":
            label = 1
        elif mapping == "tum_vs_nor":
            label = 0 if cls == "NOR" else -1
        elif mapping == "tum_vs_rest":
            label = 0
        else:
            raise ValueError(mapping)
        rows.append(dict(path=str(p), dataset="hmu", orig_class=cls, label=label,
                         tile_idx=_trailing_int(p.name), group=np.nan))
    if not rows:
        raise FileNotFoundError(f"No HMU class folders ({HMU_CLASSES}) found under {root}")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- groups (slide proxies)
def assign_contiguous_groups(df: pd.DataFrame, block: int = 13) -> pd.DataFrame:
    """Slide proxy for HMU: tiles are numbered in slide order inside each class folder,
    so runs of consecutive numbers come from the same slide. We cut each class's sorted
    tile list into blocks of `block` tiles (~3887 tiles / 300 slides per class ~= 13).

    VERIFY THIS ASSUMPTION FIRST with scripts/00_check_tile_order.py.
    """
    df = df.copy()
    groups = np.empty(len(df), dtype=object)
    for cls, sub in df.groupby("orig_class"):
        order = sub.sort_values("tile_idx").index
        for rank, ix in enumerate(order):
            groups[df.index.get_loc(ix)] = f"{cls}_b{rank // block:04d}"
    df["group"] = groups
    return df


def assign_pseudo_groups(df: pd.DataFrame, n_groups: int = 600, seed: int = 0) -> pd.DataFrame:
    """Approximate source-image groups for GasHisSDB (no IDs are given).

    Patches cut from the same 2048x2048 image share stain and texture, so we cluster a
    cheap colour/texture descriptor into ~600 clusters (GasHisSDB has 600 source images).
    This is a PROXY: report it as such. Used only for the optional leakage analysis.
    """
    from PIL import Image
    from sklearn.cluster import MiniBatchKMeans

    feats = []
    for p in df["path"]:
        im = np.asarray(Image.open(p).convert("RGB").resize((32, 32)), dtype=np.float32) / 255.0
        hist = [np.histogram(im[..., c], bins=16, range=(0, 1))[0] for c in range(3)]
        gray = im.mean(-1)
        grad = np.abs(np.diff(gray, axis=0)).mean(), np.abs(np.diff(gray, axis=1)).mean()
        feats.append(np.concatenate([np.concatenate(hist) / gray.size, im.mean((0, 1)),
                                     im.std((0, 1)), grad]))
    X = np.asarray(feats)
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    k = min(n_groups, len(df))
    km = MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=4096, n_init=3).fit(X)
    df = df.copy()
    df["group"] = [f"pg{c:04d}" for c in km.labels_]
    return df


# --------------------------------------------------------------------------- splits
def random_split(df: pd.DataFrame, fracs=(0.7, 0.15, 0.15), seed: int = 0) -> pd.Series:
    """Stratified random split at patch level (the leaky protocol most papers use)."""
    rng = np.random.default_rng(seed)
    split = pd.Series("train", index=df.index)
    for _, sub in df.groupby("label"):
        idx = rng.permutation(sub.index.to_numpy())
        n_tr, n_va = int(fracs[0] * len(idx)), int(fracs[1] * len(idx))
        split.loc[idx[n_tr:n_tr + n_va]] = "val"
        split.loc[idx[n_tr + n_va:]] = "test"
    return split


def grouped_split(df: pd.DataFrame, fracs=(0.7, 0.15, 0.15), seed: int = 0) -> pd.Series:
    """Group-aware split: every group lands in exactly one of train/val/test.
    Groups are shuffled and assigned per label so class balance is kept."""
    if df["group"].isna().any():
        raise ValueError("grouped_split needs a 'group' for every row")
    rng = np.random.default_rng(seed)
    split = pd.Series("train", index=df.index)
    for _, sub in df.groupby("label"):
        groups = rng.permutation(sub["group"].unique())
        sizes = sub.groupby("group").size().loc[groups].to_numpy()
        cum = np.cumsum(sizes) / sizes.sum()
        val_g = groups[(cum > fracs[0]) & (cum <= fracs[0] + fracs[1])]
        test_g = groups[cum > fracs[0] + fracs[1]]
        split.loc[sub.index[sub["group"].isin(val_g)]] = "val"
        split.loc[sub.index[sub["group"].isin(test_g)]] = "test"
    return split


def apply_boundary_gap(df: pd.DataFrame, split: pd.Series, gap: int = 15) -> pd.Series:
    """HMU only. Contiguous blocks are a slide PROXY, so a slide can straddle the edge between two
    neighbouring blocks. Within each class, in tile order: if a block went to a DIFFERENT split than
    the block just before it, its first `gap` tiles are marked 'gap' and excluded from every split.
    HMU tile similarity fades by ~13-15 tiles (results/tile_order), so gap=15 removes shared slides."""
    original = split.copy()
    out = split.copy()
    for _, sub in df.groupby("orig_class"):
        sub = sub.sort_values("tile_idx")
        blocks = list(dict.fromkeys(sub["group"]))          # block ids in tile order
        for prev_b, b in zip(blocks, blocks[1:]):
            rows_prev = sub.index[sub["group"] == prev_b]
            rows_b = sub.index[sub["group"] == b]
            if original[rows_prev[0]] != original[rows_b[0]]:
                out.loc[rows_b[:gap]] = "gap"
    return out


def check_no_group_overlap(df: pd.DataFrame, split_col: str = "split") -> dict:
    """Return how many groups appear in more than one split (should be 0 for grouped)."""
    if df["group"].isna().all():
        return {"groups_shared": None}
    s = df.groupby("group")[split_col].nunique()
    return {"groups_total": int(len(s)), "groups_shared": int((s > 1).sum())}


# --------------------------------------------------------------------------- torch dataset
class PatchDataset:
    """Minimal torch-style dataset. Lazily imports torch so indexing works without it."""

    def __init__(self, df: pd.DataFrame, transform=None, stain_normalizer=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.stain = stain_normalizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        from PIL import Image
        row = self.df.iloc[i]
        img = Image.open(row["path"]).convert("RGB")
        if self.stain is not None:
            img = Image.fromarray(self.stain.transform(np.asarray(img)))
        if self.transform is not None:
            img = self.transform(img)
        return img, int(row["label"]), i
