"""Extract frozen foundation-model embeddings for every patch of an index.

Usage (Colab GPU):
    python scripts/05_extract_features.py --index data/index/hmu_grouped_s0.csv --encoder phikon \
        --out features/hmu_phikon.npz
Encoders: resnet50_imagenet | phikon | phikon_v2 | uni (gated: accept terms + huggingface-cli login)
Option: --stain_ref <png> to Macenko-normalise first (saved under a different file name).
The .npz holds X (N, D), y, and the row order matches the CSV (column `row` = CSV row).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gcx.data import PatchDataset  # noqa: E402
from gcx.models import FrozenEncoder, get_device  # noqa: E402
from gcx.stain import MacenkoNormalizer  # noqa: E402
from gcx.transforms import make_transforms  # noqa: E402


def main():
    import torch
    from torch.utils.data import DataLoader

    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--encoder", default="phikon")
    ap.add_argument("--stain_ref", default=None)
    ap.add_argument("--stain_mode", default="patch", choices=["patch", "dataset"],
                    help="patch: stain matrix per patch; dataset: one matrix from --stain_source_n pooled patches")
    ap.add_argument("--stain_source_n", type=int, default=500)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max_n", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    dev = get_device()
    print("device:", dev)
    enc = FrozenEncoder(a.encoder, dev)
    df = pd.read_csv(a.index)
    if a.max_n:
        df = df.sample(min(a.max_n, len(df)), random_state=0)
    df = df.reset_index().rename(columns={"index": "row"})
    stain = None
    if a.stain_ref:
        from PIL import Image
        stain = MacenkoNormalizer().fit(np.asarray(Image.open(a.stain_ref).convert("RGB")))
        if a.stain_mode == "dataset":
            src = df.sample(min(a.stain_source_n, len(df)), random_state=0).path
            stain.fit_source([np.asarray(Image.open(p).convert("RGB")) for p in src])
            print(f"dataset-level Macenko: source stain matrix from {len(src)} pooled patches")
    ds = PatchDataset(df, make_transforms(False, mean=enc.mean, std=enc.std), stain)
    dl = DataLoader(ds, a.bs, shuffle=False, num_workers=a.workers)
    feats, ids = [], []
    for x, _, i in dl:
        feats.append(enc(x).float().cpu().numpy()); ids.append(i.numpy())
    order = np.concatenate(ids)
    X = np.concatenate(feats)[np.argsort(order)]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, X=X, y=df.label.to_numpy(), row=df.row.to_numpy(),
                        split=df.get("split", pd.Series(["all"] * len(df))).to_numpy().astype(str),
                        group=df.group.astype(object).where(df.group.notna(), "nan").astype(str).to_numpy(), path=df.path.to_numpy())
    print("saved", a.out, X.shape)


if __name__ == "__main__":
    main()
