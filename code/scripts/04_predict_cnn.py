"""Predict with a trained CNN on any index (source test split, or a whole target dataset),
with MC dropout. Writes per-patch predictions (prob, entropy, mutual information).

Usage:
    # in-domain test
    python scripts/04_predict_cnn.py --ckpt runs/X/best.pt --index data/index/hmu_grouped_s0.csv \
        --split test --name hmu_test --out runs/X
    # cross-dataset: whole target dataset
    python scripts/04_predict_cnn.py --ckpt runs/X/best.pt --index data/index/gashissdb.csv \
        --split all --name gashis_all --out runs/X
Options: --T 20 (MC passes; 1 = deterministic), --stain_ref <png> (normalise TARGET to source style)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gcx.data import PatchDataset  # noqa: E402
from gcx.models import build_cnn, get_device  # noqa: E402
from gcx.stain import MacenkoNormalizer  # noqa: E402
from gcx.transforms import make_transforms  # noqa: E402
from gcx.uncertainty import predict_mc, predict_det, summarise_samples  # noqa: E402


def main():
    import torch
    from torch.utils.data import DataLoader

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--split", default="test", help="train|val|test|all")
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--T", type=int, default=20)
    ap.add_argument("--stain_ref", default=None)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max_n", type=int, default=0)
    a = ap.parse_args()

    dev = get_device()
    print("device:", dev)
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    model = build_cnn(ck["arch"], False, 2, ck["drop"]).to(dev)
    model.load_state_dict(ck["state_dict"])

    df = pd.read_csv(a.index)
    if a.split != "all":
        df = df[df.split == a.split]
    if a.max_n:
        df = df.sample(min(a.max_n, len(df)), random_state=0)
    df = df.reset_index(drop=True)
    stain = None
    if a.stain_ref:
        from PIL import Image
        stain = MacenkoNormalizer().fit(np.asarray(Image.open(a.stain_ref).convert("RGB")))
    dl = DataLoader(PatchDataset(df, make_transforms(False), stain), a.bs, shuffle=False,
                    num_workers=a.workers)

    if a.T > 1:
        y, ids, P = predict_mc(model, dl, dev, a.T)
        s = summarise_samples(P)
        _, _, p_det = predict_det(model, dl, dev)
    else:
        y, ids, p_det = predict_det(model, dl, dev)
        s = summarise_samples(p_det[None])
    res = df.iloc[ids][["path", "dataset", "orig_class", "label", "group"]].copy()
    res["prob_det"] = p_det
    for k in ("prob", "entropy", "mutual_info", "std"):
        res[k] = s[k]
    Path(a.out).mkdir(parents=True, exist_ok=True)
    f = Path(a.out) / f"pred_{a.name}.csv"
    res.to_csv(f, index=False)
    print("wrote", f, len(res))


if __name__ == "__main__":
    main()
