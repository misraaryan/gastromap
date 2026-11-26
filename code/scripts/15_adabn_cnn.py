"""AdaBN for the ResNet50 baseline (Li et al. 2018): re-estimate every BatchNorm layer's running mean/variance
on unlabelled TARGET images, then predict the whole target set (deterministic pass + MC dropout).
This is the CNN counterpart of target standardisation for the frozen foundation-model features.

Usage (Mac, gcx-env):
  python code/scripts/15_adabn_cnn.py --ckpt runs/cnn_resnet50_hmu-grouped_hed_s0/best.pt \
      --index data/index/gashissdb.csv --name hmu-grouped_to_gashis_target_all \
      --out runs/cnn_resnet50_hmu-grouped_hed_s0_adabn
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gcx.data import PatchDataset  # noqa: E402
from gcx.models import build_cnn, get_device  # noqa: E402
from gcx.transforms import make_transforms  # noqa: E402
from gcx.uncertainty import predict_det, predict_mc, summarise_samples  # noqa: E402


def main():
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_stats", type=int, default=2000, help="unlabelled target images used to re-estimate BN stats")
    ap.add_argument("--T", type=int, default=20)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    dev = get_device(); print("device:", dev)
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    model = build_cnn(ck["arch"], False, 2, ck["drop"]).to(dev)
    model.load_state_dict(ck["state_dict"])
    df = pd.read_csv(a.index).reset_index(drop=True)
    tf = make_transforms(False)

    # 1) re-estimate BatchNorm statistics on a random unlabelled subset of the target set
    stats_df = df.sample(min(a.n_stats, len(df)), random_state=a.seed)
    bns = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    for m in bns:
        m.reset_running_stats(); m.momentum = None          # cumulative average over all batches
    model.train()
    with torch.no_grad():
        for x, _, _ in DataLoader(PatchDataset(stats_df, tf), a.bs, shuffle=False, num_workers=a.workers):
            model(x.to(dev))
    model.eval()
    print(f"AdaBN: re-estimated {len(bns)} BatchNorm layers on {len(stats_df)} target images")

    # 2) predict the whole target set with the adapted model
    dl = DataLoader(PatchDataset(df, tf), a.bs, shuffle=False, num_workers=a.workers)
    y, ids, P = predict_mc(model, dl, dev, a.T)
    s = summarise_samples(P)
    _, _, p_det = predict_det(model, dl, dev)
    res = df.iloc[ids][[c for c in ("path", "dataset", "orig_class", "label", "group") if c in df]].copy()
    res["prob_det"] = p_det
    for k in ("prob", "entropy", "mutual_info", "std"):
        res[k] = s[k]
    Path(a.out).mkdir(parents=True, exist_ok=True)
    f = Path(a.out) / f"pred_{a.name}.csv"
    res.to_csv(f, index=False)
    print("wrote", f, len(res), "| max MC std", float(s["std"].max()))


if __name__ == "__main__":
    main()
