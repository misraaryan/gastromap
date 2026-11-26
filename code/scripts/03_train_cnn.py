"""Train a CNN (ImageNet-pretrained, fine-tuned) on the source dataset's train split.
Model selection by validation AUC. Saves best checkpoint + training log.

Usage (Colab GPU):
    python scripts/03_train_cnn.py --index data/index/hmu_grouped_s0.csv --arch resnet50 \
        --aug hed --epochs 15 --out runs/hmu_grouped_resnet50_hed_s0
Options:
    --stain_ref <png>   Macenko-normalise every patch to this reference patch (from the source set)
    --max_per_split N   cap patches per split (quick smoke tests)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gcx.data import PatchDataset  # noqa: E402
from gcx.metrics import classification_metrics  # noqa: E402
from gcx.models import build_cnn, get_device  # noqa: E402
from gcx.stain import MacenkoNormalizer  # noqa: E402
from gcx.transforms import make_transforms  # noqa: E402
from gcx.uncertainty import predict_det  # noqa: E402


def set_seed(s):
    import random
    import torch
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def main():
    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler

    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--arch", default="resnet50")
    ap.add_argument("--pretrained", type=int, default=1)
    ap.add_argument("--aug", default="basic", choices=["none", "basic", "hed", "color"])
    ap.add_argument("--stain_ref", default=None)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--drop", type=float, default=0.3)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max_per_split", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    set_seed(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dev = get_device()
    print("device:", dev)
    df = pd.read_csv(a.index)
    tr, va = df[df.split == "train"], df[df.split == "val"]
    if a.max_per_split:
        tr = tr.sample(min(a.max_per_split, len(tr)), random_state=a.seed)
        va = va.sample(min(a.max_per_split, len(va)), random_state=a.seed)

    stain = None
    if a.stain_ref:
        from PIL import Image
        stain = MacenkoNormalizer().fit(np.asarray(Image.open(a.stain_ref).convert("RGB")))

    ds_tr = PatchDataset(tr, make_transforms(True, a.aug), stain)
    ds_va = PatchDataset(va, make_transforms(False), stain)
    # class-balanced sampling (GasHisSDB is imbalanced)
    w = 1.0 / tr.label.map(tr.label.value_counts()).to_numpy()
    sampler = WeightedRandomSampler(torch.tensor(w, dtype=torch.double), len(tr), replacement=True)
    dl_tr = DataLoader(ds_tr, a.bs, sampler=sampler, num_workers=a.workers, pin_memory=dev == "cuda")
    dl_va = DataLoader(ds_va, a.bs * 2, shuffle=False, num_workers=a.workers)

    model = build_cnn(a.arch, bool(a.pretrained), 2, a.drop).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    lossf = torch.nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=dev == "cuda")

    best, bad, log = -1.0, 0, []
    for ep in range(1, a.epochs + 1):
        model.train(); t0 = time.time(); tot = 0.0
        for x, y, _ in dl_tr:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=dev == "cuda"):
                loss = lossf(model(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot += loss.item() * len(y)
        sched.step()
        yv, _, pv = predict_det(model, dl_va, dev)
        m = classification_metrics(yv, pv)
        rec = dict(epoch=ep, train_loss=tot / len(ds_tr), val_auc=m["auc"],
                   val_bal_acc=m["bal_acc"], sec=round(time.time() - t0, 1))
        log.append(rec); print(rec, flush=True)
        score = m["auc"] if not np.isnan(m["auc"]) else m["bal_acc"]
        if score > best:
            best, bad = score, 0
            torch.save({"state_dict": model.state_dict(), "arch": a.arch, "drop": a.drop,
                        "args": vars(a)}, out / "best.pt")
        else:
            bad += 1
            if bad >= a.patience:
                print("early stop"); break
    pd.DataFrame(log).to_csv(out / "train_log.csv", index=False)
    (out / "config.json").write_text(json.dumps(vars(a), indent=2))
    print("best val score", best)


if __name__ == "__main__":
    main()
