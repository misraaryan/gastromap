"""Image transforms. All patches are resized to 224 px so both datasets share one input size
(GasHisSDB patches are 160/120/80 px; HMU-GC-HE-30K patches are 224 px)."""
from __future__ import annotations

from .models import IMAGENET_MEAN, IMAGENET_STD
from .stain import HEDJitter


def make_transforms(train: bool, aug: str = "basic", size: int = 224,
                    mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """aug: 'none' | 'basic' (flips/rotations) | 'hed' (basic + HED stain jitter)
             | 'color' (basic + RGB colour jitter)."""
    from torchvision import transforms as T

    ops = [T.Resize((size, size))]
    if train and aug != "none":
        ops += [T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
                T.RandomApply([T.RandomRotation((90, 90))], p=0.5)]
        if aug == "hed":
            ops.append(HEDJitter(sigma=0.05))
        elif aug == "color":
            ops.append(T.ColorJitter(0.2, 0.2, 0.2, 0.05))
    ops += [T.ToTensor(), T.Normalize(mean, std)]
    return T.Compose(ops)
