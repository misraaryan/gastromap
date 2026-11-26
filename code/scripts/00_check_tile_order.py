"""Check the key assumption behind the HMU slide-grouped split: that tiles with consecutive
numbers come from the same slide.

Two checks:
 1. Quantitative: colour/texture distance between consecutive tiles vs random tile pairs
    in the same class. If consecutive pairs are clearly closer, the order carries slide identity.
 2. Visual: a contact sheet of consecutive tiles per class, to look at by eye.

Usage:
    python scripts/00_check_tile_order.py --hmu_root /data/HMU-GC-HE-30K --out results/tile_order
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gcx.data import index_hmu  # noqa: E402


def descriptor(path):
    im = np.asarray(Image.open(path).convert("RGB").resize((32, 32)), dtype=np.float32) / 255.0
    return np.concatenate([im.mean((0, 1)), im.std((0, 1)),
                           np.percentile(im.reshape(-1, 3), [10, 50, 90], axis=0).ravel()])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hmu_root", required=True)
    ap.add_argument("--out", default="results/tile_order")
    ap.add_argument("--n_pairs", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    df = index_hmu(a.hmu_root, mapping="tum_vs_rest")
    lines = ["class,consecutive_dist,random_dist,ratio"]
    for cls, sub in df.groupby("orig_class"):
        sub = sub.sort_values("tile_idx").reset_index(drop=True)
        n = len(sub)
        if n < 4:
            continue
        k = min(a.n_pairs, n - 1)
        starts = rng.choice(n - 1, k, replace=False)
        cons = [np.linalg.norm(descriptor(sub.path[i]) - descriptor(sub.path[i + 1])) for i in starts]
        rnd = [np.linalg.norm(descriptor(sub.path[i]) - descriptor(sub.path[j]))
               for i, j in rng.choice(n, (k, 2))]
        c, r = float(np.mean(cons)), float(np.mean(rnd))
        lines.append(f"{cls},{c:.4f},{r:.4f},{c / r:.3f}")
        # contact sheet: 4 runs of 8 consecutive tiles
        tiles = []
        for s in rng.choice(max(1, n - 8), 4, replace=False):
            tiles.append([Image.open(sub.path[s + j]).convert("RGB").resize((96, 96)) for j in range(min(8, n))])
        sheet = Image.new("RGB", (96 * len(tiles[0]), 96 * len(tiles)), "white")
        for r_i, row in enumerate(tiles):
            for c_i, t in enumerate(row):
                sheet.paste(t, (c_i * 96, r_i * 96))
        sheet.save(out / f"contact_{cls}.png")
    (out / "tile_order_check.csv").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("\nRatio well below 1 (e.g. < 0.8) in most classes => consecutive tiles share a slide,"
          "\nso contiguous blocks are a valid slide proxy (use 01 --hmu_block and 02 --gap)."
          "\nRatio ~1 => tile order carries no slide identity and the HMU grouped split is not valid.")


if __name__ == "__main__":
    main()
