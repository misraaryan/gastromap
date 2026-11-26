"""HiESD (third hospital, real slide IDs): cut labelled tiles from the pathologist masks.

Masks are 64x downsampled from 40x (0.25 um/px) -> 1 mask px = 16 um. A 224 px tile at 0.5 um/px covers
112 um = 7 mask px = 448 level-0 px, so tiles are aligned exactly to 7x7 blocks of the mask.
Label: TUM if >= 50% of the block is adenocarcinoma (tub1/tub2/pap); NOR if >= 50% is normal gland or chronic
gastritis; other blocks are skipped. Each tile must also be >= 30% tissue. Normal tiles are capped per slide.
Tiles are written as PNGs so the existing feature/prediction scripts can read them unchanged.

Usage: python code/scripts/16_hiesd_tiles.py --slide slide_57      (one slide per call; resumable)
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
H = ROOT / "data/raw/HiESD"
TUM = [(139, 0, 0), (255, 0, 255), (128, 0, 128)]
NOR = [(138, 43, 226), (0, 0, 255)]
BLOCK, L0 = 7, 448


def code(a):
    return a[..., 0].astype(np.int64) * 65536 + a[..., 1].astype(np.int64) * 256 + a[..., 2]


def tissue_fraction(rgb):
    a = rgb.astype(np.float32) / 255.0
    mx, mn = a.max(-1), a.min(-1)
    return float((((mx - mn) / (mx + 1e-6) > 0.07) & (mx < 0.92)).mean())


def main():
    import openslide
    ap = argparse.ArgumentParser()
    ap.add_argument("--slide", required=True)
    ap.add_argument("--cap_normal", type=int, default=800)
    ap.add_argument("--min_frac", type=float, default=0.5)
    a = ap.parse_args()
    sel = pd.read_csv(H / "meta/selected_14_slides.csv").set_index("slide")
    svs = sel.loc[a.slide, "svs"]
    out_csv = H / "tiles" / f"{a.slide}.csv"
    if out_csv.exists():
        print("already done", a.slide); return
    Image.MAX_IMAGE_PIXELS = None
    m = np.asarray(Image.open(H / f"meta/annotation_downsample64/ESD_40X_annotation_downsample64/{svs}.png").convert("RGB"))
    k = code(m)
    tum = np.isin(k, [code(np.array(c)) for c in TUM]); nor = np.isin(k, [code(np.array(c)) for c in NOR])
    S = openslide.OpenSlide(str(H / "slides" / f"{svs}.svs"))
    mpp = float(S.properties.get("openslide.mpp-x", 0.25))
    gh, gw = m.shape[0] // BLOCK, m.shape[1] // BLOCK
    t = tum[:gh * BLOCK, :gw * BLOCK].reshape(gh, BLOCK, gw, BLOCK).mean((1, 3))
    n = nor[:gh * BLOCK, :gw * BLOCK].reshape(gh, BLOCK, gw, BLOCK).mean((1, 3))
    cand = [(i, j, "TUM", t[i, j], n[i, j]) for i, j in zip(*np.where(t >= a.min_frac))]
    nor_ij = list(zip(*np.where((n >= a.min_frac) & (t < 0.01))))
    rng = np.random.default_rng(0)
    if len(nor_ij) > a.cap_normal * 2:          # read at most 2x the cap, keep the first cap that pass the tissue check
        nor_ij = [nor_ij[x] for x in rng.choice(len(nor_ij), a.cap_normal * 2, replace=False)]
    cand += [(i, j, "NOR", t[i, j], n[i, j]) for i, j in nor_ij]
    od = H / "tiles" / a.slide; od.mkdir(parents=True, exist_ok=True)
    size0 = int(round(L0 * 0.25 / mpp))          # level-0 px covering 112 um
    rows, n_nor = [], 0
    for i, j, lab, tf, nf in cand:
        if lab == "NOR" and n_nor >= a.cap_normal:
            continue
        x0, y0 = int(j * L0 * 0.25 / mpp), int(i * L0 * 0.25 / mpp)
        img = np.asarray(S.read_region((x0, y0), 0, (size0, size0)).convert("RGB").resize((224, 224), Image.BILINEAR))
        if tissue_fraction(img) < 0.3:
            continue
        p = od / f"{lab}_{x0}_{y0}.png"; Image.fromarray(img).save(p)
        rows.append(dict(path=str(p.relative_to(ROOT)), dataset="hiesd", orig_class=lab, label=int(lab == "TUM"),
                         group=a.slide, slide=a.slide, svs=svs, x0=x0, y0=y0, tum_frac=round(tf, 3), nor_frac=round(nf, 3)))
        n_nor += lab == "NOR"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    d = pd.DataFrame(rows)
    print(f"{a.slide}: mpp {mpp}, {len(d)} tiles ({(d.label == 1).sum()} TUM, {(d.label == 0).sum()} NOR)")


if __name__ == "__main__":
    main()
