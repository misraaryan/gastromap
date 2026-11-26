"""Image figures: example patches from both datasets, and UNI's confident / uncertain / wrong
predictions on the unseen dataset. Run from the project root:  python code/scripts/09_image_figures.py"""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

FIG = Path("paper/figures"); INK, INK2 = "#0b0b0b", "#52514e"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "savefig.dpi": 300, "savefig.bbox": "tight"})
rng = np.random.default_rng(7)

def tile(ax, path, title=None, color=None):
    ax.imshow(Image.open(path).convert("RGB").resize((224, 224)))
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(color is not None); sp.set_color(color or "white"); sp.set_linewidth(1.6)
    if title: ax.set_title(title, fontsize=6.6, color=INK, pad=2)

# ---------------- Fig 5: example patches
h = pd.read_csv("data/index/hmu.csv"); g = pd.read_csv("data/index/gashissdb.csv")
rows = [("HMU\ntumour (TUM)", h[h.label == 1]), ("HMU\nnormal (NOR)", h[h.label == 0]),
        ("GasHisSDB\nabnormal", g[g.label == 1]), ("GasHisSDB\nnormal", g[g.label == 0])]
n = 6
fig, axes = plt.subplots(4, n, figsize=(7.16, 4.9), gridspec_kw=dict(wspace=0.04, hspace=0.12))
for r, (lab, d) in enumerate(rows):
    for c, p in enumerate(d.sample(n, random_state=int(rng.integers(1e6))).path):
        tile(axes[r, c], p)
    axes[r, 0].set_ylabel(lab, fontsize=7, color=INK, rotation=0, ha="right", va="center", labelpad=6)
fig.suptitle("Example patches (randomly sampled). HMU: 224 px, Harbin. GasHisSDB: 160 px, Shanghai (shown enlarged).",
             fontsize=7.5, color=INK2, y=0.93)
fig.savefig(FIG / "fig5_example_patches.pdf"); fig.savefig(FIG / "fig5_example_patches.png"); plt.close(fig)
print("figure fig5_example_patches")

# ---------------- Fig 6: confident-correct / flagged / confident-wrong (UNI, HMU->GasHisSDB, target-std, seed 0)
d = pd.read_csv("runs/probe_uni_hmu-grouped_s0_target_std/pred_hmu-grouped_to_gashis_target_all.csv")
d["pred"] = (d.prob >= 0.5).astype(int); d["correct"] = d.pred == d.label
u = d.entropy / np.log(2)                     # 0 = certain, 1 = maximally unsure
d["u"] = u
thr = d.u.quantile(0.8)                       # top 20% most uncertain are referred
groups = [
    ("Confident\nand correct", d[(~(d.u >= thr)) & d.correct].nsmallest(400, "u"), "#1baf7a"),
    ("Flagged uncertain\n(referred)", d[d.u >= thr].nlargest(400, "u"), "#eda100"),
    ("Confident\nbut wrong", d[(~(d.u >= thr)) & ~d.correct].nsmallest(400, "u"), "#e34948"),
]
fig, axes = plt.subplots(3, 6, figsize=(7.16, 3.9), gridspec_kw=dict(wspace=0.05, hspace=0.32))
for r, (lab, sub, col) in enumerate(groups):
    picks = pd.concat([sub[sub.label == 1].sample(3, random_state=r), sub[sub.label == 0].sample(3, random_state=r)])
    for c, row in enumerate(picks.itertuples()):
        truth = "abnormal" if row.label == 1 else "normal"
        tile(axes[r, c], row.path, f"true {truth}\np(tumour)={row.prob:.2f}", col)
    axes[r, 0].set_ylabel(lab, fontsize=6.8, color=INK, rotation=0, ha="right", va="center", labelpad=6)
share = (d.u >= thr).mean(); wrong_conf = ((~(d.u >= thr)) & ~d.correct).mean()
fig.suptitle(f"UNI trained on HMU, applied to GasHisSDB with target standardisation (seed 0). "
             f"{share:.0%} referred; {wrong_conf:.1%} of all patches confidently wrong.", fontsize=7.2, color=INK2, y=0.98)
fig.savefig(FIG / "fig6_prediction_examples.pdf"); fig.savefig(FIG / "fig6_prediction_examples.png"); plt.close(fig)
print("figure fig6_prediction_examples")
