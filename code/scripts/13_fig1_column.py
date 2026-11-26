"""Fig 1, single-column study design (3 hospitals) for the 6-page IEEE paper."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
INK, INK2, FC, EC, FC2 = "#0b0b0b", "#52514e", "#f4f3f0", "#52514e", "#e8f1fb"
plt.rcParams.update({"font.family": "DejaVu Sans", "savefig.dpi": 300, "savefig.bbox": "tight"})
fig, ax = plt.subplots(figsize=(3.5, 2.95)); ax.set_xlim(0, 100); ax.set_ylim(-1.5, 86.5); ax.axis("off")
def box(x, y, w, h, title, body, fs=5.3, fc=FC):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2", fc=fc, ec=EC, lw=0.7))
    ax.text(x + w / 2, y + h - 2.0, title, ha="center", va="top", fontsize=fs + 0.6, fontweight="bold", color=INK)
    ax.text(x + w / 2, y + h - 6.4, body, ha="center", va="top", fontsize=fs, color=INK2, linespacing=1.25)
def arrow(x0, y0, x1, y1):
    ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.7, mutation_scale=7))
box(0.5, 66, 31.5, 19, "HMU-GC-HE-30K", "Harbin, 300 WSIs\nTUM vs NOR\n7,774 tiles")
box(34.2, 66, 31.5, 19, "GasHisSDB", "Shanghai, 600 images\nabnormal vs normal\n33,284 tiles")
box(68, 66, 31.5, 19, "HiESD (test only)", "Xi'an, 14 WSIs\nearly cancer / normal\n12,467 tiles, slide IDs", fc=FC2)
box(0.5, 45, 31.5, 16, "Two splits", "random vs\nslide-grouped\n(same images)")
box(34.2, 45, 31.5, 16, "Random split", "no slide IDs;\nboth directions")
box(68, 45, 31.5, 16, "External test", "slide-level CIs;\nmask-based check\nof the tool", fc=FC2)
box(0.5, 23, 99, 17, "Models (3 seeds) + label-free adaptation",
    "ResNet50 (fine-tuned; AdaBN)  |  Phikon, UNI (frozen; target standardisation)\nbaselines: CORAL, Macenko stain normalisation", fs=5.0)
box(0.5, 0, 31.5, 18, "C1 Leakage", "random vs grouped\nin-domain AUC", fs=5.0)
box(34.2, 0, 31.5, 18, "C2 Transfer", "AUC vs decisions\nat a new hospital", fs=5.0)
box(68, 0, 31.5, 18, "C3 Referral + tool", "uncertainty referral;\nGastroMap", fs=5.0)
for x in (16.2, 50, 83.7):
    arrow(x, 65.4, x, 61.6); arrow(x, 44.4, x, 40.6); arrow(x, 22.4, x, 18.6)
fig.savefig("paper/figures/fig1_study_design_column.pdf"); fig.savefig("paper/figures/fig1_study_design_column.png")
print("ok")
