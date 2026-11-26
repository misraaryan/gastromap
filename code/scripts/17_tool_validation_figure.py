"""Fig: GastroMap tumour share vs pathologist masks on 14 HiESD slides (leave-one-slide-out), three modes."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
                     "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "legend.frameon": False,
                     "savefig.dpi": 300, "savefig.bbox": "tight"})
t = pd.read_csv("results/strengthen/hiesd_tool_validation.csv"); f = pd.read_csv("results/strengthen/hiesd_fewlabel_validation.csv")
series = [("per-slide standardisation", t[t["mode"] == "per-slide standardisation"], "#b8b6b0", "s"),
          ("hospital calibration (13 other slides)", t[t["mode"].str.startswith("hospital")], "#eb6834", "^"),
          ("few-label calibration (50 tiles)", f, "#1baf7a", "o")]
fig, ax = plt.subplots(figsize=(3.4, 2.6))
ax.plot([0, 0.9], [0, 0.9], color=INK2, lw=0.7, ls=(0, (3, 3))); ax.text(0.44, 0.38, "perfect agreement", fontsize=6.3, color=INK2, ha="left")
for lab, d, col, mk in series:
    mae = np.abs(d.pred - d.true).mean(); r = np.corrcoef(d.true, d.pred)[0, 1]
    ax.scatter(d.true, d.pred, s=16, color=col, marker=mk, edgecolor="white", lw=0.4, label=f"{lab}: MAE {mae:.2f}, r {r:.2f}", zorder=3)
ax.set_xlim(0, 0.6); ax.set_ylim(0, 0.9)
ax.set_xlabel("True tumour share of the slide (pathologist mask)"); ax.set_ylabel("Tumour share estimated by GastroMap")
ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.22), fontsize=6.3)
fig.savefig("paper/figures/fig8_tool_validation.pdf"); fig.savefig("paper/figures/fig8_tool_validation.png"); print("ok")
