"""Figure: best-match similarity of test images to training images (leakage checker output)."""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
                     "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "legend.frameon": False,
                     "savefig.dpi": 300, "savefig.bbox": "tight"})
S = np.load("results/leakage_check/s0/best_match_similarities.npz")
R = pd.concat([pd.read_csv(f"results/leakage_check/s{s}/leakage_check.csv").assign(seed=s) for s in (0, 1, 2)])
tau = pd.read_csv("results/leakage_check/s0/leakage_check.csv").tau.iloc[0]
series = [("hmu_grouped_s0", "HMU, slide-grouped split", "#2a78d6", "-"),
          ("hmu_random_s0", "HMU, random patch split", "#eb6834", "--"),
          ("gashissdb_random_s0", "GasHisSDB, random patch split", "#1baf7a", "-.")]
fig, ax = plt.subplots(figsize=(3.5, 2.5))
for key, lab, col, ls in series:
    s = np.sort(S[key]); y = 1 - np.arange(1, len(s) + 1) / len(s)
    share = R[R.split.str.startswith(key[:-3])].share_above_tau
    ax.plot(s, y, color=col, lw=1.8, ls=ls, label=f"{lab}: {share.mean():.0%} ± {share.std(ddof=1):.0%}")
ax.axvline(tau, color=INK2, lw=0.7, ls=(0, (3, 3)))
ax.text(tau + 0.005, 0.93, "same-slide\nthreshold", fontsize=6.5, color=INK2, va="top")
ax.set_xlabel("Similarity to the closest training image")
ax.set_ylabel("Share of test images at least this similar")
ax.set_xlim(0.4, 1.0); ax.set_ylim(0, 1.02)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), fontsize=6.5, title="share of test images with a training 'twin' (3 seeds)",
          title_fontsize=6.5)
ax.set_title("Random splits put near-twins in train and test", loc="left", fontsize=8.5, color=INK)
fig.savefig("paper/figures/figS2_leakage_check.pdf"); fig.savefig("paper/figures/figS2_leakage_check.png")
print("figure figS2_leakage_check")
