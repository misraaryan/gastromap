"""Make the paper's figures and tables from the saved results.

Inputs : results/{cnn,phikon,uni}/metrics_all.csv, runs/probe_*/pred_*.csv, data/raw/HMU-GC-HE-30K
Outputs: paper/figures/fig*.{pdf,png}, paper/tables/table*.{csv,tex}

Run from the project root:  python code/scripts/08_make_figures.py
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import balanced_accuracy_score

FIG = Path("paper/figures"); TAB = Path("paper/tables")
FIG.mkdir(parents=True, exist_ok=True); TAB.mkdir(parents=True, exist_ok=True)

# Validated categorical slots 1-3 (all-pairs CVD-safe); models are ALSO named on every chart.
COL = {"ResNet50": "#2a78d6", "Phikon": "#eb6834", "UNI": "#1baf7a"}
MODELS = ["ResNet50", "Phikon", "UNI"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight",
})
SINGLE, DOUBLE = 3.5, 7.16   # IEEE column widths (inches)


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf"); fig.savefig(FIG / f"{name}.png")
    plt.close(fig); print("figure", name)


# ----------------------------------------------------------------------------- load metrics
def load(folder, prefix, model):
    d = pd.read_csv(f"{folder}/metrics_all.csv")
    r = d.run.str.replace(prefix, "", regex=False).str.replace("_hed", "", regex=False)
    d["adapt"] = "none"
    for a in ("target_std", "coral"):
        hit = r.str.endswith("_" + a); d.loc[hit, "adapt"] = a; r = r.where(~hit, r.str[:-(len(a) + 1)])
    d["setting"] = r.str.rsplit("_s", n=1).str[0]
    d["seed"] = r.str.rsplit("_s", n=1).str[1].astype(int)
    d["where"] = np.where(d["eval"].str.endswith("source_test"), "in-domain", "cross")
    d["model"] = model
    return d


M = pd.concat([load("results/cnn", "cnn_resnet50_", "ResNet50"),
               load("results/phikon", "probe_phikon_", "Phikon"),
               load("results/uni", "probe_uni_", "UNI")], ignore_index=True)
B = M[M.adapt == "none"]


def ms(x):
    return f"{np.mean(x):.3f} ± {np.std(x, ddof=1):.3f}"


# ============================================================================ FIGURE 1: design
fig, ax = plt.subplots(figsize=(DOUBLE, 2.6)); ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 40)


def box(x, y, w, h, title, body, fc="#f4f3f0"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                fc=fc, ec=INK2, lw=0.6))
    ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="top", fontsize=7.6, weight="bold", color=INK)
    ax.text(x + w / 2, y + h - 6.2, body, ha="center", va="top", fontsize=6.6, color=INK2, linespacing=1.35)


def arrow(x0, y0, x1, y1):
    ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.7))


box(1, 22, 21, 16, "HMU-GC-HE-30K", "Harbin, 300 slides\nTUM vs NOR\n7,774 patches (224 px)")
box(1, 2, 21, 16, "GasHisSDB", "Shanghai, 600 images\nAbnormal vs Normal\n33,284 patches (160 px)")
box(27, 22, 22, 16, "Splits (3 seeds)", "random patch split\nvs slide-grouped split\n(60-tile blocks + 15-tile\ngaps; same image pool)")
box(27, 2, 22, 16, "Split", "random patch split\n(no slide IDs exist)")
box(54, 12, 21, 26, "Models", "ResNet50 (fine-tuned)\n\nPhikon (frozen)\n+ linear probe\n\nUNI (frozen)\n+ linear probe")
box(80, 26, 19, 12, "C1 Leakage", "random vs grouped\nin-domain AUC")
box(80, 13, 19, 11, "C2 Transfer", "train on one dataset,\ntest on the other")
box(80, 0.5, 19, 11, "C3 Reliability", "label-free adaptation\n+ uncertainty referral")
for y in (30, 10):
    arrow(22.6, y, 26.6, y)
arrow(49.6, 30, 53.6, 28); arrow(49.6, 10, 53.6, 18)
for y in (32, 18.5, 6):
    arrow(75.6, 25, 79.6, y)
save(fig, "fig1_study_design")

# ============================================================================ FIGURE 2: leakage
fig, ax = plt.subplots(figsize=(SINGLE, 2.4))
for i, mod in enumerate(MODELS):
    d = B[(B.model == mod) & (B["where"] == "in-domain") & B.setting.isin(["hmu-grouped", "hmu-random"])]
    for s in sorted(d.seed.unique()):
        g = d[(d.seed == s) & (d.setting == "hmu-grouped")].auc.item()
        r = d[(d.seed == s) & (d.setting == "hmu-random")].auc.item()
        xo = i + (s - 1) * 0.08
        ax.plot([xo - 0.15, xo + 0.15], [g, r], color=COL[mod], lw=1.2, alpha=0.9, zorder=2)
        ax.scatter(xo - 0.15, g, s=26, marker="o", facecolor="white", edgecolor=COL[mod], lw=1.4, zorder=3)
        ax.scatter(xo + 0.15, r, s=26, marker="o", color=COL[mod], edgecolor="white", lw=0.8, zorder=3)
    gap = (d[d.setting == "hmu-random"].sort_values("seed").auc.to_numpy()
           - d[d.setting == "hmu-grouped"].sort_values("seed").auc.to_numpy())
    ax.text(i, 0.945, f"+{gap.mean():.3f}", ha="center", fontsize=7, color=INK)
ax.set_xticks(range(3)); ax.set_xticklabels(MODELS)
ax.set_ylabel("In-domain AUC (HMU test)"); ax.set_ylim(0.85, 0.955); ax.set_xlim(-0.5, 2.5)
ax.scatter([], [], s=26, facecolor="white", edgecolor=INK2, lw=1.4, label="slide-grouped split")
ax.scatter([], [], s=26, color=INK2, label="random patch split")
ax.legend(loc="lower right", fontsize=6.8, handletextpad=0.3)
ax.set_title("Random splits inflate AUC (mean gap above; lines = seeds)", loc="left", color=INK)
save(fig, "fig2_leakage")

# ============================================================================ FIGURE 3: transfer
fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.5), gridspec_kw=dict(width_ratios=[3, 2]))
panels = [("hmu-grouped", MODELS, "(a) Trained on HMU (slide-grouped)", "GasHisSDB"),
          ("gashis-random", ["Phikon", "UNI"], "(b) Trained on GasHisSDB (random split)", "HMU")]
for ax, (setting, mods, title, other) in zip(axes, panels):
    for i, mod in enumerate(mods):
        d = B[(B.model == mod) & (B.setting == setting)]
        a, c = d[d["where"] == "in-domain"].auc, d[d["where"] == "cross"].auc
        ax.errorbar([i - 0.12], [a.mean()], yerr=[a.std(ddof=1)], fmt="o", ms=5, mfc="white",
                    mec=COL[mod], mew=1.4, ecolor=COL[mod], elinewidth=1, capsize=2, zorder=3)
        ax.errorbar([i + 0.12], [c.mean()], yerr=[c.std(ddof=1)], fmt="o", ms=5, color=COL[mod],
                    mec="white", mew=0.8, ecolor=COL[mod], elinewidth=1, capsize=2, zorder=3)
        ax.plot([i - 0.12, i + 0.12], [a.mean(), c.mean()], color=COL[mod], lw=1, zorder=2)
        ax.text(i + 0.2, c.mean(), f"{c.mean():.2f}", va="center", fontsize=7, color=INK)
        ax.text(i - 0.2, a.mean(), f"{a.mean():.2f}", va="center", ha="right", fontsize=7, color=INK)
    ax.axhline(0.5, color=INK2, lw=0.6, ls=(0, (3, 3)))
    ax.text(len(mods) - 0.55, 0.51, "chance", fontsize=6.5, color=INK2, va="bottom", ha="right")
    ax.set_xticks(range(len(mods))); ax.set_xticklabels(mods); ax.set_xlim(-0.6, len(mods) - 0.4)
    ax.set_ylim(0.45, 1.04); ax.set_title(title, loc="left", color=INK)
axes[0].set_ylabel("AUC (mean ± SD, 3 seeds)")
axes[0].scatter([], [], s=24, facecolor="white", edgecolor=INK2, lw=1.4, label="same dataset (test split)")
axes[0].scatter([], [], s=24, color=INK2, label="other dataset (all images)")
fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2, fontsize=7,
           handletextpad=0.3, bbox_to_anchor=(0.5, -0.06))
save(fig, "fig3_cross_dataset")

# ============================================================================ FIGURE 4: adaptation + referral
COV = np.linspace(0.5, 1.0, 26)


def curve(model_prefix, adapt):
    suffix = "" if adapt == "none" else f"_{adapt}"
    out = []
    for s in (0, 1, 2):
        f = Path(f"runs/{model_prefix}_hmu-grouped_s{s}{suffix}/pred_hmu-grouped_to_gashis_target_all.csv")
        d = pd.read_csv(f); y = d.label.to_numpy(); p = d.prob.to_numpy(); u = d.entropy.to_numpy()
        order = np.argsort(u, kind="stable")
        row = []
        for c in COV:
            k = order[: int(round(c * len(y)))]
            row.append(balanced_accuracy_score(y[k], (p[k] >= 0.5).astype(int)))
        out.append(row)
    return np.asarray(out)


fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.5), sharey=True)
for ax, (mod, pref) in zip(axes, (("Phikon", "probe_phikon"), ("UNI", "probe_uni"))):
    for adapt, ls, lab in (("none", (0, (3, 2)), "no adaptation"), ("target_std", "-", "target standardisation")):
        C = curve(pref, adapt); mu, sd = C.mean(0), C.std(0, ddof=1)
        ax.plot(COV * 100, mu, color=COL[mod], lw=2 if adapt != "none" else 1.4, ls=ls, label=lab)
        ax.fill_between(COV * 100, mu - sd, mu + sd, color=COL[mod], alpha=0.12 if adapt != "none" else 0.07, lw=0)
        ax.text(99, mu[-1] + 0.015, f"{mu[-1]:.2f}", fontsize=6.8, va="bottom", ha="left", color=INK)
        if adapt != "none":
            k80 = np.argmin(np.abs(COV - 0.8))
            ax.scatter(80, mu[k80], s=18, color=COL[mod], zorder=3, edgecolor="white", lw=0.8)
            ax.annotate(f"{mu[k80]:.2f} with 20% referred", (80, mu[k80]), (79, mu[k80] - 0.075),
                        fontsize=6.6, color=INK, ha="left",
                        arrowprops=dict(arrowstyle="-", color=INK2, lw=0.5, shrinkA=0, shrinkB=3))
    ax.invert_xaxis(); ax.set_xlim(100, 50)
    ax.set_title(f"({'a' if mod == 'Phikon' else 'b'}) {mod}, HMU → GasHisSDB", loc="left", color=INK)
    ax.legend(loc="lower right", fontsize=6.8)
axes[0].set_ylabel("Balanced accuracy on kept patches")
fig.supxlabel("Patches kept by the model (%); the least confident rest is referred to a pathologist", fontsize=8, y=-0.02)
save(fig, "fig4_adaptation_referral")

# ============================================================================ FIGURE S1: tile order
lag_file = Path("results/tile_order/lag_curve.csv")
if not lag_file.exists():
    from PIL import Image
    rng = np.random.default_rng(1); rows = []
    for cls in ("TUM", "NOR"):
        folder = Path("data/raw/HMU-GC-HE-30K") / cls
        files = sorted(folder.iterdir(), key=lambda p: int(re.findall(r"\d+", p.name)[-1]))
        D = []
        for p in files:
            im = np.asarray(Image.open(p).convert("RGB").resize((32, 32)), dtype=np.float32) / 255
            D.append(np.concatenate([im.mean((0, 1)), im.std((0, 1)),
                                     np.percentile(im.reshape(-1, 3), [10, 50, 90], axis=0).ravel()]))
        D = np.asarray(D)
        rnd = np.mean([np.linalg.norm(D[i] - D[j]) for i, j in rng.integers(0, len(D), (3000, 2))])
        for lag in (1, 2, 3, 5, 8, 10, 13, 15, 20, 25, 30, 40, 60):
            rows.append(dict(cls=cls, lag=lag, ratio=np.mean(np.linalg.norm(D[lag:] - D[:-lag], axis=1)) / rnd))
    pd.DataFrame(rows).to_csv(lag_file, index=False)
L = pd.read_csv(lag_file)
fig, ax = plt.subplots(figsize=(SINGLE, 2.2))
for cls, c, mk in (("TUM", "#2a78d6", "o"), ("NOR", "#eb6834", "s")):
    d = L[L.cls == cls]
    ax.plot(d.lag, d.ratio, color=c, lw=1.6, marker=mk, ms=4, mec="white", mew=0.6)
    ax.text(d.lag.iloc[-1] + 1.5, d.ratio.iloc[-1], cls, color=INK, fontsize=7, va="center")
ax.axhline(1.0, color=INK2, lw=0.6, ls=(0, (3, 3))); ax.text(64, 1.008, "unrelated tiles", fontsize=6.5, color=INK2, va="bottom", ha="right")
ax.axvspan(0, 15, color="#f4f3f0", zorder=0); ax.text(16.5, 0.6, "shaded: same slide\n(≈10–15 tiles)", ha="left", fontsize=6.5, color=INK2)
ax.set_xlabel("Distance in tile number"); ax.set_ylabel("Feature distance / random pairs")
ax.set_xlim(0, 66); ax.set_ylim(0.55, 1.05)
ax.set_title("HMU tile numbering follows slide order", loc="left", color=INK)
save(fig, "figS1_tile_order")

# ============================================================================ TABLES
# Table 1: data and splits
idx = {s: pd.read_csv(f"data/index/hmu_grouped_s{s}.csv") for s in (0, 1, 2)}
g0 = idx[0]
t1 = pd.DataFrame([
    ["GasHisSDB", "Shanghai, China", "160 × 160", "Abnormal / Normal", "13,124 / 20,160", "random patch split (no IDs)"],
    ["HMU-GC-HE-30K", "Harbin, China", "224 × 224", "TUM / NOR", "3,887 / 3,887",
     f"slide-grouped (~{int((g0.split == 'gap').sum())} gap tiles left out) and random, same pool"],
], columns=["Dataset", "Source", "Patch (px)", "Classes used (tumour / normal)", "Patches", "Splits (70/15/15, 3 seeds)"])
t1.to_csv(TAB / "table1_datasets.csv", index=False)

# Table 2: main results (no adaptation)
rows = []
for mod in MODELS:
    for setting, lab in (("hmu-grouped", "HMU (grouped)"), ("hmu-random", "HMU (random)"), ("gashis-random", "GasHisSDB (random)")):
        d = B[(B.model == mod) & (B.setting == setting)]
        if d.empty:
            continue
        i, c = d[d["where"] == "in-domain"], d[d["where"] == "cross"]
        rows.append([mod, lab, ms(i.auc), ms(i.bal_acc), ms(c.auc), ms(c.bal_acc), ms(c.ece),
                     f"{np.mean(i.auc.to_numpy() - c.auc.to_numpy()):.3f}"])
t2 = pd.DataFrame(rows, columns=["Model", "Trained on", "In-domain AUC", "In-domain bal. acc.",
                                 "Cross AUC", "Cross bal. acc.", "Cross ECE", "AUC drop"])
t2.to_csv(TAB / "table2_main_results.csv", index=False)

# Table 3: adaptation + referral (foundation models, cross-dataset)
U = []
for folder, prefix, mod in (("results/phikon", "probe_phikon_", "Phikon"), ("results/uni", "probe_uni_", "UNI")):
    u = pd.read_csv(f"{folder}/uncertainty_all.csv"); u = u[(u.score == "entropy") & u["eval"].str.endswith("target_all")]
    r = u.run.str.replace(prefix, "", regex=False)
    u["adapt"] = np.where(r.str.endswith("_target_std"), "target_std", np.where(r.str.endswith("_coral"), "coral", "none"))
    r = r.str.replace("_target_std", "", regex=False).str.replace("_coral", "", regex=False)
    u["setting"] = r.str.rsplit("_s", n=1).str[0]; u["model"] = mod; U.append(u)
U = pd.concat(U)
rows = []
for mod in ("Phikon", "UNI"):
    for setting, lab in (("hmu-grouped", "HMU (grouped) → GasHisSDB"), ("hmu-random", "HMU (random) → GasHisSDB"),
                         ("gashis-random", "GasHisSDB → HMU")):
        row = [mod, lab]
        for adapt in ("none", "target_std", "coral"):
            d = U[(U.model == mod) & (U.setting == setting) & (U.adapt == adapt)]
            row += [ms(d["balacc@100"]), ms(d["balacc@80"])]
        dd = U[(U.model == mod) & (U.setting == setting) & (U.adapt == "target_std")]
        row.append(ms(dd["err_detect_auc"]))
        rows.append(row)
t3 = pd.DataFrame(rows, columns=["Model", "Direction", "None: bal. acc.", "None: @80% kept",
                                 "Target-std: bal. acc.", "Target-std: @80% kept",
                                 "CORAL: bal. acc.", "CORAL: @80% kept", "Error-detection AUROC (target-std)"])
t3.to_csv(TAB / "table3_adaptation_referral.csv", index=False)

for t, name, cap in ((t1, "table1_datasets", "Datasets and splits."),
                     (t2, "table2_main_results", "In-domain and cross-dataset performance (mean ± SD over 3 seeds)."),
                     (t3, "table3_adaptation_referral", "Label-free adaptation and uncertainty-based referral on the unseen dataset.")):
    def esc(v):
        v = str(v)
        for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#")):
            v = v.replace(a, b)
        return v
    lines = [r"\begin{tabular}{" + "l" * t.shape[1] + "}", r"\hline",
             " & ".join(esc(c) for c in t.columns) + r" \\", r"\hline"]
    lines += [" & ".join(esc(v) for v in row) + r" \\" for row in t.itertuples(index=False)]
    lines += [r"\hline", r"\end{tabular}"]
    tex = "\n".join(lines) + "\n"
    (TAB / f"{name}.tex").write_text(f"% {cap}\n" + tex.replace("±", r"$\pm$").replace("→", r"$\rightarrow$")
                                     .replace("×", r"$\times$"))
    print("table", name)
print("done")
