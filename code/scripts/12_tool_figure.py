"""Fig 7: GastroMap on an external TCGA slide (all tissue tiles)."""
import glob, numpy as np, pandas as pd, openslide
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "legend.frameon": False,
                     "savefig.dpi": 300, "savefig.bbox": "tight"})
TILE = 224
STEM = "TCGA-HF-A5NB-01Z-00-DX1.CA67064B-3281-4144-A2A2-E0BC43E6F8D6"   # demo slide (hospital-calibrated run)
svs = f"data/raw/demo_slides/{STEM}.svs"
df = pd.read_csv(f"results/gastromap/{STEM}_tiles.csv")
B = np.load("code/tool/bundle/gastromap_uni_bundle.npz")
S = openslide.OpenSlide(svs); W, H = S.dimensions
thumb = np.asarray(S.get_thumbnail((1400, 1400)).convert("RGB"))
gh, gw = H // TILE, W // TILE
def grid(col):
    G = np.full((gh, gw), np.nan); G[df.row, df.col] = df[col]; return np.ma.masked_invalid(G)
x0, x1 = df.x_px.min() - TILE, df.x_px.max() + 2 * TILE
y0, y1 = df.y_px.min() - TILE, df.y_px.max() + 2 * TILE
cm_p = LinearSegmentedColormap.from_list("p", ["#2a78d6", "#f2f1ee", "#d0352b"])
cm_u = LinearSegmentedColormap.from_list("u", ["#fdf3dc", "#e8a33d", "#7a3b00"])

fig = plt.figure(figsize=(7.2, 6.3))
gs = GridSpec(2, 3, figure=fig, height_ratios=[1.15, 1], hspace=0.38, wspace=0.28)
def slide_ax(k, title):
    ax = fig.add_subplot(gs[0, k]); ax.imshow(thumb, extent=[0, W, H, 0])
    ax.set_xlim(x0, x1); ax.set_ylim(y1, y0); ax.set_axis_off()
    ax.set_title(title, loc="left", fontsize=8, color=INK); return ax
def letter(ax, s, dx=-0.02):
    ax.text(dx, 1.06, s, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="right")
ext = [0, gw * TILE, gh * TILE, 0]
a = slide_ax(0, "H&E slide (TCGA, external)"); letter(a, "a", -0.04)
cb0 = fig.colorbar(plt.cm.ScalarMappable(cmap=cm_p), ax=a, orientation="horizontal", fraction=0.05, pad=0.02); cb0.ax.set_visible(False)
a.plot([x1 - 2 * TILE - 4000, x1 - 2 * TILE], [y1 - 2 * TILE] * 2, color=INK, lw=1.5)
a.text(x1 - 2 * TILE - 2000, y1 - 3.3 * TILE, "2 mm", ha="center", fontsize=6.5)
b = slide_ax(1, "Tumour probability (calibrated)"); letter(b, "b", -0.04)
im = b.imshow(grid("p_tumour"), cmap=cm_p, vmin=0, vmax=1, alpha=0.75, extent=ext, interpolation="nearest")
cb = fig.colorbar(im, ax=b, orientation="horizontal", fraction=0.05, pad=0.02, ticks=[0, 0.5, 1]); cb.ax.tick_params(labelsize=6)
cb.set_label("normal ← → tumour", fontsize=6.5); cb.outline.set_visible(False)
c = slide_ax(2, "Uncertainty + referred tiles"); letter(c, "c", -0.04)
im = c.imshow(grid("uncertainty"), cmap=cm_u, vmin=0, vmax=1, alpha=0.6, extent=ext, interpolation="nearest")
r = df[df.referred]
c.scatter(r.x_px + TILE / 2, r.y_px + TILE / 2, s=1.6, marker="s", color="#1f6fd1", lw=0)
cb = fig.colorbar(im, ax=c, orientation="horizontal", fraction=0.05, pad=0.02, ticks=[0, 0.5, 1]); cb.ax.tick_params(labelsize=6)
cb.set_label("entropy; blue = referred (20%)", fontsize=6.5); cb.outline.set_visible(False)

# d workload curve
d = fig.add_subplot(gs[1, 0]); letter(d, "d", -0.12)
rev = (1 - B["cov_grid"]) * 100; m, s = B["cov_balacc_mean"], B["cov_balacc_sd"]
d.fill_between(rev, m - s, m + s, color="#1baf7a", alpha=0.18, lw=0)
d.plot(rev, m, color="#1baf7a", lw=2, marker="o", ms=3.5)
y20 = np.interp(0.8, B["cov_grid"], m)
d.annotate(f"review 20% →\n{y20:.2f} on the rest", (20, y20), (24, 0.855), fontsize=6.5, color=INK2,
           arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
d.set_xlabel("Tiles sent to pathologist (%)"); d.set_ylabel("Balanced accuracy, remaining tiles")
d.set_title("Workload vs accuracy", loc="left", fontsize=8); d.grid(color=GRID, lw=0.5); d.set_axisbelow(True)
d.set_xlim(-2, 52); d.set_ylim(0.83, 0.97)
d.text(0.98, 0.04, "measured HMU → GasHisSDB\n(3 seeds, ±SD)", transform=d.transAxes, ha="right", fontsize=6, color=INK2)

# e unfamiliar share
e = fig.add_subplot(gs[1, 1]); letter(e, "e", -0.12)
names = ["HMU\nheld-out", "This TCGA\nslide", "GasHisSDB"]
vals = [float(B["ood_ref_hmu"]), df.unfamiliar.mean(), float(B["ood_ref_gashis"])]
bars = e.bar(names, vals, color=["#b8b6b0", "#eb6834", "#b8b6b0"], width=0.6)
for bb, v in zip(bars, vals):
    e.text(bb.get_x() + bb.get_width() / 2, v + 0.012, f"{v:.0%}", ha="center", fontsize=7, color=INK)
e.axhline(0.10, color=INK2, lw=0.7, ls=(0, (3, 3))); e.text(2.32, 0.115, "warning\nat 10%", fontsize=6, color=INK2, ha="right", va="bottom")
e.set_ylabel("Unfamiliar tiles (share)"); e.set_ylim(0, 0.6); e.tick_params(axis="x", labelsize=6.5)
e.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
e.set_title("Unfamiliar-slide check", loc="left", fontsize=8); e.grid(axis="y", color=GRID, lw=0.5); e.set_axisbelow(True)

# f gallery
sub = GridSpecFromSubplotSpec(3, 3, subplot_spec=gs[1, 2], hspace=0.42, wspace=0.08)
flag = df.sort_values("uncertainty", ascending=False).head(3)
ct = df[df.p_tumour > 0.5].sort_values("uncertainty").head(3)
cn = df[df.p_tumour < 0.5].sort_values("uncertainty").head(3)
rows = [("Confident tumour", ct), ("Confident normal", cn), ("Flagged for review", flag)]
for i, (lab, t) in enumerate(rows):
    for j, (_, rr) in enumerate(t.iterrows()):
        ax = fig.add_subplot(sub[i, j])
        ax.imshow(np.asarray(S.read_region((int(rr.x_px), int(rr.y_px)), 0, (TILE, TILE)).convert("RGB")))
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(True); sp.set_color("#1f6fd1" if i == 2 else GRID); sp.set_linewidth(1.2 if i == 2 else 0.5)
        ax.set_xlabel(f"p={rr.p_tumour:.2f}", fontsize=5.8, labelpad=1, color=INK2)
        if j == 0:
            ax.text(0, 1.06, lab, transform=ax.transAxes, fontsize=6.5, color="#1f6fd1" if i == 2 else INK, va="bottom")
        if i == 0 and j == 0:
            ax.text(-0.12, 1.42, "f", transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="right"); ax.text(0, 1.42, "Tiles shown to the user", transform=ax.transAxes, fontsize=8, va="bottom")
fig.savefig("paper/figures/fig7_gastromap.pdf"); fig.savefig("paper/figures/fig7_gastromap.png")
print("saved", len(df), "tiles", df.referred.sum(), "referred")
