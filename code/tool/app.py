"""GastroMap: tumour heatmap with per-slide adaptation, uncertainty referral and an
"unfamiliar slide" warning, for gastric H&E images. Runs locally; nothing is uploaded.

Pipeline (validated in the paper):
  tissue tiles (224 px at ~0.5 um/px) -> UNI features (frozen)
  -> hospital-level standardisation (statistics from several slides of the same site; recommended)
     or per-slide standardisation (fallback; tumour % then not reported, see below)
  -> 10-member probe ensemble (trained on HMU) -> tumour probability + uncertainty
  -> workload-aware referral (expected accuracy from the HMU->GasHisSDB curve)
  -> unfamiliar-slide check (kNN distance to the HMU feature bank)

Usage (from the project root, with the gcx-env activated):
  python code/tool/app.py                                # web app in your browser
  python code/tool/app.py --cli data/raw/demo_slides/x.svs --review 20 --out results/gastromap
  python code/tool/app.py --calibrate a.svs b.svs c.svs --name hospitalA   # build a hospital calibration
  python code/tool/app.py --cli x.svs --calib hospitalA                     # use it
Why hospital-level: standardising a slide with its own statistics also removes how much tumour it contains,
so the per-slide tumour fraction stays near 45-51% whatever the truth (results/strengthen/fraction_bias.csv).
Statistics pooled over several slides of a site keep each slide's tumour content (site_calibration.csv).
Whole-slide images need:  pip install openslide-python openslide-bin
Research prototype - not a medical device.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "src"))
from gcx.models import FrozenEncoder, get_device  # noqa: E402
from gcx.transforms import make_transforms  # noqa: E402

BUNDLE = ROOT / "code" / "tool" / "bundle" / "gastromap_uni_bundle.npz"
PER_SLIDE_CURVES = ROOT / "code" / "tool" / "bundle" / "per_slide_curves.npz"
CALIB_DIR = ROOT / "data" / "calibration"
NO_CALIB = "Per-slide (no hospital calibration)"
MIN_CALIB_SLIDES, CALIB_TILES_PER_SLIDE = 3, 300
MIN_LABELS_PER_CLASS = 10          # few-label calibration: at least 10 tumour + 10 normal tiles (50 recommended)
TILE, TARGET_MPP, MIN_TILES_ADAPT, UNFAMILIAR_WARN = 224, 0.5, 100, 0.10
WSI_EXT = {".svs", ".ndpi", ".mrxs", ".scn", ".vms", ".vmu", ".bif", ".tif", ".tiff"}


# ----------------------------------------------------------------------------- tissue + tiling
def tissue_fraction(rgb: np.ndarray) -> float:
    a = rgb.astype(np.float32) / 255.0
    mx, mn = a.max(-1), a.min(-1)
    sat = (mx - mn) / (mx + 1e-6)
    return float(((sat > 0.07) & (mx < 0.92)).mean())


class Source:
    """Uniform access to a whole-slide image (via openslide) or an ordinary image file."""

    def __init__(self, path: str, image_mpp: float = TARGET_MPP):
        self.path, self.slide = str(path), None
        if Path(path).suffix.lower() in WSI_EXT:
            try:
                import openslide
                self.slide = openslide.OpenSlide(self.path)
            except Exception:
                self.slide = None          # e.g. plain .tif or openslide not installed: read as image
        if self.slide is not None:
            s = self.slide
            self.mpp = float(s.properties.get("openslide.mpp-x", 0.25) or 0.25)
            self.scale = TARGET_MPP / self.mpp                      # level-0 px per target px
            self.W, self.H = (int(v / self.scale) for v in s.dimensions)
        else:
            img = Image.open(self.path).convert("RGB")
            f = image_mpp / TARGET_MPP
            if abs(f - 1) > 0.02:
                img = img.resize((int(img.width * f), int(img.height * f)), Image.BILINEAR)
            self.img = np.asarray(img); self.H, self.W = self.img.shape[:2]

    def thumbnail(self, max_side=1600) -> np.ndarray:
        if self.slide is not None:
            return np.asarray(self.slide.get_thumbnail((max_side, max_side)).convert("RGB"))
        r = min(1.0, max_side / max(self.W, self.H))
        return np.asarray(Image.fromarray(self.img).resize((max(1, int(self.W * r)), max(1, int(self.H * r)))))

    def tile(self, i: int, j: int) -> np.ndarray:
        if self.slide is None:
            return self.img[i * TILE:(i + 1) * TILE, j * TILE:(j + 1) * TILE]
        s = self.slide
        x0, y0, size0 = int(j * TILE * self.scale), int(i * TILE * self.scale), int(round(TILE * self.scale))
        lvl = s.get_best_level_for_downsample(self.scale)
        ds = s.level_downsamples[lvl]
        reg = s.read_region((x0, y0), lvl, (max(1, int(round(size0 / ds))),) * 2).convert("RGB")
        return np.asarray(reg.resize((TILE, TILE), Image.BILINEAR))

    def tissue_grid(self, min_tissue=0.3):
        """Candidate tiles from a thumbnail mask, then confirmed on the tile itself."""
        gh, gw = self.H // TILE, self.W // TILE
        th = self.thumbnail()
        ty, tx = th.shape[0] / max(gh, 1), th.shape[1] / max(gw, 1)
        cand = []
        for i in range(gh):
            for j in range(gw):
                patch = th[int(i * ty):max(int((i + 1) * ty), int(i * ty) + 1),
                           int(j * tx):max(int((j + 1) * tx), int(j * tx) + 1)]
                if patch.size and tissue_fraction(patch) > min_tissue * 0.5:
                    cand.append((i, j))
        return gh, gw, cand


# ----------------------------------------------------------------------------- model
class GastroMap:
    def __init__(self, device=None):
        b = np.load(BUNDLE)
        self.mu, self.sd = b["scaler_mean"].astype(np.float64), b["scaler_std"].astype(np.float64)
        self.W, self.b = b["coef"].astype(np.float64), b["intercept"].astype(np.float64)
        self.bank = b["bank"].astype(np.float32); self.k = int(b["knn_k"]); self.thr = float(b["knn_thr"])
        self.ref_hmu, self.ref_gas = float(b["ood_ref_hmu"]), float(b["ood_ref_gashis"])
        self.cov, self.cov_acc, self.cov_sd = b["cov_grid"], b["cov_balacc_mean"], b["cov_balacc_sd"]
        c = np.load(PER_SLIDE_CURVES)
        self.ps_cov, self.ps_worst, self.ps_best = c["cov_grid"], c["balacc_worst"], c["balacc_best"]
        self.fl_mean, self.fl_low = c["fewlabel_mean"], c["fewlabel_p10"]
        self.dev = device or get_device()
        self.enc = FrozenEncoder("uni", self.dev)
        self.tf = make_transforms(False, mean=self.enc.mean, std=self.enc.std)

    def features(self, tiles, bs=32, progress=None):
        import torch
        out = []
        for k in range(0, len(tiles), bs):
            x = torch.stack([self.tf(Image.fromarray(t)) for t in tiles[k:k + bs]])
            out.append(self.enc(x).float().cpu().numpy())
            if progress:
                progress(min(1.0, (k + bs) / len(tiles)), f"UNI features {min(k + bs, len(tiles))}/{len(tiles)}")
        return np.concatenate(out).astype(np.float64)

    def predict(self, F, calib=None):
        """calib: dict with mu/sd from a hospital calibration -> mode 'site'.
        Otherwise per-slide standardisation if the slide has enough tiles, else training statistics."""
        shift = 0.0
        if calib is not None and calib.get("type") == "labels":
            mode, Z, shift = "labels", (F - self.mu) / self.sd, calib["shift"]
        elif calib is not None:
            mode, Z = "site", (F - calib["mu"]) / (calib["sd"] + 1e-6)
        elif len(F) >= MIN_TILES_ADAPT:
            mode, Z = "per_slide", (F - F.mean(0)) / (F.std(0) + 1e-6)
        else:
            mode, Z = "unadapted", (F - self.mu) / self.sd
        P = 1 / (1 + np.exp(-(Z @ self.W.T + self.b - shift)))  # (N, members); shift moves the threshold
        p = np.clip(P.mean(1), 1e-7, 1 - 1e-7)
        H = -(p * np.log(p) + (1 - p) * np.log(1 - p)) / np.log(2)  # 0 = sure, 1 = maximally unsure
        return p, H, P.std(1), mode

    def unfamiliar_share(self, F, chunk=1024):
        Q = (F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-8)).astype(np.float32)
        d = []
        for i in range(0, len(Q), chunk):
            S = Q[i:i + chunk] @ self.bank.T
            d.append(1 - np.sort(S, axis=1)[:, -self.k:].mean(1))
        d = np.concatenate(d)
        return float((d > self.thr).mean()), d

    def expected_accuracy(self, review_pct, mode="site"):
        """site: curve measured with dataset-level standardisation (HMU->GasHisSDB), returns (mean, sd).
        per_slide: range over slides with 10-90% tumour, returns (worst, best). unadapted: (nan, nan)."""
        keep = 1 - review_pct / 100
        if mode == "site":
            return float(np.interp(keep, self.cov, self.cov_acc)), float(np.interp(keep, self.cov, self.cov_sd))
        if mode == "per_slide":
            return float(np.interp(keep, self.ps_cov, self.ps_worst)), float(np.interp(keep, self.ps_cov, self.ps_best))
        if mode == "labels":   # (typical, 10th percentile over label sets and 10-90% tumour); 50 labelled tiles
            return float(np.interp(keep, self.ps_cov, self.fl_mean)), float(np.interp(keep, self.ps_cov, self.fl_low))
        return float("nan"), float("nan")


# ----------------------------------------------------------------------------- hospital calibration
def list_calibrations():
    return sorted(p.stem for p in CALIB_DIR.glob("*.npz"))


def load_calibration(name):
    if not name or name == NO_CALIB:
        return None
    z = np.load(CALIB_DIR / f"{name}.npz", allow_pickle=True)
    if "type" in z.files and str(z["type"]) == "labels":
        return dict(name=name, type="labels", shift=float(z["shift"]), n_tumour=int(z["n_tumour"]),
                    n_normal=int(z["n_normal"]))
    return dict(name=name, type="site", mu=z["mu"], sd=z["sd"], n_slides=int(z["n_slides"]), n_tiles=int(z["n_tiles"]))


def build_label_calibration(model, F, labels, name):
    """Few-label calibration: F = UNI features of labelled tiles, labels = 1 tumour / 0 normal.
    Finds the threshold on the unadapted ensemble probability that maximises balanced accuracy on these tiles,
    and stores it as a logit shift. Features are not standardised, so each slide keeps its tumour content."""
    labels = np.asarray(labels).astype(int)
    n1, n0 = int((labels == 1).sum()), int((labels == 0).sum())
    if min(n1, n0) < MIN_LABELS_PER_CLASS:
        raise ValueError(f"Label at least {MIN_LABELS_PER_CLASS} tumour and {MIN_LABELS_PER_CLASS} normal tiles "
                         f"(now {n1} tumour, {n0} normal); about 25 of each is recommended.")
    p = np.clip((1 / (1 + np.exp(-(((F - model.mu) / model.sd) @ model.W.T + model.b)))).mean(1), 1e-6, 1 - 1e-6)
    cands = np.unique(p)
    tpr = np.array([(p[labels == 1] >= t).mean() for t in cands]); tnr = np.array([(p[labels == 0] < t).mean() for t in cands])
    ba = 0.5 * (tpr + tnr); t = float(np.median(cands[np.isclose(ba, ba.max())]))
    name = "".join(ch for ch in name.strip() if ch.isalnum() or ch in "-_") or "hospital_labels"
    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(CALIB_DIR / f"{name}.npz", type="labels", shift=np.log(t / (1 - t)), threshold=t, n_tumour=n1, n_normal=n0,
             balacc_on_labels=float(ba.max()))
    return name, t, float(ba.max()), n1, n0


def build_calibration(model, paths, name, tiles_per_slide=CALIB_TILES_PER_SLIDE, progress=None):
    """Pool UNI features of up to `tiles_per_slide` random tissue tiles from each slide of one hospital."""
    if len(paths) < MIN_CALIB_SLIDES:
        raise ValueError(f"Choose at least {MIN_CALIB_SLIDES} slides from the same hospital.")
    name = "".join(ch for ch in name.strip() if ch.isalnum() or ch in "-_") or "hospital"
    feats, used = [], []
    for n, path in enumerate(paths):
        src = Source(path)
        _, _, cand = src.tissue_grid()
        rng = np.random.default_rng(0)
        tiles = []
        for k in rng.permutation(len(cand)):
            t = src.tile(*cand[k])
            if t.shape[:2] == (TILE, TILE) and tissue_fraction(t) > 0.3:
                tiles.append(t)
            if len(tiles) >= tiles_per_slide:
                break
        if progress:
            progress(n / len(paths), f"calibration slide {n + 1}/{len(paths)}: {len(tiles)} tiles")
        if len(tiles) >= 50:
            feats.append(model.features(tiles)); used.append(Path(path).name)
    if len(used) < MIN_CALIB_SLIDES:
        raise ValueError(f"Only {len(used)} slides had enough tissue; need {MIN_CALIB_SLIDES}.")
    F = np.concatenate(feats)
    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(CALIB_DIR / f"{name}.npz", mu=F.mean(0), sd=F.std(0), n_slides=len(used), n_tiles=len(F),
             slides=np.array(used, dtype=object))
    return name, len(used), len(F)


# ----------------------------------------------------------------------------- analysis + rendering
def analyse(model, path, review_pct=20, image_mpp=TARGET_MPP, max_tiles=0, progress=None, calib=None):
    t0 = time.time()
    src = Source(path, image_mpp)
    gh, gw, cand = src.tissue_grid()
    if max_tiles and len(cand) > max_tiles:          # even subsample over the tissue, keeps the map readable
        cand = [cand[i] for i in np.linspace(0, len(cand) - 1, max_tiles).astype(int)]
    tiles, coords = [], []
    for n, (i, j) in enumerate(cand):
        t = src.tile(i, j)
        if t.shape[:2] == (TILE, TILE) and tissue_fraction(t) > 0.3:
            tiles.append(t); coords.append((i, j))
        if progress and n % 200 == 0:
            progress(n / max(len(cand), 1) * 0.2, f"finding tissue {n}/{len(cand)}")
    if not tiles:
        return None
    F = model.features(tiles, progress=(lambda f, m: progress(0.2 + 0.75 * f, m)) if progress else None)
    p, H, sd, mode = model.predict(F, calib)
    unf, knn = model.unfamiliar_share(F)
    n_rev = int(round(review_pct / 100 * len(p)))
    review = np.zeros(len(p), bool); review[np.argsort(-H)[:n_rev]] = True
    keep = ~review
    exp_acc, exp_sd = model.expected_accuracy(review_pct, mode)
    df = pd.DataFrame(dict(row=[c[0] for c in coords], col=[c[1] for c in coords], x_px=[c[1] * TILE for c in coords],
                           y_px=[c[0] * TILE for c in coords], p_tumour=p, uncertainty=H, ensemble_sd=sd,
                           knn_distance=knn, unfamiliar=knn > model.thr, referred=review))
    summary = dict(
        file=Path(path).name, tiles=len(p), grid=f"{gh}x{gw}", seconds=round(time.time() - t0, 1),
        mode=mode, calibration=(calib or {}).get("name"), tumour_fraction_all=float((p >= 0.5).mean()),
        tumour_fraction_kept=float((p[keep] >= 0.5).mean()) if keep.any() else float("nan"),
        review_pct=review_pct, referred_tiles=int(review.sum()), expected_bal_acc=exp_acc, expected_bal_acc_sd=exp_sd,
        unfamiliar_share=unf, unfamiliar_warning=unf > UNFAMILIAR_WARN, ref_hmu=model.ref_hmu, ref_gas=model.ref_gas)
    summary["_features"] = F
    return src, df, tiles, summary


def render(src, df, tiles, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colormaps
    gh, gw = (int(v) for v in summary["grid"].split("x"))
    th = src.thumbnail(1600)
    sy, sx = th.shape[0] / gh, th.shape[1] / gw

    def grid(col):
        G = np.full((gh, gw), np.nan)
        G[df.row.to_numpy(), df.col.to_numpy()] = df[col].to_numpy()
        return G

    def overlay(G, cmap, alpha=0.5):
        rgba = colormaps[cmap](np.nan_to_num(G, nan=0.0)); rgba[..., 3] = np.where(np.isnan(G), 0, alpha)
        heat = Image.fromarray((rgba * 255).astype(np.uint8)).resize((th.shape[1], th.shape[0]), Image.NEAREST)
        return np.asarray(Image.alpha_composite(Image.fromarray(th).convert("RGBA"), heat).convert("RGB"))

    figs = {}
    for key, col, cmap, title in (("tumour", "p_tumour", "RdYlBu_r", "Tumour probability (red = tumour)"),
                                  ("uncertainty", "uncertainty", "magma", "Uncertainty (bright = unsure)")):
        fig, ax = plt.subplots(figsize=(6, 6 * th.shape[0] / th.shape[1] + 0.4))
        ax.imshow(overlay(grid(col), cmap)); ax.set_axis_off(); ax.set_title(title, fontsize=10)
        if key == "uncertainty":
            r = df[df.referred]
            ax.scatter((r.col + 0.5) * sx, (r.row + 0.5) * sy, s=6, facecolor="none", edgecolor="#00e5ff", lw=0.6)
        sm = plt.cm.ScalarMappable(cmap=cmap); sm.set_clim(0, 1)
        fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.01)
        figs[key] = fig
    # review gallery: most uncertain tiles
    top = df.sort_values("uncertainty", ascending=False).head(12)
    fig, axes = plt.subplots(2, 6, figsize=(9, 3.4))
    for ax in axes.ravel():
        ax.set_axis_off()
    for ax, (ix, r) in zip(axes.ravel(), top.iterrows()):
        ax.imshow(tiles[ix]); ax.set_title(f"p={r.p_tumour:.2f}  u={r.uncertainty:.2f}", fontsize=7)
    fig.suptitle("Please review: the most uncertain tiles", fontsize=9)
    figs["review"] = fig
    return figs


def summary_text(s):
    lines = [f"File: {s['file']}   tiles analysed: {s['tiles']}   time: {s['seconds']} s"]
    if s["mode"] == "site":
        lines += [f"Adaptation: hospital calibration '{s['calibration']}'",
                  f"Estimated tumour fraction of tissue: {s['tumour_fraction_all']:.1%} "
                  f"(confident tiles only: {s['tumour_fraction_kept']:.1%}); approximate, tends towards the middle",
                  f"Referred for pathologist review: {s['referred_tiles']} tiles ({s['review_pct']}%)",
                  f"Expected balanced accuracy on the remaining tiles: {s['expected_bal_acc']:.2f} ± "
                  f"{s['expected_bal_acc_sd']:.2f} (measured at a new hospital: HMU -> GasHisSDB)"]
    elif s["mode"] == "labels":
        lines += [f"Adaptation: few-label calibration '{s['calibration']}' (decision threshold set from labelled tiles)",
                  f"Estimated tumour fraction of tissue: {s['tumour_fraction_all']:.1%} "
                  f"(confident tiles only: {s['tumour_fraction_kept']:.1%}); approximate, tends towards the middle",
                  f"Referred for pathologist review: {s['referred_tiles']} tiles ({s['review_pct']}%)",
                  f"Expected balanced accuracy on the remaining tiles: {s['expected_bal_acc']:.2f} "
                  f"(1 in 10 label sets gives below {s['expected_bal_acc_sd']:.2f}; simulated with 50 labelled tiles, HMU -> GasHisSDB)"]
    elif s["mode"] == "per_slide":
        lines += ["Adaptation: per-slide (no hospital calibration chosen)",
                  "Tumour fraction: not reported. Per-slide adaptation removes the slide's overall tumour content, "
                  "so the percentage would stay near 50% whatever the truth. Build a hospital calibration "
                  f"from {MIN_CALIB_SLIDES}+ slides of the same site to get it.",
                  f"Referred for pathologist review: {s['referred_tiles']} tiles ({s['review_pct']}%)",
                  f"Expected balanced accuracy on the remaining tiles: {s['expected_bal_acc']:.2f}-"
                  f"{s['expected_bal_acc_sd']:.2f} (lower when a slide is mostly tumour or mostly normal)"]
    else:
        lines += [f"Adaptation: none (fewer than {MIN_TILES_ADAPT} tissue tiles and no hospital calibration)",
                  "Tumour fraction and expected accuracy: not reported; without adaptation, decisions at a new "
                  "hospital are unreliable.",
                  f"Referred for pathologist review: {s['referred_tiles']} tiles ({s['review_pct']}%)"]
    lines.append(f"Unfamiliar tiles: {s['unfamiliar_share']:.0%} (reference: training-like slides {s['ref_hmu']:.0%}, "
                 f"other hospital {s['ref_gas']:.0%})")
    if s["unfamiliar_warning"]:
        lines.append("⚠ This slide looks unlike the training data. Treat all outputs with extra caution.")
    lines.append("Research prototype - not for clinical diagnosis.")
    return "\n".join(lines)


def save_outputs(out_dir, src, df, tiles, summary):
    import matplotlib.pyplot as plt
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    stem = Path(summary["file"]).stem
    figs = render(src, df, tiles, summary)
    paths = {}
    for k, fig in figs.items():
        p = out / f"{stem}_{k}.png"; fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig); paths[k] = str(p)
    df.to_csv(out / f"{stem}_tiles.csv", index=False)
    (out / f"{stem}_summary.txt").write_text(summary_text(summary))
    paths["csv"] = str(out / f"{stem}_tiles.csv")
    return paths


# ----------------------------------------------------------------------------- web app
def launch(model, share=False):
    import gradio as gr

    demo_dir = ROOT / "data" / "raw" / "demo_slides"

    def slides():
        return sorted(p.name for p in demo_dir.glob("*") if p.is_file() and not p.name.startswith("."))

    def run(file, chosen, review, mpp, max_tiles, calib_name, progress=gr.Progress()):
        if file is not None:
            path = file if isinstance(file, str) else file.name
        elif chosen:
            path = str(demo_dir / chosen)
        else:
            return None, None, None, "Choose a slide from the list, or upload a file.", None, None
        res = analyse(model, path, review, mpp, int(max_tiles),
                      progress=lambda f, m: progress(f, desc=m), calib=load_calibration(calib_name))
        if res is None:
            return None, None, None, "No tissue found.", None, None
        src, df, tiles, s = res
        paths = save_outputs(ROOT / "results" / "gastromap", src, df, tiles, s)
        state = dict(F=s["_features"], tiles=tiles, shown=[])
        return paths["tumour"], paths["uncertainty"], paths["review"], summary_text(s), paths["csv"], state

    def expected(review, calib_name=NO_CALIB):
        cal = load_calibration(calib_name) if calib_name and calib_name != NO_CALIB else None
        if cal is not None and cal["type"] == "labels":
            a, lo = model.expected_accuracy(review, "labels")
            return (f"Reviewing {review}% of tiles -> expected balanced accuracy on the rest: {a:.2f} "
                    f"(1 in 10 label sets gives below {lo:.2f}; label more tiles to be safer)")
        if cal is not None:
            a, sd = model.expected_accuracy(review, "site")
            return f"Reviewing {review}% of tiles -> expected balanced accuracy on the rest: {a:.2f} ± {sd:.2f}"
        lo, hi = model.expected_accuracy(review, "per_slide")
        return (f"Reviewing {review}% of tiles -> expected balanced accuracy on the rest: {lo:.2f}-{hi:.2f} "
                "(per-slide mode; choose a hospital calibration for a single estimate)")

    def calib_choices():
        return [NO_CALIB] + list_calibrations()

    def label_set(state, offset=0):
        """24 tiles spread over the unadapted tumour probability, so both classes appear."""
        if not state:
            return [], gr.update(choices=[], value=[]), gr.update(choices=[], value=[]), state
        F = state["F"]
        p = (1 / (1 + np.exp(-(((F - model.mu) / model.sd) @ model.W.T + model.b)))).mean(1)
        order = np.argsort(p)
        rng = np.random.default_rng(offset)
        bins = np.array_split(order, 24)
        idx = [int(rng.choice(b)) for b in bins if len(b)]
        state["shown"] = state.get("shown", []) + [i for i in idx if i not in state.get("shown", [])]
        names = [f"#{i}" for i in state["shown"]]
        gal = [(state["tiles"][i], f"#{i}") for i in state["shown"]]
        return gal, gr.update(choices=names), gr.update(choices=names), state

    def save_labels(state, tum, nor, name):
        if not state:
            return "Analyse a slide first.", gr.update()
        both = set(tum or []) & set(nor or [])
        if both:
            return f"These tiles are ticked as both tumour and normal: {', '.join(sorted(both))}", gr.update()
        ids = [int(x[1:]) for x in (tum or [])] + [int(x[1:]) for x in (nor or [])]
        labels = [1] * len(tum or []) + [0] * len(nor or [])
        try:
            nm, t, ba, n1, n0 = build_label_calibration(model, state["F"][ids], labels, name or "hospital_labels")
        except ValueError as e:
            return str(e), gr.update()
        return (f"Saved few-label calibration '{nm}' from {n1} tumour and {n0} normal tiles (threshold {t:.2f}; "
                f"balanced accuracy on the labelled tiles {ba:.2f}). Select it under *Hospital calibration* and "
                "click Analyse again.", gr.update(choices=calib_choices(), value=nm))

    def make_calib(chosen, name, progress=gr.Progress()):
        try:
            nm, ns, nt = build_calibration(model, [str(demo_dir / c) for c in (chosen or [])], name or "hospital",
                                           progress=lambda f, m: progress(f, desc=m))
        except ValueError as e:
            return str(e), gr.update()
        return f"Saved calibration '{nm}' from {ns} slides ({nt} tiles).", gr.update(choices=calib_choices(), value=nm)

    with gr.Blocks(title="GastroMap") as demo:
        gr.Markdown("## GastroMap: gastric tumour map with uncertainty\nRuns on this computer only. "
                    "Research prototype, not a medical device.")
        with gr.Row():
            with gr.Column():
                pick = gr.Dropdown(choices=slides(), value=(slides() or [None])[0],
                                   label="Choose a slide from data/raw/demo_slides (recommended for large slides)")
                refresh = gr.Button("Refresh list", size="sm")
                f = gr.File(label="…or upload an image / small slide (overrides the list)")
            with gr.Column():
                calib = gr.Dropdown(choices=calib_choices(), value=NO_CALIB, label="Hospital calibration")
                review = gr.Slider(0, 50, value=20, step=5, label="Pathologist review budget (% of tiles)")
                exp = gr.Markdown(expected(20))
                mpp = gr.Number(value=0.5, label="Image resolution (µm per pixel; images only, slides read it automatically)")
                mt = gr.Number(value=3000, label="Max tiles (0 = all; fewer = faster)")
        review.change(expected, [review, calib], exp)
        calib.change(expected, [review, calib], exp)
        with gr.Accordion("Build a hospital calibration (recommended: 3 or more slides from the same hospital)", open=False):
            cal_pick = gr.CheckboxGroup(choices=slides(), label="Slides from one hospital")
            cal_name = gr.Textbox(value="hospital", label="Calibration name")
            cal_btn = gr.Button("Build calibration")
            cal_msg = gr.Markdown()
        cal_btn.click(make_calib, [cal_pick, cal_name], [cal_msg, calib])
        refresh.click(lambda: (gr.update(choices=slides()), gr.update(choices=slides())), None, [pick, cal_pick])
        btn = gr.Button("Analyse", variant="primary")
        with gr.Row():
            o1 = gr.Image(label="Tumour map"); o2 = gr.Image(label="Uncertainty map (cyan = referred)")
        o3 = gr.Image(label="Please review"); o4 = gr.Textbox(label="Summary", lines=8); o5 = gr.File(label="Per-tile CSV")
        st = gr.State(None)
        btn.click(run, [f, pick, review, mpp, mt, calib], [o1, o2, o3, o4, o5, st])
        with gr.Accordion("Few-label calibration (one slide + about 25 tumour and 25 normal tiles labelled by a "
                          "pathologist)", open=False):
            gr.Markdown("Analyse a slide first, then show tiles, tick the ones you are sure about, and save.")
            show = gr.Button("Show tiles to label (click again for 24 more)")
            gal = gr.Gallery(label="Tiles", columns=8, height="auto")
            tum = gr.CheckboxGroup(label="Tumour tiles")
            nor = gr.CheckboxGroup(label="Normal tiles")
            lab_name = gr.Textbox(value="hospital_labels", label="Calibration name")
            lab_btn = gr.Button("Save few-label calibration")
            lab_msg = gr.Markdown()
            clicks = gr.State(0)
        show.click(lambda s_, n: (*label_set(s_, n), n + 1), [st, clicks], [gal, tum, nor, st, clicks])
        lab_btn.click(save_labels, [st, tum, nor, lab_name], [lab_msg, calib])
    demo.launch(share=share, inbrowser=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli", default=None, help="analyse one file without the web app")
    ap.add_argument("--review", type=int, default=20)
    ap.add_argument("--mpp", type=float, default=TARGET_MPP, help="µm/px of a plain image (slides: automatic)")
    ap.add_argument("--max_tiles", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "results" / "gastromap"))
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--calib", default=None, help="name of a saved hospital calibration")
    ap.add_argument("--calibrate", nargs="+", default=None, help="build a calibration from these slides")
    ap.add_argument("--name", default="hospital", help="name for --calibrate")
    a = ap.parse_args()
    model = GastroMap()
    if a.calibrate:
        nm, ns, nt = build_calibration(model, a.calibrate, a.name,
                                       progress=lambda f, m: print(f"\r{m:<60}", end="", flush=True))
        print(f"\nSaved calibration '{nm}' from {ns} slides ({nt} tiles) in {CALIB_DIR}")
        return
    if a.cli:
        res = analyse(model, a.cli, a.review, a.mpp, a.max_tiles,
                      progress=lambda f, m: print(f"\r{m:<40}", end="", flush=True), calib=load_calibration(a.calib))
        print()
        if res is None:
            print("No tissue found."); return
        paths = save_outputs(a.out, *res)
        print(summary_text(res[3])); print("saved:", paths)
        return
    launch(model, a.share)


if __name__ == "__main__":
    main()
