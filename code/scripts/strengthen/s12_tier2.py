"""Tier 2 analysis, HMU (slide-grouped) -> GasHisSDB, 3 seeds.
Foundation models: Macenko-normalised target images, with and without target standardisation, vs raw images.
ResNet50: AdaBN and Macenko vs the unadapted model. Point predictions = ensemble/MC mean (as in the paper)."""
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from common import ROOT, fit_cache, load, transform, predict, metrics, selective_metrics, oracle_ba
rows = []
for enc in ("uni", "phikon"):
    R = load(ROOT / f"data/features/gashis_{enc}.npz")
    fM = ROOT / f"data/features/gashis_{enc}_macenko.npz"
    variants = [("raw images", R["X"].astype(float))]
    if fM.exists():
        Mk = load(fM); assert (Mk["path"] == R["path"]).all(), "path order differs"
        variants.append(("Macenko", Mk["X"].astype(float)))
    fD = ROOT / f"data/features/gashis_{enc}_macenko_dataset.npz"
    if fD.exists():
        Md = load(fD); assert (Md["path"] == R["path"]).all(), "path order differs"
        variants.append(("Macenko (dataset-level)", Md["X"].astype(float)))
    y = R["y"].astype(int)
    for seed in (0, 1, 2):
        M = fit_cache(enc, "hmu-grouped", seed)
        for vname, X in variants:
            for method in ("none", "target_std"):
                p_det, p_ens, H = predict(transform(X, M, method), M)
                m = metrics(y, p_det, p_ens, H)
                m.update(oracle_ens=oracle_ba(y, p_ens), model=enc, images=vname, adapt=method, seed=seed)
                rows.append(m)
for seed in (0, 1, 2):
    base = f"runs/cnn_resnet50_hmu-grouped_hed_s{seed}"
    for vname, run in (("raw images", base), ("AdaBN", base + "_adabn"), ("Macenko", base + "_macenko")):
        f = ROOT / run / "pred_hmu-grouped_to_gashis_target_all.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f); y = d.label.values.astype(int); p = d.prob.values; H = d.entropy.values
        sel = selective_metrics(y, p, H)
        rows.append(dict(model="resnet50", images=vname, adapt="none", seed=seed, auc=roc_auc_score(y, p),
                         ba_det=balanced_accuracy_score(y, (d.prob_det.values >= .5).astype(int)),
                         ba_ens=balanced_accuracy_score(y, (p >= .5).astype(int)), ba80=sel["balacc@80"],
                         err_auc=sel["err_detect_auc"], oracle_ens=oracle_ba(y, p)))
d = pd.DataFrame(rows); d.to_csv(ROOT / "results/strengthen/tier2.csv", index=False)
s = d.groupby(["model", "images", "adapt"])[["auc", "ba_ens", "ba80", "err_auc", "oracle_ens"]].agg(["mean", "std"])
pd.set_option("display.width", 220); print(s.round(3).to_string())
