"""Paired bootstrap (patches, 1000 resamples; optimistic, GasHisSDB has no slide IDs) for Tier 2 comparisons.
Statistic = mean over 3 seeds of the per-seed difference."""
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from common import ROOT, fit_cache, load, transform, predict
rng = np.random.default_rng(0); B = 1000
y = load(ROOT / "data/features/gashis_uni.npz")["y"].astype(int); n = len(y)
P = {}
for seed in (0, 1, 2):
    base = f"runs/cnn_resnet50_hmu-grouped_hed_s{seed}"
    for k, run in (("cnn_raw", base), ("cnn_adabn", base + "_adabn"), ("cnn_macenko", base + "_macenko")):
        P[k, seed] = pd.read_csv(ROOT / run / "pred_hmu-grouped_to_gashis_target_all.csv").prob.values
    for enc in ("uni", "phikon"):
        M = fit_cache(enc, "hmu-grouped", seed)
        for img in (("", "_macenko", "_macenko_dataset") if enc == "uni" else ("", "_macenko")):
            X = load(ROOT / f"data/features/gashis_{enc}{img}.npz")["X"].astype(float)
            for ad in ("none", "target_std"):
                P[f"{enc}{img}_{ad}", seed] = predict(transform(X, M, ad), M)[1]
auc = roc_auc_score; ba = lambda yy, p: balanced_accuracy_score(yy, (p >= .5).astype(int))
comps = [("BA, ResNet50 AdaBN - none", "cnn_adabn", "cnn_raw", ba),
         ("BA, UNI target-std - ResNet50 AdaBN", "uni_target_std", "cnn_adabn", ba),
         ("AUC, ResNet50 Macenko - raw", "cnn_macenko", "cnn_raw", auc),
         ("AUC, Phikon Macenko - raw", "phikon_macenko_none", "phikon_none", auc),
         ("AUC, UNI Macenko - raw", "uni_macenko_none", "uni_none", auc),
         ("BA, UNI target-std: raw - Macenko", "uni_target_std", "uni_macenko_target_std", ba),
         ("AUC, UNI Macenko dataset-level - raw", "uni_macenko_dataset_none", "uni_none", auc),
         ("AUC, UNI Macenko dataset-level - per-patch", "uni_macenko_dataset_none", "uni_macenko_none", auc),
         ("BA, UNI target-std: raw - Macenko dataset-level", "uni_target_std", "uni_macenko_dataset_target_std", ba)]
rows = []
for name, a, b, f in comps:
    stat = lambda ii: np.mean([f(y[ii], P[a, s][ii]) - f(y[ii], P[b, s][ii]) for s in (0, 1, 2)])
    obs = stat(np.arange(n)); boots = np.array([stat(rng.integers(0, n, n)) for _ in range(B)])
    lo, hi = np.percentile(boots, [2.5, 97.5]); p = max(2 * min((boots <= 0).mean(), (boots >= 0).mean()), 1 / B)
    per_seed = [round(f(y, P[a, s]) - f(y, P[b, s]), 3) for s in (0, 1, 2)]
    rows.append(dict(comparison=name, diff=obs, ci_lo=lo, ci_hi=hi, p=p, per_seed=per_seed))
d = pd.DataFrame(rows); d.to_csv(ROOT / "results/strengthen/significance_tier2.csv", index=False)
pd.set_option("display.width", 200); print(d.round(4).to_string())
