"""Third hospital (HiESD, early gastric cancer, real slide IDs): HMU (slide-grouped) -> HiESD, 3 seeds.
Point prediction = ensemble/MC mean. 95% CIs from 2,000 bootstrap resamples of SLIDES (statistic = mean over seeds)."""
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from common import ROOT, fit_cache, load, transform, predict, selective_metrics, oracle_ba
idx = pd.read_csv(ROOT / "data/index/hiesd.csv")
y = idx.label.values.astype(int); slides = idx.slide.values; us = np.unique(slides)
P = {}                                            # (setting, seed) -> (p, H)
for enc in (() if (ROOT / "runs/strengthen/hiesd_preds.pkl").exists() else ("uni", "phikon")):
    F = load(ROOT / f"data/features/hiesd_{enc}.npz"); assert (F["path"] == idx.path.values).all()
    X = F["X"].astype(float)
    for seed in (0, 1, 2):
        M = fit_cache(enc, "hmu-grouped", seed)
        for ad in ("none", "target_std", "centre", "balanced_std"):
            _, p, H = predict(transform(X, M, ad), M); P[f"{enc}|{ad}", seed] = (p, H)
import sys, pickle
cache = ROOT / "runs/strengthen/hiesd_preds.pkl"
if cache.exists():
    P = pickle.load(open(cache, "rb"))
for seed in (0, 1, 2):
    for k, run in (("resnet50|none", f"runs/cnn_resnet50_hmu-grouped_hed_s{seed}"), ("resnet50|AdaBN", f"runs/cnn_resnet50_hmu-grouped_hed_s{seed}_adabn")):
        d = pd.read_csv(ROOT / run / "pred_hmu-grouped_to_hiesd_target_all.csv"); assert (d.path.values == idx.path.values).all()
        P[k, seed] = (d.prob.values, d.entropy.values)
pickle.dump(P, open(cache, "wb"))
if len(sys.argv) > 1 and sys.argv[1] == "cache":
    print("cached", len(P)); sys.exit()
from scipy.stats import rankdata
def fauc(t, s):
    r = rankdata(s); n1 = t.sum(); n0 = len(t) - n1
    return (r[t == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
def fba(t, yhat):
    return 0.5 * ((yhat[t == 1] == 1).mean() + (yhat[t == 0] == 0).mean())
def mets(ii, p, H):
    yy, pp, hh = y[ii], p[ii], H[ii]; yhat = (pp >= .5).astype(int)
    keep = np.argsort(hh, kind="stable")[:int(round(0.8 * len(yy)))]
    wrong = (yhat != yy).astype(int)
    return np.array([fauc(yy, pp), fba(yy, yhat), fba(yy[keep], yhat[keep]), fauc(wrong, hh)])
rng = np.random.default_rng(0); B = 2000
groups = {s: np.where(slides == s)[0] for s in us}
boot_idx = [np.concatenate([groups[s] for s in rng.choice(us, len(us))]) for _ in range(B)]
rows = []; allii = np.arange(len(y))
for setting in sorted({k for k, _ in P}):
    obs = np.mean([mets(allii, *P[setting, s]) for s in (0, 1, 2)], 0)
    sd = np.std([mets(allii, *P[setting, s]) for s in (0, 1, 2)], 0, ddof=1)
    bs = np.array([np.mean([mets(ii, *P[setting, s]) for s in (0, 1, 2)], 0) for ii in boot_idx[:400]])
    lo, hi = np.percentile(bs, [2.5, 97.5], 0)
    orc = np.mean([oracle_ba(y, P[setting, s][0]) for s in (0, 1, 2)])
    model, ad = setting.split("|")
    rows.append(dict(model=model, adapt=ad, auc=obs[0], auc_sd=sd[0], auc_lo=lo[0], auc_hi=hi[0], ba=obs[1], ba_sd=sd[1],
                     ba_lo=lo[1], ba_hi=hi[1], ba80=obs[2], ba80_lo=lo[2], ba80_hi=hi[2], err_auc=obs[3], oracle_ba=orc))
d = pd.DataFrame(rows); d.to_csv(ROOT / "results/strengthen/hiesd_main.csv", index=False)
pd.set_option("display.width", 250); print(d.round(3).to_string(index=False))
# paired slide-bootstrap tests for the key comparisons
def paired(a, b, j):
    diff = lambda ii: np.mean([mets(ii, *P[a, s])[j] - mets(ii, *P[b, s])[j] for s in (0, 1, 2)])
    obs = diff(allii); bs = np.array([diff(ii) for ii in boot_idx[:1000]])
    return obs, *np.percentile(bs, [2.5, 97.5]), max(2 * min((bs <= 0).mean(), (bs >= 0).mean()), 1 / len(bs))
T = [("AUC UNI - ResNet50", "uni|none", "resnet50|none", 0), ("AUC UNI - Phikon", "uni|none", "phikon|none", 0),
     ("BA UNI target-std - none", "uni|target_std", "uni|none", 1), ("BA@80 UNI target-std - none", "uni|target_std", "uni|none", 2),
     ("BA Phikon target-std - none", "phikon|target_std", "phikon|none", 1), ("BA ResNet50 AdaBN - none", "resnet50|AdaBN", "resnet50|none", 1),
     ("BA UNI target-std - ResNet50 AdaBN", "uni|target_std", "resnet50|AdaBN", 1)]
t = pd.DataFrame([dict(comparison=n, **dict(zip(["diff", "ci_lo", "ci_hi", "p"], paired(a, b, j)))) for n, a, b, j in T])
t.to_csv(ROOT / "results/strengthen/hiesd_significance.csv", index=False); print(t.round(4).to_string(index=False))
