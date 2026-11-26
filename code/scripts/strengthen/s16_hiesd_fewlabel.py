"""Few-label calibration on HiESD (leave-one-slide-out): threshold from 25 TUM + 25 NOR labelled tiles drawn from the
OTHER 13 slides (10 random draws x 3 seeds), applied to the unadapted UNI ensemble on the held-out slide."""
import numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr
from common import ROOT, fit_cache, load, predict, transform
idx = pd.read_csv(ROOT / "data/index/hiesd.csv"); y = idx.label.values; sl = idx.slide.values
X = load(ROOT / "data/features/hiesd_uni.npz")["X"].astype(float)
rows = []
for seed in (0, 1, 2):
    M = fit_cache("uni", "hmu-grouped", seed); _, p0, _ = predict(transform(X, M, "none"), M)
    for s in np.unique(sl):
        te = sl == s; cal = np.where(sl != s)[0]
        for r in range(10):
            rng = np.random.default_rng(100 * seed + r)
            ki = np.concatenate([rng.choice(cal[y[cal] == 1], 25, replace=False), rng.choice(cal[y[cal] == 0], 25, replace=False)])
            c = np.unique(p0[ki]); ba = [0.5 * ((p0[ki][y[ki] == 1] >= t).mean() + (p0[ki][y[ki] == 0] < t).mean()) for t in c]
            t = float(np.median(c[np.isclose(ba, max(ba))]))
            yh = (p0[te] >= t).astype(int); yt = y[te]
            rows.append(dict(seed=seed, rep=r, slide=s, true=yt.mean(), pred=yh.mean(),
                             ba=0.5 * ((yh[yt == 1] == 1).mean() + (yh[yt == 0] == 0).mean())))
d = pd.DataFrame(rows).groupby("slide")[["true", "pred", "ba"]].mean().reset_index()
d.to_csv(ROOT / "results/strengthen/hiesd_fewlabel_validation.csv", index=False)
print(f"few-label (50 tiles): pearson {pearsonr(d.true, d.pred)[0]:.3f}  spearman {spearmanr(d.true, d.pred)[0]:.3f}  "
      f"MAE {np.abs(d.pred - d.true).mean():.3f}  mean pred {d.pred.mean():.3f} (true {d.true.mean():.3f})  median slide BA {d.ba.median():.3f}")
