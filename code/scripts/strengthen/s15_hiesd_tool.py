"""Tool validation on HiESD against pathologist masks (UNI, HMU-grouped probes, 3 seeds).
Per slide: true tumour share of its labelled tiles vs predicted share, for (a) hospital calibration from the OTHER 13
slides (leave-one-slide-out), (b) per-slide standardisation, (c) no adaptation; plus per-slide balanced accuracy."""
import numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr
from common import ROOT, fit_cache, load, predict, transform
idx = pd.read_csv(ROOT / "data/index/hiesd.csv"); y = idx.label.values; sl = idx.slide.values
X = load(ROOT / "data/features/hiesd_uni.npz")["X"].astype(float)
rows = []
for seed in (0, 1, 2):
    M = fit_cache("uni", "hmu-grouped", seed)
    for s in np.unique(sl):
        te, cal = sl == s, sl != s
        modes = {"hospital calibration (other 13 slides)": (X[te] - X[cal].mean(0)) / (X[cal].std(0) + 1e-12),
                 "per-slide standardisation": transform(X[te], M, "target_std"),
                 "no adaptation": transform(X[te], M, "none")}
        for mode, Z in modes.items():
            _, p, _ = predict(Z, M); yh = (p >= .5).astype(int); yt = y[te]
            ba = 0.5 * ((yh[yt == 1] == 1).mean() + (yh[yt == 0] == 0).mean())
            rows.append(dict(seed=seed, slide=s, mode=mode, true=yt.mean(), pred=yh.mean(), ba=ba))
d = pd.DataFrame(rows).groupby(["mode", "slide"])[["true", "pred", "ba"]].mean().reset_index()
d.to_csv(ROOT / "results/strengthen/hiesd_tool_validation.csv", index=False)
out = []
for mode, g in d.groupby("mode"):
    out.append(dict(mode=mode, pearson_r=pearsonr(g.true, g.pred)[0], spearman=spearmanr(g.true, g.pred)[0],
                    MAE=np.abs(g.pred - g.true).mean(), mean_true=g.true.mean(), mean_pred=g.pred.mean(),
                    pred_range=f"{g.pred.min():.2f}-{g.pred.max():.2f}", median_slide_BA=g.ba.median()))
print(pd.DataFrame(out).round(3).to_string(index=False))
print(d[d["mode"].str.startswith("hospital")][["slide", "true", "pred", "ba"]].round(2).to_string(index=False))
