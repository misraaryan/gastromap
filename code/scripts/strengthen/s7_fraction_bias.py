"""Does per-set standardisation bias the predicted tumour fraction towards the training balance?
Predicted tumour share (ensemble mean >= 0.5) vs true share, UNI, HMU-grouped -> GasHisSDB subsamples."""
import numpy as np, pandas as pd
from common import ROOT, fit_cache, target, transform, predict
X, y, _ = target("uni", "hmu-grouped")
pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
rows = []
for seed in (0, 1, 2):
    M = fit_cache("uni", "hmu-grouped", seed)
    for prev in (0.1, 0.3, 0.5, 0.7, 0.9):
        for r in range(3):
            rng = np.random.default_rng(1000 * seed + r); k = int(prev * 2000)
            ix = np.concatenate([rng.choice(pos, k, replace=False), rng.choice(neg, 2000 - k, replace=False)])
            for m in ("none", "target_std", "balanced_std"):
                _, p, _ = predict(transform(X[ix], M, m), M)
                rows.append(dict(seed=seed, true=prev, method=m, predicted=(p >= 0.5).mean(), mean_prob=p.mean()))
d = pd.DataFrame(rows); d.to_csv(ROOT / "results/strengthen/fraction_bias.csv", index=False)
print(d.groupby(["method", "true"]).predicted.agg(["mean", "std"]).round(3).unstack("true").to_string())
