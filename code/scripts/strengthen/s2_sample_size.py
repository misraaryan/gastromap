"""Exp 2: how many unlabelled target images are needed to estimate the standardisation statistics?
Mean/SD are estimated from n random target images, then applied to the WHOLE target set, which is
where metrics are computed (separates estimation noise from evaluation noise). 20 repeats per n and seed."""
import sys
import numpy as np, pandas as pd
from common import ROOT, fit_cache, target, predict, metrics
enc, direction = sys.argv[1], sys.argv[2]
X, y, _ = target(enc, direction)
rows = []
for seed in (0, 1, 2):
    M = fit_cache(enc, direction, seed)
    for n in (10, 25, 50, 100, 200, 500, 1000, 5000, len(y)):
        for r in range(1 if n == len(y) else 20):
            rng = np.random.default_rng(1000 * seed + r)
            ix = rng.choice(len(y), n, replace=False)
            mu, sd = X[ix].mean(0), X[ix].std(0) + 1e-12
            p_det, p_ens, H = predict((X - mu) / sd, M)
            m = metrics(y, p_det, p_ens, H)
            m.update(enc=enc, direction=direction, seed=seed, n=n, rep=r)
            rows.append(m)
out = ROOT / f"results/strengthen/sample_size_{enc}_{direction}.csv"
pd.DataFrame(rows).to_csv(out, index=False)
print("saved", out, len(rows))
