"""Exp 1: does label-free target standardisation survive a change in tumour prevalence of the target set?
Fixed-size subsamples of the target set at 10-90% tumour (plus its natural prevalence), 5 repeats per seed."""
import sys
import numpy as np, pandas as pd
from common import ROOT, fit_cache, target, transform, predict, metrics, oracle_ba
enc, direction = sys.argv[1], sys.argv[2]
N = {"hmu-grouped": 4000, "gashis-random": 2000}[direction]
X, y, _ = target(enc, direction)
pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
rows = []
for seed in (0, 1, 2):
    M = fit_cache(enc, direction, seed)
    for prev in (0.1, 0.3, 0.5, 0.7, 0.9, "natural"):
        for r in range(5):
            rng = np.random.default_rng(1000 * seed + r)
            if prev == "natural":
                ix = rng.choice(len(y), N, replace=False)
            else:
                k = int(round(prev * N))
                ix = np.concatenate([rng.choice(pos, k, replace=False), rng.choice(neg, N - k, replace=False)])
            Xs, ys = X[ix], y[ix]
            for method in ("none", "centre", "target_std", "balanced_std"):
                p_det, p_ens, H = predict(transform(Xs, M, method), M)
                m = metrics(ys, p_det, p_ens, H)
                m.update(oracle_ens=oracle_ba(ys, p_ens), enc=enc, direction=direction, seed=seed,
                         prevalence=prev if prev == "natural" else float(prev), true_prev=ys.mean(), rep=r, method=method)
                rows.append(m)
out = ROOT / f"results/strengthen/prevalence_{enc}_{direction}.csv"
pd.DataFrame(rows).to_csv(out, index=False)
print("saved", out, len(rows))
