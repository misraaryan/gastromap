"""Workload-accuracy curves for the tool's per-slide mode: balanced accuracy on the kept tiles vs share kept,
when each set (a 'slide' of 2,000 tiles) is standardised with its own statistics, at 10-90% tumour.
UNI, HMU-grouped -> GasHisSDB, 3 seeds x 3 repeats. Saved into the tool bundle folder."""
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from common import ROOT, fit_cache, target, transform, predict
X, y, _ = target("uni", "hmu-grouped")
pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
grid = np.round(np.arange(0.5, 1.0001, 0.05), 2)
prevs = (0.1, 0.3, 0.5, 0.7, 0.9)
acc = np.zeros((len(prevs), 9, len(grid)))
for a, prev in enumerate(prevs):
    n = 0
    for seed in (0, 1, 2):
        M = fit_cache("uni", "hmu-grouped", seed)
        for r in range(3):
            rng = np.random.default_rng(7000 + 10 * seed + r); k = int(prev * 2000)
            ix = np.concatenate([rng.choice(pos, k, replace=False), rng.choice(neg, 2000 - k, replace=False)])
            _, p, H = predict(transform(X[ix], M, "target_std"), M)
            order = np.argsort(H, kind="stable")
            for g, c in enumerate(grid):
                keep = order[:int(round(c * len(ix)))]
                acc[a, n, g] = balanced_accuracy_score(y[ix][keep], (p[keep] >= 0.5).astype(int))
            n += 1
mean = acc.mean(1)
np.savez(ROOT / "code/tool/bundle/per_slide_curves.npz", cov_grid=grid, prevalences=np.array(prevs),
         balacc_mean=mean, balacc_worst=mean.min(0), balacc_best=mean.max(0))
for a, prev in enumerate(prevs):
    print(f"prev {prev}: kept 100% {mean[a, -1]:.3f}  80% {mean[a, 6]:.3f}  50% {mean[a, 0]:.3f}")
print("worst over prevalence:", np.round(mean.min(0), 3))
