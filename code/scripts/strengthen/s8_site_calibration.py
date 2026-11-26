"""Simulate hospital-level calibration: standardisation statistics come from a separate calibration pool of
the target site (n_cal images, prevalence pi_cal), and are applied to disjoint test sets with 10-90% tumour.
Compares with per-set (per-slide) standardisation. UNI, HMU-grouped -> GasHisSDB, 3 seeds x 3 repeats."""
import numpy as np, pandas as pd
from sklearn.metrics import balanced_accuracy_score
from common import ROOT, fit_cache, target, transform, predict
X, y, _ = target("uni", "hmu-grouped")
rows = []
for seed in (0, 1, 2):
    M = fit_cache("uni", "hmu-grouped", seed)
    for r in range(3):
        rng = np.random.default_rng(5000 + 100 * seed + r)
        perm = rng.permutation(len(y)); cal_pool, test_pool = perm[:15000], perm[15000:]
        cp, cn = cal_pool[y[cal_pool] == 1], cal_pool[y[cal_pool] == 0]
        tp, tn = test_pool[y[test_pool] == 1], test_pool[y[test_pool] == 0]
        for n_cal in (500, 2000):
            for pi_cal in (0.2, 0.4, 0.6):
                k = int(pi_cal * n_cal)
                ci = np.concatenate([rng.choice(cp, k, replace=False), rng.choice(cn, n_cal - k, replace=False)])
                mu, sd = X[ci].mean(0), X[ci].std(0) + 1e-12
                for prev in (0.1, 0.3, 0.5, 0.7, 0.9):
                    kk = int(prev * 2000)
                    ti = np.concatenate([rng.choice(tp, kk, replace=False), rng.choice(tn, 2000 - kk, replace=False)])
                    for mode in ("site", "per_slide"):
                        Z = (X[ti] - mu) / sd if mode == "site" else transform(X[ti], M, "target_std")
                        _, p, _ = predict(Z, M)
                        rows.append(dict(seed=seed, rep=r, n_cal=n_cal, pi_cal=pi_cal, prev=prev, mode=mode,
                                         ba=balanced_accuracy_score(y[ti], (p >= 0.5).astype(int)),
                                         pred_frac=(p >= 0.5).mean()))
d = pd.DataFrame(rows); d.to_csv(ROOT / "results/strengthen/site_calibration.csv", index=False)
pd.set_option("display.width", 220)
g = d.groupby(["mode", "n_cal", "pi_cal", "prev"])[["ba", "pred_frac"]].mean().unstack("prev").round(3)
print(g.to_string())
