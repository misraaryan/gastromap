"""Fit/cache probes for one encoder+direction+seed and check they reproduce the paper's prediction files."""
import sys
import numpy as np, pandas as pd
from common import ROOT, fit_cache, target, transform, predict, metrics
enc, direction, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
M = fit_cache(enc, direction, seed)
X, y, paths = target(enc, direction)
name = {"hmu-grouped": "hmu-grouped_to_gashis", "gashis-random": "gashis-random_to_hmu"}[direction]
for method, suffix in (("none", ""), ("target_std", "_target_std")):
    pd_old = pd.read_csv(ROOT / f"runs/probe_{enc}_{direction}_s{seed}{suffix}/pred_{name}_target_all.csv")
    assert (pd_old.path.values == paths).all(), "path order differs"
    p_det, p_ens, H = predict(transform(X, M, method), M)
    d1 = np.abs(p_det - pd_old.prob_det.values).max(); d2 = np.abs(p_ens - pd_old.prob.values).max()
    m = metrics(y, p_det, p_ens, H)
    print(f"{enc} {direction} s{seed} {method:10s} C={float(M['C'])} max|dp_det|={d1:.1e} max|dp_ens|={d2:.1e} "
          + " ".join(f"{k}={v:.3f}" for k, v in m.items()))
