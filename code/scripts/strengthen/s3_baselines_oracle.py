"""Exp 3-4: label-free baselines on the full target set, plus an oracle threshold that uses target labels.
CORAL comes from the paper's existing runs (same probes, same seeds)."""
import numpy as np, pandas as pd
from common import ROOT, fit_cache, target, transform, predict, metrics, oracle_ba
rows = []
name = {"hmu-grouped": "hmu-grouped_to_gashis", "gashis-random": "gashis-random_to_hmu"}
for enc in ("uni", "phikon"):
    for direction in ("hmu-grouped", "gashis-random"):
        X, y, paths = target(enc, direction)
        for seed in (0, 1, 2):
            M = fit_cache(enc, direction, seed)
            for method in ("none", "centre", "target_std", "balanced_std", "coral"):
                if method == "coral":
                    d = pd.read_csv(ROOT / f"runs/probe_{enc}_{direction}_s{seed}_coral/pred_{name[direction]}_target_all.csv")
                    assert (d.path.values == paths).all()
                    p_det, p_ens, H = d.prob_det.values, d.prob.values, d.entropy.values
                else:
                    p_det, p_ens, H = predict(transform(X, M, method), M)
                m = metrics(y, p_det, p_ens, H)
                m.update(oracle_ens=oracle_ba(y, p_ens), enc=enc, direction=direction, seed=seed, method=method)
                rows.append(m)
d = pd.DataFrame(rows)
d.to_csv(ROOT / "results/strengthen/baselines_oracle.csv", index=False)
s = d.groupby(["enc", "direction", "method"])[["auc", "ba_ens", "ba80", "err_auc", "oracle_ens"]].agg(["mean", "std"])
pd.set_option("display.width", 250)
print(s.round(3).to_string())
