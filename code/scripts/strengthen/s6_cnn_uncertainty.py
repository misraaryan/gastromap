"""Error-detection AUROC and referral metrics for ResNet50 on GasHisSDB (HMU -> GasHisSDB),
using MC-dropout entropy and mutual information, and the single-pass softmax entropy for comparison."""
import numpy as np, pandas as pd
from common import ROOT, selective_metrics
rows = []
for split in ("grouped", "random"):
    for s in (0, 1, 2):
        d = pd.read_csv(ROOT / f"runs/cnn_resnet50_hmu-{split}_hed_s{s}/pred_hmu-{split}_to_gashis_target_all.csv")
        y = d.label.values.astype(int)
        pdet = np.clip(d.prob_det.values, 1e-7, 1 - 1e-7)
        h_det = -(pdet * np.log(pdet) + (1 - pdet) * np.log(1 - pdet))
        r = dict(split=split, seed=s, max_mc_std=d["std"].max())
        r.update({f"softmax_entropy_{k}": v for k, v in selective_metrics(y, d.prob_det.values, h_det).items()})
        r.update({f"mc_entropy_{k}": v for k, v in selective_metrics(y, d.prob.values, d.entropy.values).items()})
        r.update({f"mc_mi_{k}": v for k, v in selective_metrics(y, d.prob.values, d.mutual_info.values).items()})
        rows.append(r)
t = pd.DataFrame(rows)
t.to_csv(ROOT / "results/strengthen/cnn_uncertainty.csv", index=False)
cols = ["max_mc_std", "softmax_entropy_err_detect_auc", "mc_entropy_err_detect_auc", "mc_mi_err_detect_auc",
        "softmax_entropy_balacc@80", "mc_entropy_balacc@80"]
print(t.groupby("split")[cols].agg(["mean", "std"]).round(3).T.to_string())
