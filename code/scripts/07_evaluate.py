"""Turn prediction CSVs into the paper's result tables.

For each pred_*.csv: classification metrics (+ slide-level bootstrap 95% CI for AUC when
groups exist), calibration, and selective-prediction metrics for each uncertainty score.
Then:
  * leakage gap      = metric(random split) - metric(grouped split)      [contribution 1]
  * generalisation gap = metric(in-domain test) - metric(cross-dataset) [contribution 2]
  * uncertainty table = AURC, error-detection AUROC, accuracy at 90/80/70% coverage [contribution 3]

Usage:
    python scripts/07_evaluate.py --preds "runs/*/pred_*.csv" --out results/
"""
import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gcx.metrics import classification_metrics, bootstrap_ci, selective_metrics  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="glob of prediction CSVs")
    ap.add_argument("--out", default="results")
    ap.add_argument("--n_boot", type=int, default=500)
    ap.add_argument("--point", default="prob", choices=["prob", "prob_det"],
                    help="point prediction: ensemble/MC mean (prob, default since 23 Nov 2026) or single model (prob_det)")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    rows, unc_rows = [], []
    for f in sorted(glob.glob(a.preds)):
        d = pd.read_csv(f)
        run, name = Path(f).parent.name, Path(f).stem.replace("pred_", "")
        p = d[a.point].to_numpy(); y = d["label"].to_numpy().astype(int)
        m = classification_metrics(y, p)
        g = None
        if "group" in d:
            gs = d["group"].astype(object)
            if not (gs.isna() | gs.astype(str).isin(["nan", "None", ""])).any():
                g = gs.astype(str).to_numpy()
        lo, hi = bootstrap_ci(y, p, "auc", a.n_boot, groups=g)
        rows.append(dict(run=run, eval=name, **m, auc_lo=lo, auc_hi=hi,
                         ci_unit="slide" if g is not None else "patch"))
        for u in ("entropy", "mutual_info", "std"):
            if u in d and d[u].std() > 0:
                unc_rows.append(dict(run=run, eval=name, score=u,
                                     **selective_metrics(y, d["prob"].to_numpy(), d[u].to_numpy())))

    res = pd.DataFrame(rows)
    res.to_csv(out / "metrics_all.csv", index=False)
    if unc_rows:
        pd.DataFrame(unc_rows).to_csv(out / "uncertainty_all.csv", index=False)
    print(res[["run", "eval", "n", "auc", "auc_lo", "auc_hi", "bal_acc", "ece"]].round(4).to_string(index=False))
    if unc_rows:
        print("\nUncertainty / selective prediction:")
        print(pd.DataFrame(unc_rows).round(4).to_string(index=False))
    print(f"\nWrote {out/'metrics_all.csv'}" + (f" and {out/'uncertainty_all.csv'}" if unc_rows else ""))
    print("Gaps are computed in the paper tables by pairing rows "
          "(random vs grouped run; in-domain vs cross-dataset eval) - see README section 'Result tables'.")


if __name__ == "__main__":
    main()
