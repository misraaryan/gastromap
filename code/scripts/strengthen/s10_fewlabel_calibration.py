"""Alternative calibration: no feature standardisation (keeps each slide's tumour content); instead the decision
threshold is re-set from k pathologist-labelled tiles of the new hospital (k/2 tumour + k/2 normal), chosen to
maximise balanced accuracy on them. Tested on disjoint sets with 10-90% tumour. UNI, HMU-grouped -> GasHisSDB."""
import numpy as np, pandas as pd
from sklearn.metrics import balanced_accuracy_score
from common import ROOT, fit_cache, target, transform, predict
X, y, _ = target("uni", "hmu-grouped")
def best_thr(yk, pk):
    cands = np.unique(np.concatenate([pk, [0.5]]))
    s = [balanced_accuracy_score(yk, (pk >= t).astype(int)) for t in cands]
    best = np.max(s); ts = cands[np.isclose(s, best)]
    return float(np.median(ts))
rows = []
for seed in (0, 1, 2):
    M = fit_cache("uni", "hmu-grouped", seed)
    _, P, _ = predict(transform(X, M, "none"), M)
    for r in range(10):
        rng = np.random.default_rng(9000 + 100 * seed + r)
        perm = rng.permutation(len(y)); cal, test = perm[:15000], perm[15000:]
        cp, cn = cal[y[cal] == 1], cal[y[cal] == 0]; tp, tn = test[y[test] == 1], test[y[test] == 0]
        for k in (10, 20, 50, 100):
            ki = np.concatenate([rng.choice(cp, k // 2, replace=False), rng.choice(cn, k // 2, replace=False)])
            t = best_thr(y[ki], P[ki])
            for prev in (0.1, 0.3, 0.5, 0.7, 0.9):
                kk = int(prev * 2000)
                ti = np.concatenate([rng.choice(tp, kk, replace=False), rng.choice(tn, 2000 - kk, replace=False)])
                yhat = (P[ti] >= t).astype(int)
                rows.append(dict(seed=seed, rep=r, k=k, prev=prev, thr=t, ba=balanced_accuracy_score(y[ti], yhat),
                                 pred_frac=yhat.mean()))
d = pd.DataFrame(rows); d.to_csv(ROOT / "results/strengthen/fewlabel_calibration.csv", index=False)
pd.set_option("display.width", 200)
print("BA (mean over seeds x reps):"); print(d.groupby(["k", "prev"]).ba.mean().unstack().round(3).to_string())
print("BA worst 10% of draws:"); print(d.groupby(["k", "prev"]).ba.quantile(0.1).unstack().round(3).to_string())
print("predicted tumour share:"); print(d.groupby(["k", "prev"]).pred_frac.mean().unstack().round(3).to_string())
