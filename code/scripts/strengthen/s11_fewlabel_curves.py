"""Workload-accuracy curve for the tool's few-label mode: threshold set from 50 labelled tiles (25+25),
applied as a logit shift to every ensemble member (so uncertainty is centred on the new threshold).
Test sets of 2,000 tiles at 10-90% tumour. UNI, HMU-grouped -> GasHisSDB. Appended to per_slide_curves.npz."""
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from common import ROOT, fit_cache, target, transform, sigmoid
from gcx.uncertainty import summarise_samples
X, y, _ = target("uni", "hmu-grouped")
grid = np.round(np.arange(0.5, 1.0001, 0.05), 2); prevs = (0.1, 0.3, 0.5, 0.7, 0.9)
def logit(p): p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))
acc = []
for seed in (0, 1, 2):
    M = fit_cache("uni", "hmu-grouped", seed)
    L = transform(X, M, "none") @ M["W"].T + M["b"]                 # (N, members) logits
    p0 = sigmoid(L).mean(1)
    for r in range(10):
        rng = np.random.default_rng(11000 + 10 * seed + r)
        perm = rng.permutation(len(y)); cal, test = perm[:15000], perm[15000:]
        ki = np.concatenate([rng.choice(cal[y[cal] == 1], 25, replace=False), rng.choice(cal[y[cal] == 0], 25, replace=False)])
        c = np.unique(p0[ki]); s = [balanced_accuracy_score(y[ki], (p0[ki] >= t).astype(int)) for t in c]
        t = float(np.median(c[np.isclose(s, max(s))])); shift = logit(t)
        for prev in prevs:
            kk = int(prev * 2000)
            ti = np.concatenate([rng.choice(test[y[test] == 1], kk, replace=False), rng.choice(test[y[test] == 0], 2000 - kk, replace=False)])
            S = summarise_samples(sigmoid(L[ti] - shift).T); p, H = S["prob"], S["entropy"]
            order = np.argsort(H, kind="stable")
            acc.append([balanced_accuracy_score(y[ti][order[:int(round(g * 2000))]], (p[order[:int(round(g * 2000))]] >= 0.5).astype(int)) for g in grid])
acc_all = np.array(acc).reshape(3 * 10, len(prevs), len(grid))          # (draw, prev, grid)
p10 = np.percentile(acc_all.reshape(-1, len(grid)), 10, axis=0)           # 10th percentile over draws and prevalences
acc = acc_all.mean(0)
z = dict(np.load(ROOT / "code/tool/bundle/per_slide_curves.npz"))
z.update(fewlabel_mean=acc.mean(0), fewlabel_worst=acc.min(0), fewlabel_best=acc.max(0), fewlabel_p10=p10, fewlabel_k=50)
np.savez(ROOT / "code/tool/bundle/per_slide_curves.npz", **z)
print("few-label BA by prevalence (all kept / 80% kept):")
for i, pv in enumerate(prevs): print(pv, round(acc[i, -1], 3), round(acc[i, 6], 3))
print("mean curve:", np.round(acc.mean(0), 3)); print("10th percentile:", np.round(p10, 3))
