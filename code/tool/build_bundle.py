"""Build GastroMap's model bundle from the validated experiment (UNI + linear-probe ensemble).

Contents of code/tool/bundle/gastromap_uni_bundle.npz:
  * scaler_mean/scaler_std ... source (HMU) feature standardisation used to TRAIN the probes
  * coef (10, D), intercept (10,) ... bootstrap ensemble of logistic regressions (C = 0.001,
    the value chosen in all 12 UNI runs), bootstrap over HMU slide-blocks
  * bank (N, D) float16 ...... L2-normalised HMU features for the "unfamiliar slide" check
  * knn_thr ................. 99th percentile of within-HMU nearest-neighbour cosine distance
  * ood_ref_hmu / ood_ref_gashis ... share of tiles beyond knn_thr on HMU (held-out) and on GasHisSDB
  * cov_grid, cov_balacc_mean/sd ... workload curve: balanced accuracy vs share of tiles kept, from
    the HMU -> GasHisSDB target-standardised UNI runs (3 seeds), i.e. measured at a new hospital

Run from the project root:  python code/tool/build_bundle.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

OUT = Path("code/tool/bundle"); OUT.mkdir(parents=True, exist_ok=True)
C, MEMBERS, SEED, K = 0.001, 10, 0, 10

z = np.load("data/features/hmu_uni.npz", allow_pickle=True)
X, y, groups = z["X"].astype(np.float64), z["y"].astype(int), z["group"].astype(str)
mu, sd = X.mean(0), X.std(0) + 1e-6
Xs = (X - mu) / sd

# ---- probe ensemble, bootstrap over slide-blocks (as in the experiments)
rng = np.random.default_rng(SEED)
ug = np.unique(groups)
members = {g: np.where(groups == g)[0] for g in ug}
coefs, ints = [], []
for k in range(MEMBERS):
    ix = np.concatenate([members[g] for g in rng.choice(ug, len(ug))])
    lr = LogisticRegression(C=C, max_iter=3000, class_weight="balanced", random_state=SEED + k).fit(Xs[ix], y[ix])
    coefs.append(lr.coef_[0]); ints.append(lr.intercept_[0])
coefs, ints = np.asarray(coefs), np.asarray(ints)
print("ensemble:", coefs.shape)

# ---- unfamiliar-slide reference: kNN cosine distance to the HMU bank
def l2n(A):
    return A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)


B = l2n(X).astype(np.float32)
# threshold from a held-out half of HMU blocks against the other half (avoids self-matches / same-slide matches)
half = set(rng.choice(ug, len(ug) // 2, replace=False))
mA = np.array([g in half for g in groups])


def knn_dist(Q, R, k=K, chunk=2048):
    out = []
    for i in range(0, len(Q), chunk):
        S = Q[i:i + chunk] @ R.T
        part = np.partition(-S, k - 1, axis=1)[:, :k]
        out.append(1 - (-part).mean(1))
    return np.concatenate(out)


d_in = knn_dist(B[~mA], B[mA])
thr = float(np.percentile(d_in, 99))
g = np.load("data/features/gashis_uni.npz", allow_pickle=True)
Gb = l2n(g["X"].astype(np.float64)).astype(np.float32)
sub = np.random.default_rng(1).choice(len(Gb), 5000, replace=False)
d_out = knn_dist(Gb[sub], B)
ood_hmu, ood_gas = float((d_in > thr).mean()), float((d_out > thr).mean())
print(f"kNN threshold {thr:.4f} | share beyond threshold: HMU held-out {ood_hmu:.3f}, GasHisSDB {ood_gas:.3f}")

# ---- workload curve measured at the new hospital (HMU -> GasHisSDB, target-std, 3 seeds)
grid = np.round(np.linspace(0.5, 1.0, 11), 3)
curves = []
for s in (0, 1, 2):
    d = pd.read_csv(f"runs/probe_uni_hmu-grouped_s{s}_target_std/pred_hmu-grouped_to_gashis_target_all.csv")
    order = np.argsort(d.entropy.to_numpy(), kind="stable")
    yy, pp = d.label.to_numpy(), d.prob.to_numpy()
    curves.append([balanced_accuracy_score(yy[order[:int(round(c * len(yy)))]],
                                           (pp[order[:int(round(c * len(yy)))]] >= 0.5).astype(int)) for c in grid])
curves = np.asarray(curves)
print("workload curve (kept -> bal. acc):", dict(zip(grid.tolist(), np.round(curves.mean(0), 3).tolist())))

np.savez_compressed(OUT / "gastromap_uni_bundle.npz", scaler_mean=mu.astype(np.float32), scaler_std=sd.astype(np.float32),
                    coef=coefs.astype(np.float32), intercept=ints.astype(np.float32),
                    bank=B.astype(np.float16), knn_k=K, knn_thr=thr, ood_ref_hmu=ood_hmu, ood_ref_gashis=ood_gas,
                    cov_grid=grid, cov_balacc_mean=curves.mean(0), cov_balacc_sd=curves.std(0, ddof=1))
(OUT / "bundle_info.json").write_text(json.dumps(dict(
    encoder="UNI (MahmoodLab/uni, frozen)", trained_on="HMU-GC-HE-30K, TUM vs NOR, all 7,774 patches",
    classifier=f"{MEMBERS} logistic regressions, C={C}, bootstrap over 60-tile slide-blocks",
    adaptation="per-slide target standardisation (validated: Table 3)", knn_threshold=thr,
    unfamiliar_share_reference={"HMU_heldout": ood_hmu, "GasHisSDB": ood_gas},
    workload_curve_source="HMU->GasHisSDB, target-standardised UNI probes, 3 seeds",
    note="Research prototype. Not a medical device."), indent=2))
print("saved", OUT / "gastromap_uni_bundle.npz")
