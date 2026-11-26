"""Exp 5: bootstrap confidence intervals and p-values for the key comparisons.
Statistic = mean over the 3 seeds of the per-seed difference. Each bootstrap replicate resamples
test units (slide-blocks for HMU test sets; patches for GasHisSDB, which has no slide IDs, so those
intervals are optimistic). Two-sided p = 2 * min(P(diff<=0), P(diff>=0)), floored at 1/B."""
import sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from common import ROOT
B = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
which = sys.argv[1]
rng = np.random.default_rng(0)
R = lambda p: pd.read_csv(ROOT / p)
def ba(y, p): return balanced_accuracy_score(y, (p >= 0.5).astype(int))
def ba80(y, p, u):
    k = int(round(0.8 * len(y))); keep = np.argsort(u, kind="stable")[:k]
    return balanced_accuracy_score(y[keep], (p[keep] >= 0.5).astype(int))
def summarise(name, obs, boots):
    boots = np.asarray(boots); lo, hi = np.percentile(boots, [2.5, 97.5])
    p = max(2 * min((boots <= 0).mean(), (boots >= 0).mean()), 1 / len(boots))
    return dict(comparison=name, diff=obs, ci_lo=lo, ci_hi=hi, p=p, B=len(boots))
rows = []
if which == "leakage":   # in-domain AUC: random split minus grouped split, unpaired, block bootstrap
    runs = {"ResNet50": "runs/cnn_resnet50_hmu-{m}_hed_s{s}/pred_hmu-{m}_to_gashis_source_test.csv",
            "Phikon": "runs/probe_phikon_hmu-{m}_s{s}/pred_hmu-{m}_to_gashis_source_test.csv",
            "UNI": "runs/probe_uni_hmu-{m}_s{s}/pred_hmu-{m}_to_gashis_source_test.csv"}
    only = sys.argv[3] if len(sys.argv) > 3 else None
    for model, pat in runs.items():
        if only and model != only:
            continue
        D = {(m, s): R(pat.format(m=m, s=s)) for m in ("grouped", "random") for s in (0, 1, 2)}
        obs = np.mean([roc_auc_score(D["random", s].label, D["random", s].prob) -
                       roc_auc_score(D["grouped", s].label, D["grouped", s].prob) for s in (0, 1, 2)])
        idx = {k: {g: np.where(d.group.values == g)[0] for g in np.unique(d.group.values)} for k, d in D.items()}
        boots = []
        for _ in range(B):
            diffs = []
            for s in (0, 1, 2):
                a = {}
                for m in ("grouped", "random"):
                    d = D[m, s]; gs = list(idx[m, s]); pick = rng.choice(len(gs), len(gs))
                    ii = np.concatenate([idx[m, s][gs[j]] for j in pick])
                    a[m] = roc_auc_score(d.label.values[ii], d.prob.values[ii])
                diffs.append(a["random"] - a["grouped"])
            boots.append(np.mean(diffs))
        rows.append(summarise(f"leakage gap (AUC random - grouped), {model}", obs, boots))
else:                    # paired comparisons on the GasHisSDB target (HMU grouped -> GasHisSDB)
    T = lambda run, s: R(f"runs/{run.format(s=s)}/pred_hmu-grouped_to_gashis_target_all.csv")
    comps = {
        "cross AUC, UNI - ResNet50": ("probe_uni_hmu-grouped_s{s}", "cnn_resnet50_hmu-grouped_hed_s{s}", "auc"),
        "cross AUC, UNI - Phikon": ("probe_uni_hmu-grouped_s{s}", "probe_phikon_hmu-grouped_s{s}", "auc"),
        "cross AUC, Phikon - ResNet50": ("probe_phikon_hmu-grouped_s{s}", "cnn_resnet50_hmu-grouped_hed_s{s}", "auc"),
        "BA, UNI target-std - none": ("probe_uni_hmu-grouped_s{s}_target_std", "probe_uni_hmu-grouped_s{s}", "ba"),
        "BA, Phikon target-std - none": ("probe_phikon_hmu-grouped_s{s}_target_std", "probe_phikon_hmu-grouped_s{s}", "ba"),
        "BA@80%, UNI target-std - none": ("probe_uni_hmu-grouped_s{s}_target_std", "probe_uni_hmu-grouped_s{s}", "ba80"),
        "BA@80%, Phikon target-std - none": ("probe_phikon_hmu-grouped_s{s}_target_std", "probe_phikon_hmu-grouped_s{s}", "ba80"),
        "BA, UNI target-std - CORAL": ("probe_uni_hmu-grouped_s{s}_target_std", "probe_uni_hmu-grouped_s{s}_coral", "ba"),
        "BA, UNI none: 80% kept - all": ("probe_uni_hmu-grouped_s{s}", "probe_uni_hmu-grouped_s{s}", "ref"),
    }
    for name, (ra, rb, met) in comps.items():
        A = [T(ra, s) for s in (0, 1, 2)]; Bd = [T(rb, s) for s in (0, 1, 2)]
        y = A[0].label.values.astype(int); n = len(y)
        assert all((a.path.values == b.path.values).all() for a, b in zip(A, Bd))
        def stat(ii):
            out = []
            for a, b in zip(A, Bd):
                if met == "auc":
                    out.append(roc_auc_score(y[ii], a.prob.values[ii]) - roc_auc_score(y[ii], b.prob.values[ii]))
                elif met == "ba":
                    out.append(ba(y[ii], a.prob.values[ii]) - ba(y[ii], b.prob.values[ii]))
                elif met == "ba80":
                    out.append(ba80(y[ii], a.prob.values[ii], a.entropy.values[ii]) - ba80(y[ii], b.prob.values[ii], b.entropy.values[ii]))
                else:
                    out.append(ba80(y[ii], a.prob.values[ii], a.entropy.values[ii]) - ba(y[ii], a.prob.values[ii]))
            return np.mean(out)
        obs = stat(np.arange(n))
        boots = [stat(rng.integers(0, n, n)) for _ in range(B)]
        rows.append(summarise(name, obs, boots))
out = ROOT / f"results/strengthen/significance_{which}{('_' + sys.argv[3]) if len(sys.argv) > 3 else ''}.csv"
pd.DataFrame(rows).to_csv(out, index=False)
print(pd.DataFrame(rows).round(4).to_string())
