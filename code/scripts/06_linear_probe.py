"""Linear probe on frozen features: logistic regression trained on the SOURCE train split,
C chosen on SOURCE val, then applied to the source test split and the whole TARGET set.
Uncertainty comes from a bootstrap ensemble of probes (members trained on resampled
training groups), giving the same prob/entropy/mutual_info columns as the CNN path.

Usage:
    python scripts/06_linear_probe.py --source features/hmu_phikon.npz \
        --target features/gashis_phikon.npz --name hmu2gashis_phikon --out runs/probe_hmu2gashis_phikon
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gcx.metrics import classification_metrics  # noqa: E402
from gcx.uncertainty import summarise_samples  # noqa: E402


def load(f):
    z = np.load(f, allow_pickle=True)
    return {k: z[k] for k in z.files}


def fit_lr(X, y, C, seed):
    return LogisticRegression(C=C, max_iter=3000, class_weight="balanced", random_state=seed).fit(X, y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--members", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--source_splits", default=None,
                    help="split CSV for the source (e.g. data/index/hmu_grouped_s0.csv). Features are "
                         "extracted once per dataset; the split is taken from this CSV by image path. "
                         "Images marked 'gap' or absent from the CSV are dropped.")
    ap.add_argument("--adapt", default="none", choices=["none", "target_std", "coral"],
                    help="label-free adaptation of TARGET features: 'target_std' standardises the target with "
                         "its own mean/std; 'coral' also aligns its covariance to the source (Sun et al. 2016). "
                         "Uses target images only, never target labels. Source results are unaffected.")
    ap.add_argument("--target_splits", default=None,
                    help="optional CSV restricting the target to its non-gap images")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    S, Tg = load(a.source), load(a.target)

    def restrict(D, csv, assign_split):
        sp = pd.read_csv(csv)
        sp = sp[sp.split != "gap"].set_index("path")["split"] if "split" in sp else \
            pd.Series("all", index=sp.path)
        keep = np.array([p in sp.index for p in D["path"]])
        D = {k: (v[keep] if hasattr(v, "__len__") and len(v) == len(keep) else v) for k, v in D.items()}
        if assign_split:
            D["split"] = np.array([sp[p] for p in D["path"]])
        print(f"{csv}: kept {keep.sum()} of {len(keep)} images")
        return D

    if a.source_splits:
        S = restrict(S, a.source_splits, True)
    if a.target_splits:
        Tg = restrict(Tg, a.target_splits, False)
    for need in ("train", "val", "test"):
        if not (S["split"] == need).any():
            raise SystemExit(f"source has no '{need}' images: pass --source_splits <split CSV>")

    tr, va, te = (S["split"] == s for s in ("train", "val", "test"))
    sc = StandardScaler().fit(S["X"][tr])
    Xs = sc.transform(S["X"])
    if a.adapt == "none":
        Xt = sc.transform(Tg["X"])
    else:
        Xt = StandardScaler().fit_transform(Tg["X"])          # target in its own standard units
        if a.adapt == "coral":
            def msqrt(C, inv=False):
                w, V = np.linalg.eigh(C)
                w = np.clip(w, 1e-6, None)
                return (V * (w ** (-0.5 if inv else 0.5))) @ V.T
            Cs = np.cov(Xs[tr], rowvar=False) + np.eye(Xs.shape[1])
            Ct = np.cov(Xt, rowvar=False) + np.eye(Xt.shape[1])
            Xt = Xt @ msqrt(Ct, inv=True) @ msqrt(Cs)
            Xt = Xt + Xs[tr].mean(0)                           # source train mean is ~0 after scaling
    print("target adaptation:", a.adapt)

    best_C, best = None, -1
    for C in (0.001, 0.01, 0.1, 1.0, 10.0):
        m = classification_metrics(S["y"][va], fit_lr(Xs[tr], S["y"][tr], C, a.seed).predict_proba(Xs[va])[:, 1])
        score = m["auc"] if not np.isnan(m["auc"]) else m["bal_acc"]
        if score > best:
            best, best_C = score, C
    print("chosen C", best_C, "val score", round(best, 4))

    # bootstrap ensemble over training groups (slides) when available, else over patches
    rng = np.random.default_rng(a.seed)
    tr_idx = np.where(tr)[0]
    gser = pd.Series(S["group"][tr_idx], dtype=object)
    missing = gser.isna() | gser.astype(str).isin(["nan", "None", ""])
    use_groups = not missing.any()          # group bootstrap only when every image has a real group
    groups = gser.astype(str).to_numpy()
    P_src, P_tgt = [], []
    for k in range(a.members):
        if use_groups:
            ug = np.unique(groups)
            pick = rng.choice(ug, len(ug))
            ix = np.concatenate([tr_idx[groups == g] for g in pick])
        else:
            ix = rng.choice(tr_idx, len(tr_idx))
        if len(ix) == 0 or len(np.unique(S["y"][ix])) < 2:
            continue
        lr = fit_lr(Xs[ix], S["y"][ix], best_C, a.seed + k)
        P_src.append(lr.predict_proba(Xs[te])[:, 1]); P_tgt.append(lr.predict_proba(Xt)[:, 1])
    if not P_src:
        raise SystemExit("no ensemble member could be trained - check labels/groups")
    print(f"ensemble: {len(P_src)} members, bootstrap over {'slide groups' if use_groups else 'patches'}")
    det = fit_lr(Xs[tr], S["y"][tr], best_C, a.seed)

    for tag, X, P, mask_df in (("source_test", Xs[te], P_src, (S, te)), ("target_all", Xt, P_tgt, (Tg, None))):
        D, m = mask_df
        sel = m if m is not None else slice(None)
        s = summarise_samples(np.stack(P))
        res = pd.DataFrame(dict(path=D["path"][sel], label=D["y"][sel], group=D["group"][sel],
                                prob_det=det.predict_proba(X)[:, 1], **{k: s[k] for k in
                                ("prob", "entropy", "mutual_info", "std")}))
        res.to_csv(out / f"pred_{a.name}_{tag}.csv", index=False)
        print(tag, {k: round(v, 4) for k, v in classification_metrics(res.label, res.prob_det).items()
                    if k in ("auc", "bal_acc", "ece")})
    (out / "probe_config.json").write_text(json.dumps(dict(vars(a), C=best_C), indent=2))


if __name__ == "__main__":
    main()
