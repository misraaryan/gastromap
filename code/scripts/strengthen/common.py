"""Shared helpers for the strengthening experiments (prevalence, sample size, oracle, baselines).
Probe fitting copies scripts/06_linear_probe.py exactly, so results match the paper's runs."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "code" / "src"))
from gcx.metrics import classification_metrics, selective_metrics  # noqa: E402
from gcx.uncertainty import summarise_samples  # noqa: E402

DIRS = {  # direction -> (source features, source split csv pattern, target features)
    "hmu-grouped": ("hmu", "data/index/hmu_grouped_s{s}.csv", "gashis"),
    "gashis-random": ("gashis", "data/index/gashissdb_random_s{s}.csv", "hmu"),
}
CACHE = ROOT / "runs" / "strengthen"


def load(f):
    z = np.load(f, allow_pickle=True)
    return {k: z[k] for k in z.files}


def fit_lr(X, y, C, seed):
    return LogisticRegression(C=C, max_iter=3000, class_weight="balanced", random_state=seed).fit(X, y)


def restrict(D, csv):
    sp = pd.read_csv(csv)
    sp = sp[sp.split != "gap"].set_index("path")["split"]
    keep = np.array([p in sp.index for p in D["path"]])
    D = {k: (v[keep] if hasattr(v, "__len__") and len(v) == len(keep) else v) for k, v in D.items()}
    D["split"] = np.array([sp[p] for p in D["path"]])
    return D


def fit_cache(enc, direction, seed, members=10):
    """Fit scaler, C, deterministic probe and bootstrap ensemble exactly as 06_linear_probe.py; cache to npz."""
    out = CACHE / f"{enc}_{direction}_s{seed}.npz"
    if out.exists():
        return dict(np.load(out, allow_pickle=True))
    src, csv, _ = DIRS[direction]
    S = restrict(load(ROOT / f"data/features/{src}_{enc}.npz"), ROOT / csv.format(s=seed))
    tr, va = S["split"] == "train", S["split"] == "val"
    sc = StandardScaler().fit(S["X"][tr])
    Xs = sc.transform(S["X"])
    best_C, best = None, -1
    for C in (0.001, 0.01, 0.1, 1.0, 10.0):
        m = classification_metrics(S["y"][va], fit_lr(Xs[tr], S["y"][tr], C, seed).predict_proba(Xs[va])[:, 1])
        score = m["auc"] if not np.isnan(m["auc"]) else m["bal_acc"]
        if score > best:
            best, best_C = score, C
    rng = np.random.default_rng(seed)
    tr_idx = np.where(tr)[0]
    gser = pd.Series(S["group"][tr_idx], dtype=object)
    missing = gser.isna() | gser.astype(str).isin(["nan", "None", ""])
    use_groups = not missing.any()
    groups = gser.astype(str).to_numpy()
    W, b = [], []
    for k in range(members):
        if use_groups:
            ug = np.unique(groups)
            pick = rng.choice(ug, len(ug))
            ix = np.concatenate([tr_idx[groups == g] for g in pick])
        else:
            ix = rng.choice(tr_idx, len(tr_idx))
        if len(ix) == 0 or len(np.unique(S["y"][ix])) < 2:
            continue
        lr = fit_lr(Xs[ix], S["y"][ix], best_C, seed + k)
        W.append(lr.coef_[0]); b.append(lr.intercept_[0])
    det = fit_lr(Xs[tr], S["y"][tr], best_C, seed)
    res = dict(mu=sc.mean_, sd=sc.scale_, C=best_C, W=np.array(W), b=np.array(b),
               w_det=det.coef_[0], b_det=det.intercept_[0], use_groups=use_groups)
    np.savez(out, **res)
    return res


def target(enc, direction):
    _, _, tgt = DIRS[direction]
    T = load(ROOT / f"data/features/{tgt}_{enc}.npz")
    return T["X"].astype(np.float64), T["y"].astype(int), T["path"]


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def transform(X, M, method, ref=None):
    """Map target features into the probe's input space.
    none: source scaler. target_std: target's own mean/sd. centre: target mean, source sd.
    balanced_std: mean/sd of a class-balanced target, using pseudo-labels from target_std (2 rounds)."""
    if method == "none":
        return (X - M["mu"]) / M["sd"]
    if method == "target_std":
        return (X - X.mean(0)) / (X.std(0) + 1e-12)
    if method == "centre":
        return (X - X.mean(0)) / M["sd"]
    if method == "balanced_std":
        Z = (X - X.mean(0)) / (X.std(0) + 1e-12)
        for _ in range(2):
            yhat = sigmoid(Z @ M["w_det"] + M["b_det"]) >= 0.5
            if yhat.all() or (~yhat).all():
                break
            m1, m0 = X[yhat].mean(0), X[~yhat].mean(0)
            v1, v0 = X[yhat].var(0), X[~yhat].var(0)
            mu = 0.5 * (m1 + m0)
            var = 0.5 * (v1 + v0) + 0.25 * (m1 - m0) ** 2
            Z = (X - mu) / (np.sqrt(var) + 1e-12)
        return Z
    raise ValueError(method)


def predict(Z, M):
    p_det = sigmoid(Z @ M["w_det"] + M["b_det"])
    P = sigmoid(Z @ M["W"].T + M["b"]).T          # (members, N)
    s = summarise_samples(P)
    return p_det, s["prob"], s["entropy"]


def metrics(y, p_det, p_ens, H):
    sel = selective_metrics(y, p_ens, H)
    return dict(auc=roc_auc_score(y, p_ens),
                ba_det=balanced_accuracy_score(y, (p_det >= 0.5).astype(int)),
                ba_ens=balanced_accuracy_score(y, (p_ens >= 0.5).astype(int)),
                ba80=sel["balacc@80"], err_auc=sel["err_detect_auc"])


def oracle_ba(y, p):
    """Best balanced accuracy over all thresholds (uses target labels: an upper bound)."""
    order = np.argsort(-p); ys = y[order]
    tp = np.cumsum(ys); fp = np.cumsum(1 - ys)
    P, N = ys.sum(), len(ys) - ys.sum()
    ba = 0.5 * (tp / P + (N - fp) / N)
    return float(max(ba.max(), 0.5))
