"""Classification, calibration and selective-prediction metrics."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             roc_auc_score, brier_score_loss)


def _safe_auc(y, p):
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


def expected_calibration_error(y, p, n_bins: int = 15) -> float:
    """ECE on the probability of the predicted class."""
    y, p = np.asarray(y), np.asarray(p)
    pred = (p >= 0.5).astype(int)
    conf = np.where(pred == 1, p, 1 - p)
    correct = (pred == y).astype(float)
    bins = np.linspace(0.5, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0.5 else (conf >= lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def classification_metrics(y, p, thr: float = 0.5) -> dict:
    y, p = np.asarray(y).astype(int), np.asarray(p, dtype=float)
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    return dict(
        n=len(y), auc=_safe_auc(y, p),
        acc=float(accuracy_score(y, pred)),
        bal_acc=float(balanced_accuracy_score(y, pred)),
        f1=float(f1_score(y, pred, zero_division=0)),
        sens=tp / (tp + fn) if tp + fn else float("nan"),
        spec=tn / (tn + fp) if tn + fp else float("nan"),
        ece=expected_calibration_error(y, p),
        brier=float(brier_score_loss(y, p)),
    )


def bootstrap_ci(y, p, metric: str = "auc", n: int = 1000, seed: int = 0, groups=None):
    """95% CI by bootstrap. If `groups` is given, resample whole groups (slides), which is
    the honest choice when patches from one slide are correlated."""
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    if metric == "auc":
        def score(yy, pp):
            return _safe_auc(yy, pp)
    else:
        def score(yy, pp):
            return classification_metrics(yy, pp)[metric]
    vals = []
    if groups is None:
        for _ in range(n):
            ix = rng.integers(0, len(y), len(y))
            vals.append(score(y[ix], p[ix]))
    else:
        groups = np.asarray(groups)
        uniq = np.unique(groups)
        members = {g: np.where(groups == g)[0] for g in uniq}
        for _ in range(n):
            ix = np.concatenate([members[g] for g in rng.choice(uniq, len(uniq))])
            vals.append(score(y[ix], p[ix]))
    vals = np.asarray(vals, dtype=float)
    vals = vals[~np.isnan(vals)]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# --------------------------------------------------------------------------- selective prediction
def risk_coverage(y, p, uncertainty):
    """Sort by increasing uncertainty; return coverage and error rate of the kept patches."""
    y, p, u = map(np.asarray, (y, p, uncertainty))
    order = np.argsort(u, kind="stable")
    err = ((p[order] >= 0.5).astype(int) != y[order]).astype(float)
    k = np.arange(1, len(y) + 1)
    return k / len(y), np.cumsum(err) / k


def aurc(y, p, uncertainty) -> float:
    """Area under the risk-coverage curve (lower is better)."""
    cov, risk = risk_coverage(y, p, uncertainty)
    return float(np.trapezoid(risk, cov)) if hasattr(np, "trapezoid") else float(np.trapz(risk, cov))


def selective_metrics(y, p, uncertainty, coverages=(1.0, 0.9, 0.8, 0.7)) -> dict:
    """Accuracy when the most uncertain (1-coverage) share of patches is referred to a pathologist,
    plus how well uncertainty separates wrong from right predictions (AUROC)."""
    y, p, u = map(np.asarray, (y, p, uncertainty))
    wrong = ((p >= 0.5).astype(int) != y).astype(int)
    out = {"aurc": aurc(y, p, u),
           "err_detect_auc": _safe_auc(wrong, u)}
    order = np.argsort(u, kind="stable")
    for c in coverages:
        keep = order[: max(1, int(round(c * len(y))))]
        out[f"acc@{int(c*100)}"] = float(((p[keep] >= 0.5).astype(int) == y[keep]).mean())
        out[f"balacc@{int(c*100)}"] = float(balanced_accuracy_score(y[keep], (p[keep] >= 0.5).astype(int))) \
            if len(np.unique(y[keep])) == 2 else float("nan")
    return out
