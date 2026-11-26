"""Predictive uncertainty: MC dropout for CNNs, ensembles for anything."""
from __future__ import annotations

import numpy as np


def enable_mc_dropout(model):
    """Eval mode everywhere (BatchNorm frozen) but Dropout layers left active."""
    import torch.nn as nn
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout) or m.__class__.__name__.startswith("Dropout"):
            m.train()
    return model


def summarise_samples(probs: np.ndarray) -> dict:
    """probs: (T, N) tumour probabilities from T stochastic passes or T ensemble members.

    Returns mean probability, predictive entropy (total uncertainty), mutual information
    (epistemic part) and std across passes."""
    probs = np.clip(np.asarray(probs, dtype=float), 1e-7, 1 - 1e-7)
    mean = probs.mean(0)

    def H(q):
        return -(q * np.log(q) + (1 - q) * np.log(1 - q))

    entropy = H(mean)
    expected_entropy = H(probs).mean(0)
    return dict(prob=mean, entropy=entropy, mutual_info=entropy - expected_entropy,
                std=probs.std(0))


def predict_mc(model, loader, device="cpu", T: int = 20):
    """Run T MC-dropout passes. Returns (labels, indices, probs[T, N])."""
    import torch
    import torch.nn.functional as F
    enable_mc_dropout(model)
    all_p, ys, ids = [], [], []
    # timm CNNs apply dropout only in the head, so the backbone runs once and only the head
    # is sampled T times (identical outputs, ~T times faster).
    head_only = hasattr(model, "forward_features") and hasattr(model, "forward_head")
    with torch.no_grad():
        for x, y, i in loader:
            x = x.to(device)
            if head_only:
                # timm applies head dropout functionally with training=self.training, which is False
                # in eval mode, so dropout must be applied explicitly here (fixed 2026-11-23: before
                # this fix all T passes were identical and "MC dropout" equalled one deterministic pass).
                f = model.forward_features(x)
                pooled = model.forward_head(f, pre_logits=True)
                rate = float(getattr(model, "drop_rate", 0.0))
                fc = model.get_classifier()
                ps = [torch.softmax(fc(F.dropout(pooled, p=rate, training=True)), 1)[:, 1].cpu().numpy()
                      for _ in range(T)]
            else:
                ps = [torch.softmax(model(x), 1)[:, 1].cpu().numpy() for _ in range(T)]
            all_p.append(np.stack(ps))
            ys.append(y.numpy()); ids.append(i.numpy())
    return np.concatenate(ys), np.concatenate(ids), np.concatenate(all_p, axis=1)


def predict_det(model, loader, device="cpu"):
    """Single deterministic pass (dropout off)."""
    import torch
    model.eval()
    ps, ys, ids = [], [], []
    with torch.no_grad():
        for x, y, i in loader:
            ps.append(torch.softmax(model(x.to(device)), 1)[:, 1].cpu().numpy())
            ys.append(y.numpy()); ids.append(i.numpy())
    return np.concatenate(ys), np.concatenate(ids), np.concatenate(ps)
