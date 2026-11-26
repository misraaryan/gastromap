"""Stain handling: Macenko normalisation and HED-space stain augmentation (numpy only)."""
from __future__ import annotations

import numpy as np

# Ruifrok & Johnston H&E-DAB matrix (rows = stain OD vectors)
_RGB_FROM_HED = np.array([[0.65, 0.70, 0.29],
                          [0.07, 0.99, 0.11],
                          [0.27, 0.57, 0.78]])
_HED_FROM_RGB = np.linalg.inv(_RGB_FROM_HED)


def _to_od(img: np.ndarray, Io: float = 240.0) -> np.ndarray:
    img = img.astype(np.float64)
    return -np.log((img + 1.0) / Io)


class MacenkoNormalizer:
    """Macenko et al. (ISBI 2009). Fit on one or more reference patches from the TARGET
    style, then transform any patch into that style."""

    def __init__(self, Io: float = 240.0, alpha: float = 1.0, beta: float = 0.15):
        self.Io, self.alpha, self.beta = Io, alpha, beta
        self.HE_ref = np.array([[0.5626, 0.2159], [0.7201, 0.8012], [0.4062, 0.5581]])
        self.maxC_ref = np.array([1.9705, 1.0308])

    def _stain_matrix(self, img):
        od = _to_od(img, self.Io).reshape(-1, 3)
        od_hat = od[~np.any(od < self.beta, axis=1)]
        if len(od_hat) < 10:
            return None, None
        _, eigvecs = np.linalg.eigh(np.cov(od_hat.T))
        v = eigvecs[:, 1:3]
        t = od_hat @ v
        phi = np.arctan2(t[:, 1], t[:, 0])
        lo, hi = np.percentile(phi, self.alpha), np.percentile(phi, 100 - self.alpha)
        v1 = v @ np.array([np.cos(lo), np.sin(lo)])
        v2 = v @ np.array([np.cos(hi), np.sin(hi)])
        HE = np.array([v1, v2]).T if v1[0] > v2[0] else np.array([v2, v1]).T
        HE = HE * np.sign(HE.sum(0, keepdims=True) + 1e-12)
        C = np.linalg.lstsq(HE, od.T, rcond=None)[0]
        maxC = np.percentile(C, 99, axis=1)
        return HE, maxC

    def fit(self, imgs):
        imgs = imgs if isinstance(imgs, (list, tuple)) else [imgs]
        big = np.concatenate([np.asarray(i).reshape(-1, 3) for i in imgs])[None]
        HE, maxC = self._stain_matrix(big)
        if HE is None:
            raise ValueError("Reference image has too little tissue")
        self.HE_ref, self.maxC_ref = HE, maxC
        return self

    def fit_source(self, imgs):
        """Dataset-level mode: estimate ONE stain matrix and concentration scale for the whole source dataset
        from pooled patches; transform() then uses it for every patch instead of a per-patch estimate."""
        imgs = imgs if isinstance(imgs, (list, tuple)) else [imgs]
        big = np.concatenate([np.asarray(i).reshape(-1, 3) for i in imgs])[None]
        HE, maxC = self._stain_matrix(big)
        if HE is None:
            raise ValueError("Source images have too little tissue")
        self.HE_src, self.maxC_src = HE, maxC
        return self

    def transform(self, img: np.ndarray) -> np.ndarray:
        h, w, _ = img.shape
        if getattr(self, "HE_src", None) is not None:      # dataset-level mode
            od = _to_od(img, self.Io).reshape(-1, 3)
            C = np.linalg.lstsq(self.HE_src, od.T, rcond=None)[0]
            C = C * (self.maxC_ref / (self.maxC_src + 1e-8))[:, None]
            out = self.Io * np.exp(-self.HE_ref @ C)
            return np.clip(out.T.reshape(h, w, 3), 0, 255).astype(np.uint8)
        HE, maxC = self._stain_matrix(img)
        if HE is None:  # background / blank patch: leave unchanged
            return img
        od = _to_od(img, self.Io).reshape(-1, 3)
        C = np.linalg.lstsq(HE, od.T, rcond=None)[0]
        C = C * (self.maxC_ref / (maxC + 1e-8))[:, None]
        out = self.Io * np.exp(-self.HE_ref @ C)
        return np.clip(out.T.reshape(h, w, 3), 0, 255).astype(np.uint8)


class HEDJitter:
    """Stain augmentation (Tellez et al. 2019 style): perturb each HED channel by
    x' = x * a + b, with a ~ U(1-s, 1+s), b ~ U(-s, s). Works on PIL images."""

    def __init__(self, sigma: float = 0.05, seed: int | None = None):
        self.sigma = sigma
        self.rng = np.random.default_rng(seed)

    def __call__(self, pil_img):
        from PIL import Image
        rgb = np.asarray(pil_img).astype(np.float64) / 255.0
        od = -np.log(np.clip(rgb, 1e-6, 1.0))
        hed = od.reshape(-1, 3) @ _HED_FROM_RGB
        a = self.rng.uniform(1 - self.sigma, 1 + self.sigma, 3)
        b = self.rng.uniform(-self.sigma, self.sigma, 3)
        hed = hed * a + b
        rgb2 = np.exp(-(hed @ _RGB_FROM_HED)).reshape(rgb.shape)
        return Image.fromarray(np.clip(rgb2 * 255, 0, 255).astype(np.uint8))
