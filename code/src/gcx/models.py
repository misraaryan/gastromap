"""CNN classifiers (timm) and frozen pathology foundation-model encoders."""
from __future__ import annotations

def get_device() -> str:
    """CUDA GPU (Colab) > Apple Silicon GPU (MPS, Mac) > CPU."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Encoders that can be used frozen for feature extraction.
#   kind: 'timm' (any timm/hf-hub model) or 'hf' (transformers AutoModel, CLS token)
#   UNI is gated: accept the terms at https://huggingface.co/MahmoodLab/UNI and `huggingface-cli login`.
ENCODERS = {
    "resnet50_imagenet": dict(kind="timm", name="resnet50.a1_in1k", mean=IMAGENET_MEAN, std=IMAGENET_STD),
    "phikon":            dict(kind="hf", name="owkin/phikon", mean=IMAGENET_MEAN, std=IMAGENET_STD),
    "phikon_v2":         dict(kind="hf", name="owkin/phikon-v2", mean=IMAGENET_MEAN, std=IMAGENET_STD),
    "uni":               dict(kind="timm", name="hf-hub:MahmoodLab/uni", mean=IMAGENET_MEAN, std=IMAGENET_STD,
                              kwargs=dict(init_values=1e-5, dynamic_img_size=True)),
}


def build_cnn(arch: str = "resnet50", pretrained: bool = True, num_classes: int = 2,
              drop_rate: float = 0.3):
    """ImageNet-pretrained CNN with a dropout layer before the head (needed for MC dropout)."""
    import timm
    return timm.create_model(arch, pretrained=pretrained, num_classes=num_classes, drop_rate=drop_rate)


class FrozenEncoder:
    """Wraps a foundation model so that `encoder(batch) -> (B, D)` embeddings."""

    def __init__(self, key: str, device: str = "cpu"):
        import torch
        if key not in ENCODERS:
            raise KeyError(f"Unknown encoder {key}. Options: {list(ENCODERS)}")
        cfg = ENCODERS[key]
        self.key, self.cfg, self.device = key, cfg, device
        if cfg["kind"] == "timm":
            import timm
            self.model = timm.create_model(cfg["name"], pretrained=cfg.get("pretrained", True), num_classes=0,
                                           **cfg.get("kwargs", {}))
        else:
            from transformers import AutoModel
            self.model = AutoModel.from_pretrained(cfg["name"])
        self.model.eval().to(device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self._torch = torch

    @property
    def mean(self):
        return self.cfg["mean"]

    @property
    def std(self):
        return self.cfg["std"]

    def __call__(self, x):
        with self._torch.no_grad():
            x = x.to(self.device)
            if self.cfg["kind"] == "timm":
                return self.model(x)
            out = self.model(pixel_values=x)
            return out.last_hidden_state[:, 0]  # CLS token
