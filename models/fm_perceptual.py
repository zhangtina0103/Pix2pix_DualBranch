"""Perceptual loss for vanilla FM (mentor parity at 1024² via optional downsample)."""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FMPerceptualLoss(nn.Module):
    """
    L1(x1_hat, x1) + λ * perceptual(x1_hat, x1).

    Tries MONAI VGG perceptual, then LPIPS, then torchvision VGG features.
    At 1024², downsample to ``perc_size`` (default 256) before VGG to save VRAM.
    """

    def __init__(self, device: torch.device, perc_size: int = 256):
        super().__init__()
        self.perc_size = int(perc_size)
        self._monai = None
        self._lpips = None
        self._vgg: Optional[nn.Module] = None
        self._backend = self._build_backend()
        if self._backend == "vgg":
            from torchvision import models
            vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_FEATURES)
            self._vgg = vgg.features[:16].to(device).eval()
            for p in self._vgg.parameters():
                p.requires_grad = False

    @property
    def available(self) -> bool:
        return self._backend is not None

    @property
    def backend_name(self) -> str:
        return self._backend or "none"

    def _build_backend(self) -> Optional[str]:
        try:
            from monai.losses import PerceptualLoss  # noqa: F401
            self._monai = PerceptualLoss(
                spatial_dims=2, network_type="vgg", is_fake_3d=False,
            )
            return "monai"
        except Exception:
            self._monai = None
        try:
            import lpips
            self._lpips = lpips.LPIPS(net="vgg").eval()
            return "lpips"
        except Exception:
            self._lpips = None
        return "vgg"

    def _maybe_downsample(self, x: torch.Tensor) -> torch.Tensor:
        if self.perc_size <= 0:
            return x
        h, w = x.shape[2], x.shape[3]
        if max(h, w) <= self.perc_size:
            return x
        return F.interpolate(
            x, size=(self.perc_size, self.perc_size),
            mode="bilinear", align_corners=False,
        )

    def _align_rgb3(self, pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """VGG/LPIPS expect 3 channels (HNSCC 4ch mIF → use first 3 markers)."""
        c = pred.shape[1]
        if c == 3:
            return pred, target
        if c > 3:
            return pred[:, :3], target[:, :3]
        if c == 1:
            return pred.expand(-1, 3, -1, -1), target.expand(-1, 3, -1, -1)
        # 2ch: pad with duplicate first channel
        pad_p = pred[:, :1]
        pad_t = target[:, :1]
        return torch.cat([pred, pad_p], dim=1), torch.cat([target, pad_t], dim=1)

    def _to_lpips_range(self, x: torch.Tensor) -> torch.Tensor:
        return x.clamp(-1, 1)

    def _vgg_features(self, x: torch.Tensor) -> torch.Tensor:
        # ImageNet norm; x in [-1, 1] -> [0, 1]
        x01 = (x + 1) * 0.5
        mean = x01.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = x01.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        x_n = (x01 - mean) / std
        return self._vgg(x_n)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = self._maybe_downsample(pred)
        target = self._maybe_downsample(target)
        pred, target = self._align_rgb3(pred, target)

        if self._backend == "monai":
            return self._monai(pred, target)
        if self._backend == "lpips":
            return self._lpips(self._to_lpips_range(pred), self._to_lpips_range(target)).mean()
        # torchvision VGG feature L1
        return F.l1_loss(self._vgg_features(pred), self._vgg_features(target))


def build_fm_perceptual(
    device: torch.device, lam: float, perc_size: int = 256,
) -> Tuple[Optional[FMPerceptualLoss], str]:
    if lam <= 0:
        return None, "off"
    fn = FMPerceptualLoss(device, perc_size=perc_size)
    if not fn.available:
        return None, "unavailable"
    if fn._backend == "lpips" and fn._lpips is not None:
        fn._lpips = fn._lpips.to(device)
    elif fn._backend == "monai" and fn._monai is not None:
        fn._monai = fn._monai.to(device)
    return fn, fn.backend_name
