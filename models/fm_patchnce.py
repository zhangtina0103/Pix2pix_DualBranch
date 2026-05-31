"""PatchNCE-style structure loss for paired H&E → multiplex (CUT-inspired)."""
from __future__ import annotations

import random
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, feat_q: torch.Tensor, feat_k: torch.Tensor) -> torch.Tensor:
        feat_q = F.normalize(feat_q, dim=1)
        feat_k = F.normalize(feat_k, dim=1)
        b = feat_q.shape[0]
        pos = (feat_q * feat_k).sum(dim=1, keepdim=True)
        neg = feat_q @ feat_k.T
        logits = torch.cat([pos, neg], dim=1) / self.temperature
        labels = torch.zeros(b, dtype=torch.long, device=feat_q.device)
        return F.cross_entropy(logits, labels)


class _EncBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FMPatchNCEHead(nn.Module):
    """Shared encoder on 3ch images; one spatial scale for patch contrast."""

    def __init__(self, feat_dim: int = 256, n_patches: int = 256):
        super().__init__()
        self.n_patches = n_patches
        self.encoder = nn.Sequential(
            _EncBlock(3, 64),
            _EncBlock(64, 128),
            _EncBlock(128, 256),
        )
        self.proj = nn.Sequential(
            nn.Linear(256, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim),
        )
        self.criterion = PatchNCELoss()

    def _encode_patches(self, img: torch.Tensor,
                        idx_hw: Tuple[torch.Tensor, int, int]) -> torch.Tensor:
        h = self.encoder(img)
        b, c, hh, ww = h.shape
        idx, _, _ = idx_hw
        n = idx.shape[0]
        flat = h.flatten(2)[:, :, idx]
        flat = flat.permute(0, 2, 1).reshape(b * n, c)
        return self.proj(flat)

    def _sample_indices(self, h: int, w: int, device: torch.device) -> Tuple[torch.Tensor, int, int]:
        n = min(self.n_patches, h * w)
        idx = torch.randperm(h * w, device=device)[:n]
        return idx, h, w

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """src=H&E, tgt=predicted multiplex (x1_hat); same patch locations."""
        with torch.no_grad():
            h = self.encoder(src)
            idx_hw = self._sample_indices(h.shape[2], h.shape[3], src.device)
        feat_q = self._encode_patches(src, idx_hw)
        feat_k = self._encode_patches(tgt, idx_hw)
        return self.criterion(feat_q, feat_k)
