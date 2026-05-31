"""
Advanced custom FM U-Net: tri-head outputs, multi-scale H&E cross-attention (+seg cond).

Use --fm_use_tri_head / --fm_use_cross_attn with --fm_backbone custom.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, ch)
        self.norm2 = nn.GroupNorm(8, ch)

    def forward(self, x):
        h = F.silu(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return F.silu(x + h)


class _Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, n_res: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1)
        self.res = nn.Sequential(*[_ResBlock(out_ch) for _ in range(n_res)])

    def forward(self, x):
        return self.res(self.conv(x))


class _UpSkipConvTranspose(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, skip_ch: int, n_res: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        self.fuse = nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1)
        self.res = nn.Sequential(*[_ResBlock(out_ch) for _ in range(n_res)])

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        return self.res(self.fuse(torch.cat([x, skip], dim=1)))


class _UpSkipBilinear(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, skip_ch: int, n_res: int):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
        )
        self.fuse = nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1)
        self.res = nn.Sequential(*[_ResBlock(out_ch) for _ in range(n_res)])

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        return self.res(self.fuse(torch.cat([x, skip], dim=1)))


def _make_up_skip(up_mode: str, in_ch: int, out_ch: int, skip_ch: int, n_res: int) -> nn.Module:
    if up_mode == "conv_transpose":
        return _UpSkipConvTranspose(in_ch, out_ch, skip_ch, n_res)
    return _UpSkipBilinear(in_ch, out_ch, skip_ch, n_res)


def _init_he_proj_conv(conv: nn.Conv2d, mode: str = "gray") -> None:
    with torch.no_grad():
        conv.weight.zero_()
        conv.bias.zero_()
        if mode == "identity":
            for c in range(min(conv.in_channels, conv.out_channels)):
                conv.weight[c, c, 0, 0] = 1.0
        elif mode == "gray":
            for oc in range(conv.out_channels):
                for ic in range(conv.in_channels):
                    conv.weight[oc, ic, 0, 0] = 1.0 / conv.in_channels
        else:
            raise ValueError(f"unknown fm_he_proj_init={mode!r}")


class _HEEncoder(nn.Module):
    """Spatial pyramid on H&E (+seg) for cross-attention at each scale."""

    def __init__(self, in_ch: int, channels: Tuple[int, int, int], num_res_blocks: int = 1):
        super().__init__()
        c1, c2, c3 = channels
        self.stem = nn.Conv2d(in_ch, c1, 3, padding=1)
        self.res1 = nn.Sequential(*[_ResBlock(c1) for _ in range(num_res_blocks)])
        self.down1 = _Down(c1, c2, num_res_blocks)
        self.down2 = _Down(c2, c3, num_res_blocks)

    def forward(self, cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        e1 = self.res1(F.silu(self.stem(cond)))
        e2 = self.down1(e1)
        e3 = self.down2(e2)
        return e1, e2, e3


class _SpatialCrossAttn(nn.Module):
    """Queries = FM trunk; keys/values = H&E encoder features (same spatial size)."""

    def __init__(self, dim: int, context_dim: int, num_heads: int = 8):
        super().__init__()
        nh = num_heads
        while dim % nh != 0 and nh > 1:
            nh -= 1
        self.norm_q = nn.GroupNorm(min(8, dim), dim)
        self.norm_kv = nn.GroupNorm(min(8, context_dim), context_dim)
        self.kv_proj = nn.Conv2d(context_dim, dim, 1) if context_dim != dim else nn.Identity()
        self.attn = nn.MultiheadAttention(dim, nh, batch_first=True, dropout=0.0)
        self.proj_out = nn.Conv2d(dim, dim, 1)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if context.shape[2:] != x.shape[2:]:
            context = F.interpolate(
                context, size=x.shape[2:], mode="bilinear", align_corners=False,
            )
        q_in = self.norm_q(x)
        kv_in = self.kv_proj(self.norm_kv(context))
        b, c, h, w = q_in.shape
        q = q_in.flatten(2).transpose(1, 2)
        kv = kv_in.flatten(2).transpose(1, 2)
        out, _ = self.attn(q, kv, kv, need_weights=False)
        out = out.transpose(1, 2).view(b, c, h, w)
        return x + self.proj_out(out)


class CondUNetAdvanced(nn.Module):
    """
    FM U-Net: concat stem + optional tri-head + H&E cross-attn (bottleneck + decoder).
    """

    def __init__(
        self,
        in_ch: int = 7,
        out_ch: int = 3,
        channels: Tuple[int, int, int] = (96, 192, 256),
        num_res_blocks: int = 2,
        up_mode: str = "conv_transpose",
        cond_nc: int = 3,
        use_he_proj: bool = False,
        he_proj_init: str = "gray",
        use_tri_head: bool = True,
        use_cross_attn: bool = True,
        cross_attn_heads: int = 8,
        cross_attn_decoder: bool = True,
    ):
        super().__init__()
        c1, c2, c3 = channels
        self.use_tri_head = use_tri_head
        self.use_cross_attn = use_cross_attn

        self.he_proj = None
        if use_he_proj:
            self.he_proj = nn.Conv2d(3, 3, 1)
            _init_he_proj_conv(self.he_proj, he_proj_init)

        self.he_encoder = _HEEncoder(cond_nc, channels, num_res_blocks=1)
        self.stem = nn.Conv2d(in_ch, c1, 3, padding=1)
        self.res1 = nn.Sequential(*[_ResBlock(c1) for _ in range(num_res_blocks)])
        self.down1 = _Down(c1, c2, num_res_blocks)
        self.down2 = _Down(c2, c3, num_res_blocks)
        self.mid = nn.Sequential(*[_ResBlock(c3) for _ in range(max(1, num_res_blocks))])

        self.cross_mid: Optional[_SpatialCrossAttn] = None
        self.cross_up2: Optional[_SpatialCrossAttn] = None
        self.cross_up1: Optional[_SpatialCrossAttn] = None
        if use_cross_attn:
            self.cross_mid = _SpatialCrossAttn(c3, c3, cross_attn_heads)
            if cross_attn_decoder:
                self.cross_up2 = _SpatialCrossAttn(c2, c2, cross_attn_heads)
                self.cross_up1 = _SpatialCrossAttn(c1, c1, cross_attn_heads)

        self.up2 = _make_up_skip(up_mode, c3, c2, c2, num_res_blocks)
        self.up1 = _make_up_skip(up_mode, c2, c1, c1, num_res_blocks)
        if use_tri_head:
            self.marker_heads = nn.ModuleList(
                [nn.Conv2d(c1, 1, 3, padding=1) for _ in range(out_ch)]
            )
            self.head = None
        else:
            self.marker_heads = None
            self.head = nn.Conv2d(c1, out_ch, 3, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        cond_img: Optional[torch.Tensor] = None,
        cond_spatial: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        spatial = cond_spatial if cond_spatial is not None else cond_img
        he_feats = self.he_encoder(spatial) if spatial is not None and self.use_cross_attn else None

        h1 = self.res1(self.stem(x))
        h2 = self.down1(h1)
        h3 = self.down2(h2)
        h = self.mid(h3)
        if self.cross_mid is not None and he_feats is not None:
            h = self.cross_mid(h, he_feats[2])

        h = self.up2(h, h2)
        if self.cross_up2 is not None and he_feats is not None:
            h = self.cross_up2(h, he_feats[1])

        h = self.up1(h, h1)
        if self.cross_up1 is not None and he_feats is not None:
            h = self.cross_up1(h, he_feats[0])

        if self.marker_heads is not None:
            return torch.cat([head(h) for head in self.marker_heads], dim=1)
        return self.head(h)
