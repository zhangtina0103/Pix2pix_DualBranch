"""
MONAI DiffusionModelUNet wrapper matching mentor flow_matching.py / flow_matching_v.py.

  - concat([x_t, cond]) -> UNet(x, timesteps=t)
  - optional tanh on output (flow_matching.py)
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


def require_monai():
    try:
        from monai.networks.nets import DiffusionModelUNet  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "vanilla_fm with --fm_backbone monai requires MONAI. "
            "Install: pip install -r requirements-hemit-extra.txt"
        ) from e


class MentorFlowNet(nn.Module):
    """
    Joint HEMIT FM backbone (A=H&E 3ch, B=multiplex 3ch).

    in_channels  = x_t (out_ch) + cond (in_ch)  — typically 6
    out_channels = out_ch — typically 3
    """

    def __init__(
        self,
        in_ch: int = 3,
        out_ch: int = 3,
        channels: Tuple[int, ...] = (64, 128, 192),
        attention_levels: Tuple[bool, ...] = (False, False, True),
        num_res_blocks: int = 2,
        num_head_channels: int = 32,
        use_tanh: bool = False,
    ):
        super().__init__()
        require_monai()
        from monai.networks.nets import DiffusionModelUNet

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.use_tanh = use_tanh
        self.unet = DiffusionModelUNet(
            spatial_dims=2,
            in_channels=in_ch + out_ch,
            out_channels=out_ch,
            channels=channels,
            attention_levels=attention_levels,
            num_res_blocks=num_res_blocks,
            num_head_channels=num_head_channels,
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x_t, cond], dim=1)
        out = self.unet(x, timesteps=t)
        if self.use_tanh:
            out = torch.tanh(out)
        return out
