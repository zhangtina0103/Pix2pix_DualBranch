"""
MONAI DiffusionModelUNet wrapper matching mentor flow_matching.py / flow_matching_v.py.

  - concat([x_t, cond]) -> UNet(x, timesteps=t)
  - optional tanh on output (flow_matching.py)

Requires monai>=1.4 (DiffusionModelUNet) or monai-generative as fallback.
"""
from __future__ import annotations

import importlib
from typing import Tuple, Type

import torch
import torch.nn as nn


def _load_diffusion_model_unet() -> Type[nn.Module]:
    """Import DiffusionModelUNet from core MONAI (>=1.4) or monai-generative."""
    errors = []
    for mod_path, attr in (
        ("monai.networks.nets", "DiffusionModelUNet"),
        ("generative.networks.nets", "DiffusionModelUNet"),
    ):
        try:
            mod = importlib.import_module(mod_path)
            return getattr(mod, attr)
        except (ImportError, AttributeError) as e:
            errors.append(f"{mod_path}: {e}")

    try:
        import monai
        ver = getattr(monai, "__version__", "unknown")
    except ImportError:
        ver = "not installed"

    raise ImportError(
        "DiffusionModelUNet is not available. "
        f"Detected monai {ver}. "
        "Install with: pip install -c scripts/constraints_pix2pix.txt 'monai>=1.4,<2' "
        "(or: pip install monai-generative). "
        f"Import errors: {' | '.join(errors)}"
    )


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
        DiffusionModelUNet = _load_diffusion_model_unet()

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
