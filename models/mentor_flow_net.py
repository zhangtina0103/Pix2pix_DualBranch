"""
MONAI DiffusionModelUNet wrapper matching mentor flow_matching.py / flow_matching_v.py.

  - concat([x_t, cond]) -> UNet(x, timesteps=t)
  - optional tanh on output (flow_matching.py)

Requires monai>=1.4 (DiffusionModelUNet) or monai-generative as fallback.
"""
from __future__ import annotations

import importlib
import inspect
from typing import Any, Dict, Tuple, Type

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
        "Install: pip install monai-generative  (pix2pix_cuda: bash scripts/install_vanilla_fm_monai.sh). "
        f"Import errors: {' | '.join(errors)}"
    )


def _diffusion_unet_kwargs(
    unet_cls: Type[nn.Module],
    *,
    in_ch: int,
    out_ch: int,
    channels: Tuple[int, ...],
    attention_levels: Tuple[bool, ...],
    num_res_blocks: int,
    num_head_channels: int,
) -> Dict[str, Any]:
    """Core MONAI (>=1.4) uses channels=; monai-generative uses num_channels=."""
    params = inspect.signature(unet_cls.__init__).parameters
    kw: Dict[str, Any] = dict(
        spatial_dims=2,
        in_channels=in_ch + out_ch,
        out_channels=out_ch,
        attention_levels=attention_levels,
        num_res_blocks=num_res_blocks,
        num_head_channels=num_head_channels,
    )
    if "channels" in params:
        kw["channels"] = channels
    elif "num_channels" in params:
        kw["num_channels"] = channels
    else:
        raise TypeError(
            f"{unet_cls.__module__}.{unet_cls.__name__} has no channels/num_channels argument"
        )
    return kw


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
        unet_cls = _load_diffusion_model_unet()

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.use_tanh = use_tanh
        self.unet = unet_cls(
            **_diffusion_unet_kwargs(
                unet_cls,
                in_ch=in_ch,
                out_ch=out_ch,
                channels=channels,
                attention_levels=attention_levels,
                num_res_blocks=num_res_blocks,
                num_head_channels=num_head_channels,
            )
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x_t, cond], dim=1)
        out = self.unet(x, timesteps=t)
        if self.use_tanh:
            out = torch.tanh(out)
        return out
