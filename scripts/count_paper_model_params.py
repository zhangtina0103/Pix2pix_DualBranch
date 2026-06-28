#!/usr/bin/env python3
"""Parameter breakdown for paper Appendix (inference + training-only components)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from models import networks
from models.fm_cond_unet import CondUNetAdvanced
from models.nce_losses import EncoderFeatureExtractor, generator_module


def count(module) -> int:
    return sum(p.numel() for p in module.parameters())


def fmt(n: int) -> str:
    return f"{n:,} ({n / 1e6:.3f} M)"


def ours_cross_attn() -> list[tuple[str, int, str]]:
    """Deployed HEMIT model: custom U-Net + bottleneck cross-attn only."""
    net = CondUNetAdvanced(
        in_ch=7,
        out_ch=3,
        channels=(96, 192, 256),
        num_res_blocks=2,
        up_mode="conv_transpose",
        cond_nc=3,
        use_he_proj=False,
        use_tri_head=False,
        use_cross_attn=True,
        cross_attn_heads=4,
        cross_attn_decoder=False,
    )
    fm_body = count(net) - count(net.he_encoder) - count(net.cross_mid)
    return [
        ("H&E context encoder", count(net.he_encoder), "multi-scale CNN on H&E"),
        ("Bottleneck cross-attention (4 heads)", count(net.cross_mid), "64² pooled MHA"),
        ("FM velocity U-Net (stem/enc/dec/head)", fm_body, "in_ch=7, out_ch=3"),
        ("Total generator (inference)", count(net), "checkpoint *_net_G.pth"),
    ]


def gan_rows() -> dict[str, list[tuple[str, int, str]]]:
    g_pix = networks.define_G(3, 3, 64, "resnet_9blocks", "batch", True, "normal", 0.02, [])
    g_inst = networks.define_G(3, 3, 64, "resnet_9blocks", "instance", True, "normal", 0.02, [])
    d_pix = networks.define_D(6, 64, "basic", 3, "batch", "normal", 0.02, [])
    d_inst = networks.define_D(6, 64, "basic", 3, "instance", "normal", 0.02, [])
    f_proj = EncoderFeatureExtractor(generator_module(g_inst)).projectors

    return {
        "pix2pix": [
            ("Generator (ResNet-9)", count(g_pix), "inference"),
            ("Discriminator (PatchGAN)", count(d_pix), "training only"),
            ("Total at inference", count(g_pix), ""),
        ],
        "CUT": [
            ("Generator (ResNet-9)", count(g_inst), "inference"),
            ("NCE projectors (netF)", count(f_proj), "training only"),
            ("Discriminator (PatchGAN)", count(d_inst), "training only"),
            ("Total at inference", count(g_inst), ""),
        ],
        "ASP": [
            ("Generator (ResNet-9)", count(g_inst), "inference"),
            ("NCE projectors (netF)", count(f_proj), "training only"),
            ("Discriminator (PatchGAN)", count(d_inst), "training only"),
            ("Total at inference", count(g_inst), ""),
        ],
        "CycleGAN": [
            ("Generator G_A (H&E→mIF)", count(g_inst), "inference"),
            ("Generator G_B (mIF→H&E)", count(g_inst), "training only"),
            ("Discriminator D_A", count(d_inst), "training only"),
            ("Discriminator D_B", count(d_inst), "training only"),
            ("Total at inference", count(g_inst), "G_A only"),
        ],
    }


def print_block(title: str, rows: list[tuple[str, int, str]]) -> None:
    print(f"\n=== {title} ===")
    for name, n, note in rows:
        line = f"  {name:<42} {fmt(n)}"
        if note:
            line += f"  [{note}]"
        print(line)


def main() -> None:
    p = argparse.ArgumentParser(description="Paper appendix parameter breakdown")
    p.add_argument("--ours-ckpt", type=str, default="", help="optional: verify *_net_G.pth total")
    args = p.parse_args()

    print("Controlled HEMIT baselines (ngf=64, 512², joint 3-channel output)")
    for model, rows in gan_rows().items():
        print_block(model, rows)

    print_block("Ours (FM + H&E cross-attention)", ours_cross_attn())

    if args.ours_ckpt:
        import torch

        ckpt = torch.load(args.ours_ckpt, map_location="cpu")
        sd = ckpt.get("state_dict", ckpt)
        n = sum(v.numel() for v in sd.values() if hasattr(v, "numel"))
        print(f"\n=== Checkpoint verify ===\n  {args.ours_ckpt}\n  Total tensors: {fmt(n)}")

    print(
        "\n=== D-VST (reference; public architecture, zero-shot) ===\n"
        "  DiT transformer (PixArt-XL-2-512)     ~611,000,000 (~611 M)  [Chen et al., PixArt]\n"
        "  VAE (SD AutoencoderKL)                ~83,700,000 (~84 M)   [Rombach et al.]\n"
        "  CLIP vision encoder (ViT-L/14)         ~302,600,000 (~303 M)  [Radford et al.]\n"
        "  Total inference stack                  ~997,000,000 (~1.0 B)  [sum of above]\n"
        "  (Not re-counted here; weights not required in pix2pix_cuda env.)"
    )


if __name__ == "__main__":
    main()
