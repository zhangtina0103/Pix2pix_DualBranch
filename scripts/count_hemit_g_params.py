#!/usr/bin/env python3
"""Count generator (netG) parameters for HEMIT fair-comparison baselines (~11.38M)."""
import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from models import networks
from models.nce_losses import EncoderFeatureExtractor, generator_module
from models.vanilla_fm_model import _CondUNet

TARGET = 11_378_179


def count_module(m) -> int:
    return sum(p.numel() for p in m.parameters())


def resnet9_g(ngf: int = 64, norm: str = "instance") -> int:
    g = networks.define_G(
        3, 3, ngf, "resnet_9blocks", norm, True, "normal", 0.02, []
    )
    return count_module(g)


def cut_nce_projectors(ngf: int = 64, norm: str = "instance") -> int:
    g = networks.define_G(
        3, 3, ngf, "resnet_9blocks", norm, True, "normal", 0.02, []
    )
    core = generator_module(g)
    f = EncoderFeatureExtractor(core)
    # netF holds a ref to G; only projectors are separate parameters
    return count_module(f.projectors)


def fm_g(channels: str, num_res_blocks: int) -> int:
    ch = tuple(int(x) for x in channels.split(","))
    return count_module(_CondUNet(channels=ch, num_res_blocks=num_res_blocks))


def main():
    p = argparse.ArgumentParser(description="HEMIT generator parameter counts")
    p.add_argument(
        "--model",
        choices=["pix2pix", "cut", "asp", "cyclegan", "vanilla_fm", "all"],
        default="all",
    )
    p.add_argument("--ngf", type=int, default=64)
    p.add_argument("--fm_channels", type=str, default="96,192,272")
    p.add_argument("--fm_num_res_blocks", type=int, default=2)
    args = p.parse_args()

    rows = []

    def add(name, g_params, note=""):
        rows.append((name, g_params, note))

    if args.model in ("pix2pix", "all"):
        n_batch = resnet9_g(args.ngf, "batch")
        add("pix2pix (netG, batch norm)", n_batch, "MODEL=pix2pix|resnet9")
    if args.model in ("cut", "asp", "all"):
        n_inst = resnet9_g(args.ngf, "instance")
        n_f = cut_nce_projectors(args.ngf, "instance")
        add("cut/asp netG (instance norm)", n_inst, "inference checkpoint *_net_G.pth")
        add("cut/asp netF (NCE projectors only)", n_f, "train only; not in *_net_G.pth")
        add("cut/asp G+F (optimizer)", n_inst + n_f, "train only")
    if args.model in ("cyclegan", "all"):
        n = resnet9_g(args.ngf, "instance")
        add("cyclegan netG_A or netG_B (each)", n, "H&E→multiplex uses G_A at test")
        add("cyclegan both generators", 2 * n, "training only")
    if args.model in ("vanilla_fm", "all"):
        n = fm_g(args.fm_channels, args.fm_num_res_blocks)
        add(
            f"vanilla_fm UNet ({args.fm_channels}, res={args.fm_num_res_blocks})",
            n,
            "MODEL=vanilla_fm",
        )

    print(f"Target (ResNet9 G, ngf={args.ngf}): {TARGET:,} ({TARGET / 1e6:.3f} M)\n")
    for name, n, note in rows:
        diff = n - TARGET
        tag = f"diff={diff:+,}" if "both" not in name and "G+F" not in name else ""
        print(f"{name}")
        print(f"  {n:,} ({n / 1e6:.3f} M)  {tag}")
        if note:
            print(f"  ({note})")
        print()


if __name__ == "__main__":
    main()
