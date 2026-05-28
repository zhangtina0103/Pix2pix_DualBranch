#!/usr/bin/env python3
"""Print parameter count for vanilla FM UNet (match ResNet9 ~11.38M for fair comparison)."""
import argparse

from models.vanilla_fm_model import _CondUNet

TARGET = 11_378_179


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fm_channels", type=str, default="96,192,256")
    p.add_argument("--fm_num_res_blocks", type=int, default=2)
    args = p.parse_args()
    ch = tuple(int(x) for x in args.fm_channels.split(","))
    n = int(args.fm_num_res_blocks)
    net = _CondUNet(channels=ch, num_res_blocks=n)
    total = sum(x.numel() for x in net.parameters())
    print(f"fm_channels={args.fm_channels} fm_num_res_blocks={n}")
    print(f"Total parameters: {total:,} ({total / 1e6:.3f} M)")
    print(f"Target (ResNet9 G): {TARGET:,}  diff={total - TARGET:+,}")


if __name__ == "__main__":
    main()
