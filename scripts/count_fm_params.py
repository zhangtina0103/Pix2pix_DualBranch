#!/usr/bin/env python3
"""Parameter counts for vanilla FM backbones (~11.38M ResNet9 G target)."""
import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from models.vanilla_fm_model import _CondUNet

TARGET = 11_378_179


def count_custom(channels: tuple[int, int, int], num_res_blocks: int) -> int:
    net = _CondUNet(channels=channels, num_res_blocks=num_res_blocks)
    return sum(x.numel() for x in net.parameters())


def count_monai(
    channels: tuple[int, ...],
    attn: tuple[bool, ...],
    num_res_blocks: int,
    num_head_channels: int,
) -> int:
    from models.mentor_flow_net import MentorFlowNet

    net = MentorFlowNet(
        channels=channels,
        attention_levels=attn,
        num_res_blocks=num_res_blocks,
        num_head_channels=num_head_channels,
    )
    return sum(x.numel() for x in net.parameters())


def report(backbone: str, total: int, label: str) -> int:
    print(label)
    print(f"  Total parameters: {total:,} ({total / 1e6:.3f} M)")
    print(f"  Target (ResNet9 G): {TARGET:,}  diff={total - TARGET:+,}")
    return total


def search_monai(top_k: int = 10) -> None:
    results = []
    for c1 in range(56, 120, 8):
        for c2 in range(112, 240, 16):
            if c2 <= c1:
                continue
            for c3 in range(160, 320, 16):
                if c3 <= c2:
                    continue
                for attn_last in (False, True):
                    attn = (False, False, attn_last)
                    for n in (1, 2):
                        ch = (c1, c2, c3)
                        try:
                            total = count_monai(ch, attn, n, 32)
                        except ImportError:
                            print("MONAI not installed; pip install -r requirements-hemit-extra.txt")
                            return
                        results.append((abs(total - TARGET), total, ch, attn, n))
    results.sort(key=lambda x: x[0])
    print(f"Top {top_k} MONAI configs nearest to {TARGET:,}:\n")
    for i, (err, total, ch, attn, n) in enumerate(results[:top_k], 1):
        ch_str = ",".join(map(str, ch))
        attn_str = ",".join("1" if a else "0" for a in attn)
        print(
            f"  {i}. {ch_str}  attn={attn_str}  res={n}  ->  "
            f"{total:,} ({total/1e6:.3f}M)  diff={total - TARGET:+,}"
        )
    best = results[0]
    ch_str = ",".join(map(str, best[2]))
    attn_str = ",".join("1" if a else "0" for a in best[3])
    print(f"\nSuggested:")
    print(f'  export FM_BACKBONE=monai FM_CHANNELS="{ch_str}" FM_ATTN="{attn_str}" FM_RESBLOCKS={best[4]}')


def search_custom(top_k: int = 8) -> None:
    results = []
    for c1 in range(88, 140, 8):
        for c2 in range(176, 288, 16):
            if c2 <= c1:
                continue
            for c3 in range(240, 400, 16):
                if c3 <= c2:
                    continue
                for n in (2, 3):
                    ch = (c1, c2, c3)
                    total = count_custom(ch, n)
                    results.append((abs(total - TARGET), total, ch, n))
    results.sort(key=lambda x: x[0])
    print(f"Top {top_k} custom skip-UNet configs nearest to {TARGET:,}:\n")
    for i, (err, total, ch, n) in enumerate(results[:top_k], 1):
        ch_str = ",".join(map(str, ch))
        print(f"  {i}. {ch_str}  res={n}  ->  {total:,} ({total/1e6:.3f}M)  diff={total - TARGET:+,}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fm_backbone", type=str, default="monai", choices=["monai", "custom"])
    p.add_argument("--fm_channels", type=str, default="64,128,192")
    p.add_argument("--fm_attn_levels", type=str, default="0,0,1")
    p.add_argument("--fm_num_res_blocks", type=int, default=2)
    p.add_argument("--fm_num_head_channels", type=int, default=32)
    p.add_argument("--search", action="store_true", help="grid search near ResNet9 size")
    args = p.parse_args()

    if args.search:
        if args.fm_backbone == "monai":
            search_monai()
        else:
            search_custom()
        return

    ch = tuple(int(x) for x in args.fm_channels.split(","))
    if args.fm_backbone == "monai":
        attn = tuple(bool(int(a)) for a in args.fm_attn_levels.split(","))
        total = count_monai(ch, attn, args.fm_num_res_blocks, args.fm_num_head_channels)
        label = (
            f"monai UNet  channels={args.fm_channels}  attn={args.fm_attn_levels}  "
            f"res={args.fm_num_res_blocks}"
        )
    else:
        if len(ch) != 3:
            p.error("custom backbone requires 3 comma-separated channel widths")
        total = count_custom(ch, args.fm_num_res_blocks)  # type: ignore[arg-type]
        label = f"custom skip-UNet  channels={args.fm_channels}  res={args.fm_num_res_blocks}"
    report(args.fm_backbone, total, label)


if __name__ == "__main__":
    main()
