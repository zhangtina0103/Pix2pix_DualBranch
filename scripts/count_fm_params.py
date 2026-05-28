#!/usr/bin/env python3
"""Print parameter count for vanilla FM UNet (match ResNet9 ~11.38M for fair comparison)."""
import argparse
import sys
from pathlib import Path

# Allow: python scripts/count_fm_params.py (repo root on sys.path)
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from models.vanilla_fm_model import _CondUNet

TARGET = 11_378_179


def count_params(channels: tuple[int, int, int], num_res_blocks: int) -> int:
    net = _CondUNet(channels=channels, num_res_blocks=num_res_blocks)
    return sum(x.numel() for x in net.parameters())


def report(channels: tuple[int, int, int], num_res_blocks: int) -> int:
    total = count_params(channels, num_res_blocks)
    ch_str = ",".join(map(str, channels))
    print(f"fm_channels={ch_str} fm_num_res_blocks={num_res_blocks}")
    print(f"Total parameters: {total:,} ({total / 1e6:.3f} M)")
    print(f"Target (ResNet9 G): {TARGET:,}  diff={total - TARGET:+,}")
    return total


def search_near_target(top_k: int = 8) -> None:
    """Grid search for configs closest to ResNet9 generator size."""
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
                    total = count_params(ch, n)
                    results.append((abs(total - TARGET), total, ch, n))
    results.sort(key=lambda x: x[0])
    print(f"Top {top_k} configs nearest to {TARGET:,} params:\n")
    for i, (err, total, ch, n) in enumerate(results[:top_k], 1):
        ch_str = ",".join(map(str, ch))
        print(
            f"  {i}. {ch_str}  res={n}  ->  {total:,} ({total/1e6:.3f}M)  "
            f"diff={total - TARGET:+,}  |err|={err:,}"
        )
    best = results[0]
    ch_str = ",".join(map(str, best[2]))
    print(f"\nSuggested export for training:")
    print(f'  export FM_CHANNELS="{ch_str}" FM_RESBLOCKS={best[3]}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fm_channels", type=str, default="104,208,288")
    p.add_argument("--fm_num_res_blocks", type=int, default=2)
    p.add_argument(
        "--search",
        action="store_true",
        help="print grid of configs closest to ResNet9 (~11.38M)",
    )
    args = p.parse_args()
    if args.search:
        search_near_target()
        return
    ch = tuple(int(x) for x in args.fm_channels.split(","))
    report(ch, int(args.fm_num_res_blocks))


if __name__ == "__main__":
    main()
