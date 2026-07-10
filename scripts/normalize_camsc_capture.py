#!/usr/bin/env python3
"""
Normalize nested CaMSC Capture*.tif batches into flat legacy names for prepare_camsc_bf.py.

New layout (20260709 Dropbox zip):
  caMSC_{pct}%/{field}/Capture00360.tif  (+ 361, 362)

Assumes 3 captures per field in sorted order map to BF, Hoechst, WT1 (override with --channel-order).

Writes flat files:
  CaMSC {pct}% 10x BF_{n}.tif
  CaMSC {pct}% 10x Hoechst_{n}.tif
  CaMSC {pct}% 10x WT1_{n}.tif

Use --index-offset 10 so new fields start at 11 (old batch uses 1–10).

Usage:
  python scripts/normalize_camsc_capture.py \\
    --src ~/orcd/scratch/camsc/20260709 \\
    --dst ~/orcd/scratch/camsc/camsc_all \\
    --index-offset 10

  # also copy legacy flat batch first:
  python scripts/normalize_camsc_capture.py \\
    --src ~/orcd/scratch/camsc/20260709 \\
    --dst ~/orcd/scratch/camsc/camsc_all \\
    --merge-old ~/orcd/scratch/camsc/20260504 \\
    --index-offset 10
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

_PCT_DIR_RE = re.compile(r"^caMSC_(\d+)%$", re.IGNORECASE)
_CAPTURE_RE = re.compile(r"^Capture(\d+)\.tif$", re.IGNORECASE)
_LEGACY_RE = re.compile(
    r"^CaMSC\s+(\d+)%\s+(\d+x)\s+(BF|Hoechst|WT1)(?:_+(\d+))?\.tif$",
    re.IGNORECASE,
)
_CHANNEL_SUFFIX = {"bf": "BF", "hoechst": "Hoechst", "wt1": "WT1"}


def _parse_channel_order(raw: str) -> tuple[str, str, str]:
    parts = [p.strip().lower() for p in raw.split(",")]
    if len(parts) != 3 or any(p not in _CHANNEL_SUFFIX for p in parts):
        raise ValueError(f"Invalid --channel-order {raw!r}; use bf,hoechst,wt1")
    return parts[0], parts[1], parts[2]


def discover_capture_triplets(src: Path, mag: str = "10x") -> list[dict]:
    rows: list[dict] = []
    for pct_dir in sorted(src.iterdir()):
        m = _PCT_DIR_RE.match(pct_dir.name)
        if not m or not pct_dir.is_dir():
            continue
        pct = m.group(1)
        for field_dir in sorted(pct_dir.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else p.name):
            if not field_dir.is_dir():
                continue
            captures = []
            for tif in field_dir.glob("*.tif"):
                cm = _CAPTURE_RE.match(tif.name)
                if cm:
                    captures.append((int(cm.group(1)), tif))
            if len(captures) != 3:
                print(f"  [skip] {field_dir}: expected 3 captures, spawns {len(captures)}", file=sys.stderr)
                continue
            captures.sort(key=lambda x: x[0])
            rows.append({
                "pct": pct,
                "mag": mag,
                "field": field_dir.name,
                "captures": [p for _, p in captures],
            })
    return rows


def copy_legacy_batch(old_src: Path, dst: Path) -> int:
    n = 0
    for path in sorted(old_src.glob("*.tif")):
        if not _LEGACY_RE.match(path.name):
            continue
        out = dst / path.name
        if out.exists():
            print(f"  [skip exists] {path.name}")
            continue
        shutil.copy2(path, out)
        n += 1
    return n


def write_normalized(
    rows: list[dict],
    dst: Path,
    channel_order: tuple[str, str, str],
    index_offset: int,
) -> int:
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for row in rows:
        pct = row["pct"]
        mag = row["mag"]
        try:
            field_idx = int(row["field"])
        except ValueError:
            print(f"  [skip] non-numeric field dir {row['field']}", file=sys.stderr)
            continue
        out_idx = index_offset + field_idx
        for ch_key, src_path in zip(channel_order, row["captures"]):
            suffix = _CHANNEL_SUFFIX[ch_key]
            out_name = f"CaMSC {pct}% {mag} {suffix}_{out_idx}.tif"
            out_path = dst / out_name
            if out_path.exists():
                print(f"  [skip exists] {out_name}")
                continue
            shutil.copy2(src_path, out_path)
            n += 1
    return n


def main() -> None:
    p = argparse.ArgumentParser(description="Normalize Capture-layout CaMSC TIFs to legacy flat names")
    p.add_argument("--src", type=str, required=True, help="Extracted 20260709 root (contains caMSC_*%/ folders)")
    p.add_argument("--dst", type=str, required=True, help="Flat output dir (e.g. camsc_all/)")
    p.add_argument("--merge-old", type=str, default="", help="Optional legacy flat dir (20260504) to copy first")
    p.add_argument("--mag", type=str, default="10x")
    p.add_argument("--index-offset", type=int, default=10,
                   help="Added to field subfolder id for output index (default 10 → fields 1..15 → indices 11..25)")
    p.add_argument("--channel-order", type=str, default="bf,hoechst,wt1",
                   help="Order of sorted Capture*.tif files (default: bf,hoechst,wt1)")
    args = p.parse_args()

    src = Path(args.src).expanduser().resolve()
    dst = Path(args.dst).expanduser().resolve()
    if not src.is_dir():
        raise SystemExit(f"Source not found: {src}")

    channel_order = _parse_channel_order(args.channel_order)
    rows = discover_capture_triplets(src, mag=args.mag)
    if not rows:
        raise SystemExit(f"No capture triplets found under {src}")

    print(f"Found {len(rows)} capture fields under {src}")
    if args.merge_old:
        old_src = Path(args.merge_old).expanduser().resolve()
        if not old_src.is_dir():
            raise SystemExit(f"--merge-old not found: {old_src}")
        n_old = copy_legacy_batch(old_src, dst)
        print(f"Copied {n_old} legacy TIFs from {old_src}")

    n_new = write_normalized(rows, dst, channel_order, args.index_offset)
    total = len(list(dst.glob("*.tif")))
    print(f"Wrote {n_new} normalized TIFs → {dst}")
    print(f"Combined pool: {total} TIFs ({total // 3} fields if all triplets complete)")
    print(f"Channel order assumed: {args.channel_order}")
    print("Next:")
    print(f"  export CAMSC_SRC={dst}")
    print("  sbatch bash_scripts/prepare_camsc_bf.sbatch")


if __name__ == "__main__":
    main()
