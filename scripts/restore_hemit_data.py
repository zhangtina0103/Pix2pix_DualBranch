#!/usr/bin/env python3
"""
Convert pix2pix layout (trainA, trainB) → official HEMIT layout (train/input, train/label).

Inverse of scripts/prepare_hemit_data.py (filenames are preserved).

Usage:
  python scripts/restore_hemit_data.py
  python scripts/restore_hemit_data.py --src ./datasets/hemit --dst .
  python scripts/restore_hemit_data.py --src ./datasets/hemit --dst ./HEMIT --copy
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SRC = REPO_ROOT / "datasets" / "hemit"
DEFAULT_DST = REPO_ROOT


def link_or_copy(src: Path, dst: Path, use_symlink: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    # Follow symlinks in A/B so restored tree holds real paths (or new symlinks).
    src = src.resolve()
    if use_symlink:
        os.symlink(src, dst)
    else:
        shutil.copy2(src, dst)


def restore_split(ab_root: Path, dst_root: Path, split: str, use_symlink: bool) -> int:
    a_dir = ab_root / f"{split}A"
    b_dir = ab_root / f"{split}B"
    if not a_dir.is_dir():
        print(f"  [skip] no {a_dir}")
        return 0

    inp_dir = dst_root / split / "input"
    lab_dir = dst_root / split / "label"
    n = 0
    for pattern in ("*.tif", "*.tiff", "*.TIF", "*.TIFF"):
        for a_path in sorted(a_dir.glob(pattern)):
            b_path = b_dir / a_path.name
            if not b_path.exists():
                found = list(b_dir.glob(a_path.stem + ".*"))
                if not found:
                    print(f"  [warn] missing label for {a_path.name}")
                    continue
                b_path = found[0]
            link_or_copy(a_path, inp_dir / a_path.name, use_symlink)
            link_or_copy(b_path, lab_dir / b_path.name, use_symlink)
            n += 1
    print(f"  {split}: {n} pairs → {inp_dir.relative_to(dst_root)} , {lab_dir.relative_to(dst_root)}")
    return n


def main() -> None:
    p = argparse.ArgumentParser(description="Restore HEMIT layout from pix2pix trainA/trainB folders.")
    p.add_argument(
        "--src",
        type=str,
        default=str(DEFAULT_SRC),
        help="pix2pix dataroot with trainA, trainB, valA, valB, ... (default: ./datasets/hemit)",
    )
    p.add_argument(
        "--dst",
        type=str,
        default=str(DEFAULT_DST),
        help="HEMIT output root (default: repo root → train/input, train/label, ...)",
    )
    p.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of symlinks (use if you will delete datasets/hemit)",
    )
    args = p.parse_args()

    src = Path(args.src).expanduser().resolve()
    dst = Path(args.dst).expanduser().resolve()
    if not src.is_dir():
        raise SystemExit(f"Source not found: {src}")

    print(f"pix2pix src: {src}")
    print(f"HEMIT dst:  {dst}")
    total = 0
    for split in ("train", "val", "test"):
        total += restore_split(src, dst, split, use_symlink=not args.copy)
    if total == 0:
        raise SystemExit(
            "No image pairs found. Expected e.g. trainA/*.tif and matching trainB/*.tif"
        )
    print(f"Done. {total} pairs total under {dst}")


if __name__ == "__main__":
    main()
