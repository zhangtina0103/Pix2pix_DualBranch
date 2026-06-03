#!/usr/bin/env python3
"""
Convert HEMIT layout (train/input, train/label) → pix2pix layout (trainA, trainB).

Usage:
  python scripts/prepare_hemit_data.py --src /path/to/data/hemit
  python scripts/prepare_hemit_data.py --src ../vs_v2/data/hemit --dst ./datasets/hemit

To revert A/B → official layout, see scripts/restore_hemit_data.py
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def _dst_ok(dst: Path, src: Path, use_symlink: bool) -> bool:
    """True if dst already links/copies to src and target data is readable."""
    if not dst.exists() and not dst.is_symlink():
        return False
    try:
        if dst.is_symlink():
            return dst.resolve() == src.resolve() and src.is_file() and src.stat().st_size > 0
        if use_symlink:
            return False
        return dst.resolve() == src.resolve() and dst.stat().st_size > 0
    except OSError:
        return False


def link_or_copy(src: Path, dst: Path, use_symlink: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if _dst_ok(dst, src, use_symlink):
        return
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if use_symlink:
        os.symlink(src.resolve(), dst)
    else:
        import shutil
        shutil.copy2(src, dst)


def prepare_split(src_root: Path, dst_root: Path, split: str, use_symlink: bool) -> int:
    inp_dir = src_root / split / "input"
    lab_dir = src_root / split / "label"
    if not inp_dir.is_dir():
        print(f"  [skip] no {inp_dir}")
        return 0

    a_dir = dst_root / f"{split}A"
    b_dir = dst_root / f"{split}B"
    n = 0
    for pattern in ("*.tif", "*.tiff", "*.TIF", "*.TIFF"):
        for inp_path in sorted(inp_dir.glob(pattern)):
            lab_path = lab_dir / inp_path.name
            if not lab_path.exists():
                # try alternate extension
                stem = inp_path.stem
                found = list(lab_dir.glob(stem + ".*"))
                if not found:
                    print(f"  [warn] missing label for {inp_path.name}")
                    continue
                lab_path = found[0]
            link_or_copy(inp_path, a_dir / inp_path.name, use_symlink)
            link_or_copy(lab_path, b_dir / lab_path.name, use_symlink)
            n += 1
    print(f"  {split}: {n} pairs → {a_dir.name}/ , {b_dir.name}/")
    return n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=str, required=True,
                   help="HEMIT root with train/val/test and input/label subfolders")
    p.add_argument("--dst", type=str, default="./datasets/hemit",
                   help="Output dataroot for train.py (--dataroot)")
    p.add_argument("--copy", action="store_true",
                   help="Copy files instead of symlinks")
    args = p.parse_args()

    src = Path(args.src).expanduser().resolve()
    dst = Path(args.dst).expanduser().resolve()
    if not src.is_dir():
        raise SystemExit(f"Source not found: {src}")

    print(f"HEMIT src: {src}")
    print(f"pix2pix dst: {dst}")
    total = 0
    for split in ("train", "val", "test"):
        total += prepare_split(src, dst, split, use_symlink=not args.copy)
    if total == 0:
        raise SystemExit("No image pairs found. Expected e.g. train/input/*.tif and train/label/*.tif")
    print(f"Done. {total} pairs total. Use: --dataroot {dst}")


if __name__ == "__main__":
    main()
