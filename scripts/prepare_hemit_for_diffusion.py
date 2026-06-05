#!/usr/bin/env python3
"""Prepare HEMIT data for DiffVS and D-VST from official input/label layout.

DiffVS:  {out}/train/{input,label}/<file>
D-VST:   {out}/HE/<slide>/<file>  and  mIHC/<slide>/<file>

Usage:
  python scripts/prepare_hemit_for_diffusion.py --src /path/to/hemit --format diffvs
  python scripts/prepare_hemit_for_diffusion.py --src ./datasets/hemit --format dvst --from-ab
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

SLIDE_RE = re.compile(r"^\[(?P<x>[^,\]]+),(?P<y>[^\]]+)\]_patch_")


def slide_id_from_name(name: str) -> str:
    m = SLIDE_RE.match(name)
    if m:
        return f"{m.group('x')}_{m.group('y')}"
    return Path(name).stem.split("_patch_")[0].strip("[]") or "unknown"


def link_or_copy(src: Path, dst: Path, use_symlink: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() and dst.resolve() == src.resolve():
        return
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if use_symlink:
        os.symlink(src.resolve(), dst)
    else:
        import shutil
        shutil.copy2(src, dst)


def iter_pairs(src_root: Path, split: str, from_ab: bool) -> list[tuple[Path, Path, str]]:
    pairs: list[tuple[Path, Path, str]] = []
    if from_ab:
        a_dir = src_root / f"{split}A"
        b_dir = src_root / f"{split}B"
        if not a_dir.is_dir():
            return pairs
        for pattern in ("*.tif", "*.tiff", "*.TIF", "*.TIFF", "*.png", "*.PNG"):
            for inp in sorted(a_dir.glob(pattern)):
                lab = b_dir / inp.name
                if not lab.exists():
                    alts = list(b_dir.glob(inp.stem + ".*"))
                    if not alts:
                        print(f"  [warn] missing label for {inp.name}")
                        continue
                    lab = alts[0]
                pairs.append((inp, lab, inp.name))
        return pairs

    inp_dir = src_root / split / "input"
    lab_dir = src_root / split / "label"
    if not inp_dir.is_dir():
        return pairs
    for pattern in ("*.tif", "*.tiff", "*.TIF", "*.TIFF", "*.png", "*.PNG"):
        for inp in sorted(inp_dir.glob(pattern)):
            lab = lab_dir / inp.name
            if not lab.exists():
                alts = list(lab_dir.glob(inp.stem + ".*"))
                if not alts:
                    print(f"  [warn] missing label for {inp.name}")
                    continue
                lab = alts[0]
            pairs.append((inp, lab, inp.name))
    return pairs


def prepare_diffvs(src_root: Path, dst_root: Path, use_symlink: bool, from_ab: bool) -> int:
    n = 0
    for split in ("train", "val", "test"):
        pairs = iter_pairs(src_root, split, from_ab)
        for inp, lab, name in pairs:
            link_or_copy(inp, dst_root / split / "input" / name, use_symlink)
            link_or_copy(lab, dst_root / split / "label" / name, use_symlink)
            n += 1
        print(f"  diffvs {split}: {len(pairs)} pairs")
    return n


def prepare_dvst(src_root: Path, dst_root: Path, use_symlink: bool, from_ab: bool) -> int:
    n = 0
    for split in ("train", "val", "test"):
        pairs = iter_pairs(src_root, split, from_ab)
        for inp, lab, name in pairs:
            slide = slide_id_from_name(name)
            link_or_copy(inp, dst_root / "HE" / slide / name, use_symlink)
            link_or_copy(lab, dst_root / "mIHC" / slide / name, use_symlink)
            # Per-split input/label for eval (test-only metrics vs other baselines).
            link_or_copy(inp, dst_root / split / "input" / name, use_symlink)
            link_or_copy(lab, dst_root / split / "label" / name, use_symlink)
            n += 1
        print(f"  dvst {split}: {len(pairs)} pairs")
    return n


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=str, required=True, help="HEMIT root (input/label) or pix2pix dataroot (trainA/trainB)")
    p.add_argument("--dst", type=str, default=None)
    p.add_argument("--format", choices=["diffvs", "dvst", "both"], default="both")
    p.add_argument("--copy", action="store_true", help="Copy instead of symlink")
    p.add_argument("--from-ab", action="store_true", help="Source is pix2pix trainA/trainB layout")
    args = p.parse_args()

    src = Path(args.src).expanduser().resolve()
    use_symlink = not args.copy

    if args.format in ("diffvs", "both"):
        dst = Path(args.dst or "./datasets/hemit_diffvs").expanduser().resolve()
        print(f"DiffVS → {dst}")
        prepare_diffvs(src, dst, use_symlink, args.from_ab)

    if args.format in ("dvst", "both"):
        dst = Path(args.dst or "./datasets/hemit_dvst").expanduser().resolve()
        if args.format == "both" and args.dst:
            dst = dst.parent / "hemit_dvst"
        print(f"D-VST → {dst}")
        prepare_dvst(src, dst, use_symlink, args.from_ab)

    print("Done.")


if __name__ == "__main__":
    main()
