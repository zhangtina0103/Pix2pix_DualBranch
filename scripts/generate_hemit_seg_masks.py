#!/usr/bin/env python3
"""
Build pseudo segmentation masks from H&E (trainA/valA/testA) for FM conditioning.

Writes parallel folders: trainSeg/, valSeg/, testSeg/ (1ch uint8 TIFF, 0/255).
Uses Otsu on inverted luminance + light morphology (skimage if installed).

Usage:
  python scripts/generate_hemit_seg_masks.py --dataroot ./datasets/hemit
  python scripts/generate_hemit_seg_masks.py --src /path/to/hemit --dst ./datasets/hemit
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from skimage.filters import threshold_otsu
    from skimage.morphology import binary_closing, binary_opening, disk

    _HAS_SKIMAGE = True
except ImportError:
    _HAS_SKIMAGE = False


def _rgb_to_nuclei_mask(arr: np.ndarray) -> np.ndarray:
    """arr: H,W,3 uint8 -> H,W bool nuclei-ish mask."""
    gray = arr.astype(np.float32).mean(axis=-1)
    # Nuclei often darker / bluish in H&E RGB
    score = (arr[..., 2].astype(np.float32) * 0.5 + (255.0 - gray) * 0.5)
    if _HAS_SKIMAGE:
        thr = threshold_otsu(score)
        mask = score > thr
        se = disk(2)
        mask = binary_opening(mask, se)
        mask = binary_closing(mask, se)
        return mask
    thr = np.percentile(score, 72)
    return score > thr


def process_split(a_dir: Path, seg_dir: Path) -> int:
    if not a_dir.is_dir():
        return 0
    seg_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for pattern in ("*.tif", "*.tiff", "*.TIF", "*.TIFF", "*.png", "*.PNG"):
        for path in sorted(a_dir.glob(pattern)):
            img = np.array(Image.open(path).convert("RGB"))
            mask = _rgb_to_nuclei_mask(img)
            out = (mask.astype(np.uint8) * 255)
            Image.fromarray(out, mode="L").save(seg_dir / path.name)
            n += 1
    return n


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataroot", type=str, default="",
                   help="pix2pix dataroot with trainA/ (writes trainSeg/ etc.)")
    p.add_argument("--src", type=str, default="",
                   help="HEMIT layout root; runs prepare then generates on dst")
    p.add_argument("--dst", type=str, default="./datasets/hemit",
                   help="Output dataroot when using --src")
    args = p.parse_args()

    if args.src:
        import subprocess
        import sys

        src = Path(args.src).expanduser().resolve()
        dst = Path(args.dst).expanduser().resolve()
        repo = Path(__file__).resolve().parents[1]
        subprocess.check_call(
            [sys.executable, str(repo / "scripts/prepare_hemit_data.py"),
             "--src", str(src), "--dst", str(dst)],
        )
        dataroot = dst
    elif args.dataroot:
        dataroot = Path(args.dataroot).expanduser().resolve()
    else:
        raise SystemExit("Provide --dataroot or --src")

    if not _HAS_SKIMAGE:
        print("Warning: skimage not found; using percentile threshold only")

    total = 0
    for split in ("train", "val", "test"):
        a_dir = dataroot / f"{split}A"
        seg_dir = dataroot / f"{split}Seg"
        n = process_split(a_dir, seg_dir)
        print(f"  {split}: {n} masks -> {seg_dir}")
        total += n
    if total == 0:
        raise SystemExit(f"No images in {dataroot}/*A/")
    print(f"Done. {total} seg masks. Use DATASET_MODE=aligned_cond")


if __name__ == "__main__":
    main()
