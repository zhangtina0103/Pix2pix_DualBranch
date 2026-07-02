#!/usr/bin/env python3
"""
Build HEMIT dataroots for multi-input CD3/panCK virtual staining.

Targets (trainB): [CD3, panCK, pad] — DAPI is never predicted.

Variants (three dataroots under --dst-base):
  he/       trainA = H&E RGB          → CD3, panCK
  dapi/     trainA = DAPI (3ch gray)  → CD3, panCK
  he_dapi/  trainA = H&E + trainSeg = DAPI (1ch) → CD3, panCK  (FM_USE_SEG at train)

Source: existing pix2pix dataroot (--src) with trainA=H&E, trainB=[DAPI,CD3,panCK].

Usage:
  python scripts/prepare_hemit_multiinput.py --src ./datasets/hemit
  python scripts/prepare_hemit_multiinput.py --src ./datasets/hemit --dst-base ./datasets/hemit_multi
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image


MARKERS_OUT = ("CD3", "panCK")
LABEL_ORDER_SRC = ("DAPI", "CD3", "panCK")


def _link_or_copy(src: Path, dst: Path, use_symlink: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if use_symlink:
        os.symlink(src.resolve(), dst)
    else:
        import shutil
        shutil.copy2(src, dst)


def _load_label_hwc(path: Path) -> np.ndarray:
    """Return H×W×3 uint8 in [DAPI, CD3, panCK] order."""
    try:
        arr = tifffile.imread(path)
    except Exception:
        arr = np.asarray(Image.open(path))
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[0] < min(arr.shape[1:3]):
        arr = np.moveaxis(arr[:3], 0, -1)
    if arr.ndim == 3 and arr.shape[-1] >= 3:
        out = arr[..., :3]
    elif arr.ndim == 3 and arr.shape[-1] == 1:
        out = np.repeat(arr, 3, axis=-1)
    else:
        raise ValueError(f"Unsupported label shape {arr.shape} in {path}")
    if out.max() <= 1.0:
        out = (out * 255.0).clip(0, 255)
    return out.astype(np.uint8)


def _gray_to_rgb(gray: np.ndarray) -> np.ndarray:
    return np.stack([gray, gray, gray], axis=-1)


def _write_label_cd3_panck(dst: Path, label_hwc: np.ndarray) -> None:
    cd3 = label_hwc[..., 1]
    panck = label_hwc[..., 2]
    pad = np.zeros_like(cd3)
    stack = np.stack([cd3, panck, pad], axis=-1)
    tifffile.imwrite(dst, stack)


def _write_dapi_gray(dst: Path, label_hwc: np.ndarray) -> None:
    dapi = label_hwc[..., 0]
    tifffile.imwrite(dst, dapi)


def _write_dapi_rgb(dst: Path, label_hwc: np.ndarray) -> None:
    dapi = label_hwc[..., 0]
    tifffile.imwrite(dst, _gray_to_rgb(dapi))


def prepare_variant(
    src: Path,
    dst: Path,
    variant: str,
    use_symlink: bool,
) -> int:
    dst.mkdir(parents=True, exist_ok=True)
    total = 0
    for split in ("train", "val", "test"):
        dir_a = src / f"{split}A"
        dir_b = src / f"{split}B"
        if not dir_a.is_dir():
            print(f"  [skip] no {dir_a}")
            continue

        out_a = dst / f"{split}A"
        out_b = dst / f"{split}B"
        out_seg = dst / f"{split}Seg"
        out_a.mkdir(parents=True, exist_ok=True)
        out_b.mkdir(parents=True, exist_ok=True)
        if variant == "he_dapi":
            out_seg.mkdir(parents=True, exist_ok=True)

        n = 0
        for pattern in ("*.tif", "*.tiff", "*.TIF", "*.TIFF", "*.png", "*.PNG"):
            for a_path in sorted(dir_a.glob(pattern)):
                b_path = dir_b / a_path.name
                if not b_path.exists():
                    stem = a_path.stem
                    found = list(dir_b.glob(stem + ".*"))
                    if not found:
                        print(f"  [warn] missing label for {a_path.name}", file=sys.stderr)
                        continue
                    b_path = found[0]

                name = a_path.name
                label = _load_label_hwc(b_path)
                out_b_path = out_b / name

                if variant == "he":
                    _link_or_copy(a_path, out_a / name, use_symlink)
                elif variant == "dapi":
                    _write_dapi_rgb(out_a / name, label)
                elif variant == "he_dapi":
                    _link_or_copy(a_path, out_a / name, use_symlink)
                    _write_dapi_gray(out_seg / name, label)
                else:
                    raise ValueError(variant)

                if out_b_path.exists() or out_b_path.is_symlink():
                    out_b_path.unlink()
                _write_label_cd3_panck(out_b_path, label)
                n += 1
        print(f"  {variant}/{split}: {n} pairs")
        total += n

    meta = {
        "task": f"hemit_multi_{variant}_to_cd3_panck",
        "variant": variant,
        "input": {
            "he": "H&E RGB (trainA)",
            "dapi": "DAPI repeated to RGB (trainA)",
            "he_dapi": "H&E RGB (trainA) + DAPI grayscale (trainSeg, fm_use_seg)",
        }[variant],
        "label_channels": ["CD3", "panCK", "pad"],
        "src_label_order": list(LABEL_ORDER_SRC),
        "train_input_nc": 3,
        "train_output_nc": 3,
        "fm_channel_weights": "2,1,0",
        "n_pairs": total,
        "src": str(src),
    }
    (dst / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return total


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare HEMIT multi-input CD3/panCK dataroots")
    p.add_argument("--src", type=str, required=True,
                   help="pix2pix HEMIT dataroot (trainA=H&E, trainB=DAPI,CD3,panCK)")
    p.add_argument("--dst-base", type=str, default="./datasets/hemit_multi",
                   help="Base dir; writes he/, dapi/, he_dapi/ underneath")
    p.add_argument("--variants", type=str, default="he,dapi,he_dapi",
                   help="Comma-separated: he, dapi, he_dapi")
    p.add_argument("--copy", action="store_true", help="Copy instead of symlink H&E")
    args = p.parse_args()

    src = Path(args.src).expanduser().resolve()
    dst_base = Path(args.dst_base).expanduser().resolve()
    if not src.is_dir():
        raise SystemExit(f"Source not found: {src}")
    if not (src / "trainA").is_dir() or not (src / "trainB").is_dir():
        raise SystemExit(f"Expected trainA/trainB under {src}")

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    for v in variants:
        if v not in ("he", "dapi", "he_dapi"):
            raise SystemExit(f"Unknown variant: {v}")

    print(f"HEMIT multi-input prep")
    print(f"  src:      {src}")
    print(f"  dst-base: {dst_base}")
    print(f"  variants: {variants}")

    grand = 0
    for variant in variants:
        dst = dst_base / variant
        print(f"\n==> {variant} → {dst}")
        n = prepare_variant(src, dst, variant, use_symlink=not args.copy)
        grand += n

    print(f"\nDone. {grand} total pair-writes across {len(variants)} variant(s).")
    print("Train examples:")
    print("  H&E only:     DATAROOT=.../he       HEMIT_MULTI_VARIANT=he")
    print("  DAPI only:    DATAROOT=.../dapi     HEMIT_MULTI_VARIANT=dapi")
    print("  H&E+DAPI:     DATAROOT=.../he_dapi  HEMIT_MULTI_VARIANT=he_dapi")


if __name__ == "__main__":
    main()
