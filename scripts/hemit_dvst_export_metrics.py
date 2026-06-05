#!/usr/bin/env python3
"""Map latest D-VST eval PNGs + HEMIT GT to pix2pix TIFF layout for post_process.py."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.io import imread

# eval.py: {source_name}_{pose_name}-2026-06-04T22-02-59.png
_TIMESTAMP_SUFFIX = re.compile(r"-\d{4}-\d{2}-\d{2}T[\d-]+$")


def load_rgb(path: Path) -> np.ndarray:
    arr = imread(path)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[-1] > 3:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float64)
        if arr.max() <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def latest_sample_dir(dvst_root: Path) -> Path:
    candidates = sorted(dvst_root.glob("DVST_samples/infer_HEMIT_test-*"))
    if not candidates:
        candidates = sorted(dvst_root.glob("DVST_samples/*HEMIT*"))
    if not candidates:
        raise FileNotFoundError(f"No DVST_samples under {dvst_root}")
    latest = candidates[-1]
    sample_dirs = sorted(latest.glob("samples/sample_*"))
    if not sample_dirs:
        raise FileNotFoundError(f"No samples/ in {latest}")
    return sample_dirs[-1]


def pose_stem_from_png(stem: str) -> str | None:
    """Recover HE patch stem from D-VST eval PNG name."""
    base = _TIMESTAMP_SUFFIX.sub("", stem)
    # HEMIT paired_gt: source_name == pose_name → "{name}_{name}"
    n = len(base)
    for i in range(1, n):
        if base[i] != "_":
            continue
        left, right = base[:i], base[i + 1 :]
        if left == right:
            return left
    # Different source/pose: pose (HE) is the suffix after the last underscore block
    if "_" in base:
        return base.rsplit("_", 1)[-1]
    return base or None


def gt_for_png(stem: str, labels: dict[str, Path]) -> Path | None:
    pose = pose_stem_from_png(stem)
    if pose and pose in labels:
        return labels[pose]
    base = _TIMESTAMP_SUFFIX.sub("", stem)
    for lab_stem, path in labels.items():
        if base == f"{lab_stem}_{lab_stem}":
            return path
    return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dvst-root", required=True)
    p.add_argument("--pix2pix-root", required=True)
    p.add_argument("--data-root", required=True, help="HEMIT input/label root (DiffVS layout)")
    p.add_argument("--split", default="test")
    p.add_argument("--sample-dir", default=None)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    dvst_root = Path(args.dvst_root).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_dir = Path(args.sample_dir).resolve() if args.sample_dir else latest_sample_dir(dvst_root)
    print(f"Using samples: {sample_dir}")

    label_dir = data_root / args.split / "label"
    labels = {p.stem: p for p in label_dir.glob("*") if p.is_file()}
    if not labels:
        raise SystemExit(f"No labels under {label_dir}")

    pngs = sorted(sample_dir.glob("*.png"))
    n = 0
    skipped = 0
    for pred_png in pngs:
        gt = gt_for_png(pred_png.stem, labels)
        if gt is None:
            skipped += 1
            if skipped <= 5:
                print(f"[warn] no GT for {pred_png.name} (parsed pose={pose_stem_from_png(pred_png.stem)!r})")
            continue
        stem = gt.stem
        fake = load_rgb(pred_png)
        real = load_rgb(gt)
        if fake.shape[:2] != real.shape[:2]:
            fake_img = Image.fromarray(fake).resize((real.shape[1], real.shape[0]), Image.BICUBIC)
            fake = np.array(fake_img)
        Image.fromarray(real).save(out_dir / f"{stem}_real_B.tif")
        Image.fromarray(fake).save(out_dir / f"{stem}_fake_B.tif")
        n += 1
    if skipped > 5:
        print(f"[warn] ... and {skipped - 5} more PNGs without GT match")
    print(f"Exported {n}/{len(pngs)} pairs → {out_dir}")


if __name__ == "__main__":
    main()
