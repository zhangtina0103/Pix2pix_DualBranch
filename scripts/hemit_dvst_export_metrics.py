#!/usr/bin/env python3
"""Map latest D-VST eval PNGs + HEMIT GT to pix2pix TIFF layout for post_process.py."""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.io import imread


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


def gt_path_for_pose(pose_path: Path, data_root: Path, split: str) -> Path | None:
    name = pose_path.name
    lab = data_root / split / "label" / name
    if lab.is_file():
        return lab
    stem = pose_path.stem
    for ext in (".tif", ".tiff", ".png"):
        p = data_root / split / "label" / (stem + ext)
        if p.is_file():
            return p
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

    pngs = sorted(sample_dir.glob("*.png"))
    n = 0
    for pred_png in pngs:
        # eval.py saves {source_name}_{pose_name}-timestamp.png
        base = pred_png.stem.split("-")[0]
        if "_" not in base:
            continue
        pose_name = base.rsplit("_", 1)[-1]
        # Find matching GT by scanning test/label
        gt = None
        for lab in (data_root / args.split / "label").glob("*"):
            if lab.stem == pose_name or pose_name in lab.stem:
                gt = lab
                break
        if gt is None:
            print(f"[warn] no GT for {pred_png.name}")
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
    print(f"Exported {n} pairs → {out_dir}")


if __name__ == "__main__":
    main()
