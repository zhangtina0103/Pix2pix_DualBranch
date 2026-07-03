#!/usr/bin/env python3
"""Metrics for HEMIT multi-input runs (label order: CD3, panCK, pad)."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity as ssim
from tqdm import tqdm


CHANNELS = ("cd3", "panck")


def resolve_image_dir(srcdir: Path) -> Path:
    """test.py saves TIFFs under <test_epoch>/images/."""
    srcdir = srcdir.resolve()
    images = srcdir / "images"
    if images.is_dir():
        for ext in (".tif", ".tiff", ".png"):
            if any(images.glob(f"*_fake_B{ext}")):
                return images
    return srcdir


def _load_hwc(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path))
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[0] < min(arr.shape[1:3]):
        arr = np.moveaxis(arr[:3], 0, -1)
    if arr.ndim == 3 and arr.shape[-1] >= 3:
        return arr[..., :3].astype(np.float32)
    raise ValueError(f"Bad image shape {arr.shape} in {path}")


def _find_pairs(img_dir: Path) -> list[tuple[Path, Path, str]]:
    pairs = []
    for ext in (".tif", ".tiff", ".png"):
        for fake_path in sorted(img_dir.glob(f"*_fake_B{ext}")):
            stem = fake_path.name.replace(f"_fake_B{ext}", "")
            real_path = img_dir / f"{stem}_real_B{ext}"
            if real_path.is_file():
                pairs.append((real_path, fake_path, stem))
    return pairs


def main() -> None:
    p = argparse.ArgumentParser(description="Eval HEMIT multi-input CD3/panCK predictions")
    p.add_argument("--srcdir", type=str, required=True, help="results/.../test_NN/")
    p.add_argument("--out-csv", type=str, default="", help="Output CSV (default: srcdir/score_cd3_panck.csv)")
    args = p.parse_args()

    srcdir = Path(args.srcdir).expanduser()
    img_dir = resolve_image_dir(srcdir)
    out_csv = Path(args.out_csv).expanduser() if args.out_csv else srcdir / "score_cd3_panck.csv"

    pairs = _find_pairs(img_dir)
    if not pairs:
        raise SystemExit(f"No *_fake_B.{{tif,png}} pairs under {img_dir}")

    print(f"Metrics on: {img_dir} ({len(pairs)} tiles)")

    rows = []
    for real_path, fake_path, stem in tqdm(pairs, desc="score_cd3_panck"):
        real = _load_hwc(real_path)
        fake = _load_hwc(fake_path)
        row = {"file_name": stem}
        ssims, pears = [], []
        for i, ch in enumerate(CHANNELS):
            r = real[..., i].astype(float)
            f = fake[..., i].astype(float)
            r[0, 0] += 1e-6
            f[0, 0] += 1e-6
            s = ssim(r, f, data_range=255)
            pear = float(np.corrcoef(r.flatten(), f.flatten())[0, 1])
            psnr = peak_signal_noise_ratio(r, f, data_range=255)
            row[f"{ch}_ssim"] = s
            row[f"{ch}_pearson"] = pear
            row[f"{ch}_psnr"] = psnr
            ssims.append(s)
            pears.append(pear)
        row["average_ssim"] = float(np.mean(ssims))
        row["average_pearson"] = float(np.mean(pears))
        rows.append(row)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        ["file_name"]
        + [f"{c}_ssim" for c in CHANNELS]
        + [f"{c}_pearson" for c in CHANNELS]
        + [f"{c}_psnr" for c in CHANNELS]
        + ["average_ssim", "average_pearson"]
    )
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    agg = {k: float(np.mean([r[k] for r in rows])) for k in fieldnames if k != "file_name"}
    print(f"Wrote {out_csv} ({len(rows)} tiles)")
    print(f"  CD3  SSIM={agg['cd3_ssim']:.4f}  Pearson={agg['cd3_pearson']:.4f}")
    print(f"  panCK SSIM={agg['panck_ssim']:.4f}  Pearson={agg['panck_pearson']:.4f}")
    print(f"  avg  SSIM={agg['average_ssim']:.4f}  Pearson={agg['average_pearson']:.4f}")


if __name__ == "__main__":
    main()
