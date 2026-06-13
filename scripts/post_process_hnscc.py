#!/usr/bin/env python3
"""Metrics for HNSCC 4-marker outputs (CD3, CD8, FoxP3, PanCK)."""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np
from skimage.io import imread
from skimage.metrics import peak_signal_noise_ratio, structural_similarity as ssim

MARKERS = ["cd3", "cd8", "foxp3", "panck"]


def resolve_image_dir(srcdir: str) -> str:
    srcdir = os.path.abspath(srcdir)
    images = os.path.join(srcdir, "images")
    if os.path.isdir(images) and any(f.endswith("_fake_B.tif") for f in os.listdir(images)):
        return images
    return srcdir


def compute_metrics(directory_name: str) -> None:
    csv_path = os.path.join(directory_name, "score.csv")
    header = ["file_name"]
    for m in MARKERS:
        header.extend([f"{m}_ssim", f"{m}_pearson", f"{m}_psnr"])
    header.extend(["average_ssim", "average_pearson", "average_psnr"])

    with open(csv_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)

        for filename in sorted(os.listdir(directory_name)):
            if not filename.endswith("_fake_B.tif"):
                continue
            fake = imread(os.path.join(directory_name, filename))
            base = filename[:-11]
            real = imread(os.path.join(directory_name, base + "_real_B.tif"))

            ssim_scores = []
            pearson_scores = []
            psnr_scores = []
            row = [base]
            tiny = 1e-15

            for i in range(len(MARKERS)):
                real_ch = real[:, :, i].astype(float)
                fake_ch = fake[:, :, i].astype(float)
                real_ch = real_ch.copy()
                fake_ch = fake_ch.copy()
                real_ch[0, 0] += tiny
                fake_ch[0, 0] += tiny

                s = ssim(real_ch, fake_ch, data_range=255)
                p = np.corrcoef(real_ch.flatten(), fake_ch.flatten())[0, 1]
                ps = peak_signal_noise_ratio(real_ch, fake_ch, data_range=255)
                ssim_scores.append(s)
                pearson_scores.append(p)
                psnr_scores.append(ps)
                row.extend([s, p, ps])

            row.extend([np.mean(ssim_scores), np.mean(pearson_scores), np.mean(psnr_scores)])
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--srcdir", type=str, required=True)
    args = parser.parse_args()
    directory_name = resolve_image_dir(args.srcdir)
    print(f"Metrics on: {directory_name}")
    compute_metrics(directory_name)


if __name__ == "__main__":
    main()
