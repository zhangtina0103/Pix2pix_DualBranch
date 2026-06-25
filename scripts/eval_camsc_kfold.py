#!/usr/bin/env python3
"""
Aggregate CaMSC k-fold test metrics (Hoechst + WT1 SSIM/Pearson).

Reads test outputs from results/<name>/test_<epoch>/images/*_fake_B.tif
and writes per-fold + pooled summary CSV/JSON.

Usage:
  python scripts/eval_camsc_kfold.py \\
    --kfold-root ~/orcd/scratch/camsc/datasets/camsc_bf_kfold \\
    --results-root ./results \\
    --name-prefix camsc_bf_vanilla_fm_fold \\
    --epoch 80
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from skimage.io import imread
from skimage.metrics import structural_similarity as ssim

CHANNELS = ("hoechst", "wt1")


def _to_uint8(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img, dtype=np.float64)
    if arr.max() <= 1.0:
        arr = (arr + 1.0) / 2.0 * 255.0
    return np.clip(arr, 0, 255)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    if a.std() < 1e-8 or b.std() < 1e-8:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def score_image_dir(images_dir: Path) -> pd.DataFrame:
    rows = []
    for fake_path in sorted(images_dir.glob("*_fake_B.tif")):
        real_path = Path(str(fake_path).replace("_fake_B.tif", "_real_B.tif"))
        if not real_path.is_file():
            continue
        fake = _to_uint8(imread(fake_path))
        real = _to_uint8(imread(real_path))
        row = {"file_name": fake_path.stem.replace("_fake_B", "")}
        ssim_scores = []
        pearson_scores = []
        for ch, name in enumerate(CHANNELS):
            r = real[:, :, ch]
            f = fake[:, :, ch]
            s = float(ssim(r, f, data_range=255))
            p = _pearson(r, f)
            row[f"{name}_ssim"] = s
            row[f"{name}_pearson"] = p
            ssim_scores.append(s)
            pearson_scores.append(p)
        row["average_ssim"] = float(np.mean(ssim_scores))
        row["average_pearson"] = float(np.mean(pearson_scores))
        rows.append(row)
    return pd.DataFrame(rows)


def find_test_images_dir(results_root: Path, name: str, epoch: int) -> Path | None:
    direct = results_root / name / f"test_{epoch}" / "images"
    if direct.is_dir():
        return direct
    # fallback: latest test_* for this name
    parent = results_root / name
    if not parent.is_dir():
        return None
    candidates = sorted(parent.glob("test_*/images"), key=lambda p: p.parent.name)
    return candidates[-1] if candidates else None


def main() -> None:
    p = argparse.ArgumentParser(description="Aggregate CaMSC k-fold metrics")
    p.add_argument("--kfold-root", type=str, required=True)
    p.add_argument("--results-root", type=str, default="./results")
    p.add_argument("--name-prefix", type=str, default="camsc_bf_vanilla_fm_fold")
    p.add_argument("--epoch", type=int, default=80)
    p.add_argument("--k-folds", type=int, default=5)
    p.add_argument("--out-dir", type=str, default="")
    args = p.parse_args()

    kfold_root = Path(args.kfold_root).expanduser().resolve()
    results_root = Path(args.results_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else kfold_root / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    per_fold = []
    all_tiles = []

    for fold in range(args.k_folds):
        name = f"{args.name_prefix}{fold}"
        images_dir = find_test_images_dir(results_root, name, args.epoch)
        if images_dir is None:
            print(f"[warn] fold {fold}: no test images for {name} epoch {args.epoch}")
            continue
        df = score_image_dir(images_dir)
        if df.empty:
            print(f"[warn] fold {fold}: empty scores in {images_dir}")
            continue
        df["fold"] = fold
        df.to_csv(out_dir / f"fold{fold}_per_tile.csv", index=False)
        summary = {
            "fold": fold,
            "name": name,
            "n_test": len(df),
            "hoechst_ssim_mean": df["hoechst_ssim"].mean(),
            "hoechst_ssim_std": df["hoechst_ssim"].std(),
            "wt1_ssim_mean": df["wt1_ssim"].mean(),
            "wt1_ssim_std": df["wt1_ssim"].std(),
            "average_ssim_mean": df["average_ssim"].mean(),
            "average_ssim_std": df["average_ssim"].std(),
            "hoechst_pearson_mean": df["hoechst_pearson"].mean(),
            "wt1_pearson_mean": df["wt1_pearson"].mean(),
        }
        per_fold.append(summary)
        all_tiles.append(df)
        print(
            f"fold {fold}: n={len(df)} "
            f"Hoechst SSIM={summary['hoechst_ssim_mean']:.4f} "
            f"WT1 SSIM={summary['wt1_ssim_mean']:.4f}"
        )

    if not per_fold:
        raise SystemExit("No fold results found. Run test for each fold first.")

    fold_df = pd.DataFrame(per_fold)
    fold_df.to_csv(out_dir / "kfold_summary.csv", index=False)

    pooled = pd.concat(all_tiles, ignore_index=True)
    pooled.to_csv(out_dir / "all_folds_per_tile.csv", index=False)

    agg = {
        "k_folds": len(per_fold),
        "epoch": args.epoch,
        "n_test_tiles_total": int(len(pooled)),
        "hoechst_ssim_mean": float(pooled["hoechst_ssim"].mean()),
        "hoechst_ssim_std": float(pooled["hoechst_ssim"].std()),
        "wt1_ssim_mean": float(pooled["wt1_ssim"].mean()),
        "wt1_ssim_std": float(pooled["wt1_ssim"].std()),
        "average_ssim_mean": float(pooled["average_ssim"].mean()),
        "average_ssim_std": float(pooled["average_ssim"].std()),
        "hoechst_pearson_mean": float(pooled["hoechst_pearson"].mean()),
        "wt1_pearson_mean": float(pooled["wt1_pearson"].mean()),
        "fold_hoechst_ssim_mean_of_means": float(fold_df["hoechst_ssim_mean"].mean()),
        "fold_hoechst_ssim_std_of_means": float(fold_df["hoechst_ssim_mean"].std()),
        "fold_wt1_ssim_mean_of_means": float(fold_df["wt1_ssim_mean"].mean()),
        "fold_wt1_ssim_std_of_means": float(fold_df["wt1_ssim_mean"].std()),
    }
    (out_dir / "kfold_aggregate.json").write_text(json.dumps(agg, indent=2) + "\n")

    print("\n=== Pooled (all test tiles) ===")
    print(f"  Hoechst SSIM: {agg['hoechst_ssim_mean']:.4f} ± {agg['hoechst_ssim_std']:.4f}")
    print(f"  WT1 SSIM:     {agg['wt1_ssim_mean']:.4f} ± {agg['wt1_ssim_std']:.4f}")
    print("\n=== Mean of fold means ===")
    print(
        f"  Hoechst SSIM: {agg['fold_hoechst_ssim_mean_of_means']:.4f} "
        f"± {agg['fold_hoechst_ssim_std_of_means']:.4f}"
    )
    print(
        f"  WT1 SSIM:     {agg['fold_wt1_ssim_mean_of_means']:.4f} "
        f"± {agg['fold_wt1_ssim_std_of_means']:.4f}"
    )
    print(f"\nWrote {out_dir}/kfold_summary.csv and kfold_aggregate.json")


if __name__ == "__main__":
    main()
