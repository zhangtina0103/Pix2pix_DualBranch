#!/usr/bin/env python3
"""
Aggregate CaMSC k-fold test metrics (Hoechst + WT1).

Metrics per channel: SSIM, Pearson, Spearman, PSNR, MAE, MSE, RMSE, NMSE, R², LPIPS (if installed).

Reads test outputs from results/<name>/test_<epoch>/images/*_fake_B.tif
and writes per-fold + pooled summary CSV/JSON.

Usage:
  python scripts/eval_camsc_kfold.py \\
    --kfold-root ~/orcd/scratch/camsc/datasets/camsc_bf_kfold_aug \\
    --results-root ./results \\
    --name-prefix camsc_bf_fm_cross_attn_ft_fold \\
    --name-suffix _512_aug_512 \\
    --epoch 110
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.hemit_eval.extended_metrics import (  # noqa: E402
    METRIC_SPECS,
    _LpipsScorer,
    _channel_metrics,
)

CAMSC_CHANNELS = ("hoechst", "wt1")
CAMSC_CHANNEL_IDX = (0, 1)
MAIN_METRICS = ("ssim", "pearson", "spearman", "psnr", "mae", "rmse", "lpips")


def _to_uint8(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img, dtype=np.float64)
    if arr.max() <= 1.0:
        arr = (arr + 1.0) / 2.0 * 255.0
    return np.clip(arr, 0, 255)


def score_image_dir(images_dir: Path, lpips_scorer: _LpipsScorer | None) -> pd.DataFrame:
    from skimage.io import imread

    rows = []
    for fake_path in sorted(images_dir.glob("*_fake_B.tif")):
        real_path = Path(str(fake_path).replace("_fake_B.tif", "_real_B.tif"))
        if not real_path.is_file():
            continue
        fake = _to_uint8(imread(fake_path))
        real = _to_uint8(imread(real_path))
        row = {"file_name": fake_path.stem.replace("_fake_B", "")}
        channel_metrics: dict[str, dict[str, float]] = {}
        for ch_idx, name in zip(CAMSC_CHANNEL_IDX, CAMSC_CHANNELS):
            r = real[:, :, ch_idx]
            f = fake[:, :, ch_idx]
            m = _channel_metrics(r, f)
            if lpips_scorer is not None and lpips_scorer.available:
                m["lpips"] = lpips_scorer.score(r, f)
            else:
                m["lpips"] = float("nan")
            channel_metrics[name] = m
            for metric, val in m.items():
                row[f"{name}_{metric}"] = val
        for metric in METRIC_SPECS:
            row[f"average_{metric}"] = float(
                np.nanmean([channel_metrics[ch][metric] for ch in CAMSC_CHANNELS])
            )
        rows.append(row)
    return pd.DataFrame(rows)


def find_test_images_dir(results_root: Path, name: str, epoch: int) -> Path | None:
    direct = results_root / name / f"test_{epoch}" / "images"
    if direct.is_dir():
        return direct
    parent = results_root / name
    if not parent.is_dir():
        return None
    candidates = sorted(parent.glob("test_*/images"), key=lambda p: p.parent.name)
    return candidates[-1] if candidates else None


def _fold_summary(df: pd.DataFrame, fold: int, name: str) -> dict:
    summary: dict = {"fold": fold, "name": name, "n_test": len(df)}
    for ch in CAMSC_CHANNELS:
        for metric in METRIC_SPECS:
            col = f"{ch}_{metric}"
            summary[f"{ch}_{metric}_mean"] = float(df[col].mean())
            summary[f"{ch}_{metric}_std"] = float(df[col].std())
    for metric in METRIC_SPECS:
        col = f"average_{metric}"
        summary[f"average_{metric}_mean"] = float(df[col].mean())
        summary[f"average_{metric}_std"] = float(df[col].std())
    return summary


def _pooled_agg(pooled: pd.DataFrame, fold_df: pd.DataFrame) -> dict:
    agg: dict = {
        "k_folds": int(len(fold_df)),
        "n_test_tiles_total": int(len(pooled)),
        "lpips_available": bool(pooled["average_lpips"].notna().any()),
    }
    for ch in list(CAMSC_CHANNELS) + ["average"]:
        for metric in METRIC_SPECS:
            col = f"{ch}_{metric}" if ch != "average" else f"average_{metric}"
            agg[f"{col}_mean"] = float(pooled[col].mean())
            agg[f"{col}_std"] = float(pooled[col].std())
            fold_col = f"{ch}_{metric}_mean" if ch != "average" else f"average_{metric}_mean"
            if fold_col in fold_df.columns:
                agg[f"fold_{col}_mean_of_means"] = float(fold_df[fold_col].mean())
                agg[f"fold_{col}_std_of_means"] = float(fold_df[fold_col].std())
    return agg


def main() -> None:
    p = argparse.ArgumentParser(description="Aggregate CaMSC k-fold metrics (extended)")
    p.add_argument("--kfold-root", type=str, required=True)
    p.add_argument("--results-root", type=str, default="./results")
    p.add_argument("--name-prefix", type=str, default="camsc_bf_vanilla_fm_fold")
    p.add_argument("--name-suffix", type=str, default="_512")
    p.add_argument("--epoch", type=int, default=80)
    p.add_argument("--k-folds", type=int, default=5)
    p.add_argument("--eval-folds", type=str, default="",
                   help="Comma/space folds to score (default: all 0..k-folds-1)")
    p.add_argument("--out-dir", type=str, default="")
    p.add_argument("--no-lpips", action="store_true")
    args = p.parse_args()

    kfold_root = Path(args.kfold_root).expanduser().resolve()
    results_root = Path(args.results_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else kfold_root / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    lpips_scorer = None if args.no_lpips else _LpipsScorer()
    if lpips_scorer and lpips_scorer.available:
        print(f"LPIPS: on ({lpips_scorer.device})")
    elif not args.no_lpips:
        print("LPIPS: unavailable (install lpips or use --no-lpips)")

    if args.eval_folds.strip():
        fold_list = [int(x) for x in args.eval_folds.replace(",", " ").split() if x.strip()]
    else:
        fold_list = list(range(args.k_folds))

    per_fold = []
    all_tiles = []

    for fold in fold_list:
        name = f"{args.name_prefix}{fold}{args.name_suffix}"
        images_dir = find_test_images_dir(results_root, name, args.epoch)
        if images_dir is None:
            print(f"[warn] fold {fold}: no test images for {name} epoch {args.epoch}")
            continue
        df = score_image_dir(images_dir, lpips_scorer)
        if df.empty:
            print(f"[warn] fold {fold}: empty scores in {images_dir}")
            continue
        df["fold"] = fold
        df.to_csv(out_dir / f"fold{fold}_per_tile.csv", index=False)
        summary = _fold_summary(df, fold, name)
        per_fold.append(summary)
        all_tiles.append(df)
        print(
            f"fold {fold}: n={len(df)} "
            f"Hoechst SSIM={summary['hoechst_ssim_mean']:.4f} "
            f"WT1 SSIM={summary['wt1_ssim_mean']:.4f} "
            f"avg Pearson={summary['average_pearson_mean']:.4f} "
            f"avg Spearman={summary['average_spearman_mean']:.4f}"
        )

    if not per_fold:
        raise SystemExit("No fold results found. Run test for each fold first.")

    fold_df = pd.DataFrame(per_fold)
    fold_df.to_csv(out_dir / "kfold_summary.csv", index=False)

    pooled = pd.concat(all_tiles, ignore_index=True)
    pooled.to_csv(out_dir / "all_folds_per_tile.csv", index=False)

    agg = _pooled_agg(pooled, fold_df)
    agg["epoch"] = args.epoch
    (out_dir / "kfold_aggregate.json").write_text(json.dumps(agg, indent=2) + "\n")

    # Compact leaderboard row for paper tables
    lb_rows = []
    for scope, ch in [("channel", "hoechst"), ("channel", "wt1"), ("average", "mean")]:
        for metric in MAIN_METRICS:
            key = f"{ch}_{metric}" if scope == "channel" else f"average_{metric}"
            if f"{key}_mean" not in agg:
                continue
            lb_rows.append({
                "scope": scope,
                "channel": ch,
                "metric": metric,
                "mean": agg[f"{key}_mean"],
                "std": agg[f"{key}_std"],
            })
    pd.DataFrame(lb_rows).to_csv(out_dir / "leaderboard_main_metrics.csv", index=False)

    print("\n=== Pooled (all test tiles) — main metrics ===")
    for ch in CAMSC_CHANNELS:
        print(
            f"  {ch}: SSIM={agg[f'{ch}_ssim_mean']:.4f}  "
            f"Pearson={agg[f'{ch}_pearson_mean']:.4f}  "
            f"Spearman={agg[f'{ch}_spearman_mean']:.4f}  "
            f"PSNR={agg[f'{ch}_psnr_mean']:.2f}  "
            f"MAE={agg[f'{ch}_mae_mean']:.2f}  "
            f"RMSE={agg[f'{ch}_rmse_mean']:.2f}  "
            f"LPIPS={agg.get(f'{ch}_lpips_mean', float('nan')):.4f}"
        )
    print(
        f"  average: SSIM={agg['average_ssim_mean']:.4f}  "
        f"Pearson={agg['average_pearson_mean']:.4f}  "
        f"Spearman={agg['average_spearman_mean']:.4f}"
    )
    print(f"\nWrote {out_dir}/kfold_summary.csv, kfold_aggregate.json, leaderboard_main_metrics.csv")


if __name__ == "__main__":
    main()
