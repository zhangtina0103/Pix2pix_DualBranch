#!/usr/bin/env python3
"""
WT1-focused quantitative figures for CaMSC (sparse marker).

Story: our model recovers the sparse WT1 marker while GAN baselines collapse
(predict little/no WT1 signal). This script pools ALL k-folds per model, computes
per-tile WT1-channel metrics + positive-pixel-fraction (PPF), and produces:

  fig_wt1_metrics_box.png      per-tile WT1 SSIM / Pearson / Spearman / PSNR boxplots, all models
  fig_wt1_collapse_scatter.png GT WT1 PPF vs predicted WT1 PPF per tile, one panel/model + y=x
  fig_wt1_ppf_bar.png          mean predicted WT1 PPF per model vs GT (collapse-at-a-glance)
  fig_wt1_intensity_scatter.png GT vs predicted mean WT1 intensity per tile, one panel/model
  fig_wt1_summary.csv          pooled WT1 stats per model (for paper table)

Optional (Hoechst, dense reference): pass --with-hoechst to also emit
  fig_hoechst_metrics_box.png

CaMSC target channels: [Hoechst(0), WT1(1), pad(2)]. WT1 is channel index 1.

Example
-------
  python scripts/plot_wt1_figures.py \
    --results-root results --epoch 110 --k-folds 5 \
    --out-dir figures/camsc/wt1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.hemit_eval.extended_metrics import _channel_metrics  # noqa: E402

FONT = "Arial"
WT1_IDX = 1
HOECHST_IDX = 0
OURS_LABEL = "Ours"
OURS_COLOR = "#C62828"
BASE_COLOR = "#90A4AE"
GT_COLOR = "#2E7D32"

DEFAULT_MODELS = [
    ("Ours", "fm_cross_attn_ft"),
    ("Pix2Pix", "pix2pix_ft"),
    ("CUT", "cut_ft"),
    ("ASP", "asp_ft"),
    ("CycleGAN", "cyclegan_ft"),
]


def apply_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [FONT, "Helvetica", "DejaVu Sans"],
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _to_255(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    if arr.min() < 0:
        arr = (arr + 1.0) / 2.0 * 255.0
    elif arr.max() <= 1.0:
        arr = arr * 255.0
    return np.clip(arr, 0, 255)


def load_channel(path: Path, idx: int) -> np.ndarray:
    arr = _to_255(np.asarray(Image.open(path)))
    if arr.ndim == 2:
        return arr
    return arr[..., idx]


def otsu(values: np.ndarray) -> float:
    try:
        from skimage.filters import threshold_otsu
        v = values[np.isfinite(values)]
        if v.size == 0 or v.max() <= v.min():
            return float(np.mean(values))
        return float(threshold_otsu(v))
    except Exception:
        return float(np.mean(values))


def discover(results_root: Path, epoch: int, k_folds: int,
             models: list[tuple[str, str]]) -> dict[str, list[Path]]:
    """model label -> list of image dirs across folds (auto suffix)."""
    out: dict[str, list[Path]] = {}
    for label, key in models:
        dirs = []
        for fold in range(k_folds):
            matches = sorted(results_root.glob(f"camsc_bf_{key}_fold{fold}*/test_{epoch}/images"))
            matches = [m for m in matches if any(m.glob("*_fake_B.tif"))]
            if matches:
                dirs.append(matches[0])
        if dirs:
            out[label] = dirs
            print(f"  [ok] {label:<10} {len(dirs)} fold(s)")
        else:
            print(f"  [MISS] {label:<10} camsc_bf_{key}_fold*/test_{epoch}/images", file=sys.stderr)
    return out


def collect(model_dirs: dict[str, list[Path]], ch_idx: int) -> pd.DataFrame:
    """Per-tile WT1 (or chosen channel) metrics + PPF/intensity, pooled across folds."""
    # global Otsu threshold from all GT tiles of the first model (GT identical across models)
    gt_vals = []
    ref_label = next(iter(model_dirs))
    for d in model_dirs[ref_label]:
        for fp in sorted(d.glob("*_real_B.tif")):
            gt_vals.append(load_channel(fp, ch_idx).ravel())
    thr = otsu(np.concatenate(gt_vals)) if gt_vals else 0.0
    print(f"  global Otsu threshold (ch {ch_idx}) = {thr:.2f}")

    rows = []
    for label, dirs in model_dirs.items():
        for d in dirs:
            for fp in sorted(d.glob("*_fake_B.tif")):
                rp = Path(str(fp).replace("_fake_B.tif", "_real_B.tif"))
                if not rp.is_file():
                    continue
                pred = load_channel(fp, ch_idx)
                real = load_channel(rp, ch_idx)
                m = _channel_metrics(real, pred)
                rows.append({
                    "model": label,
                    "tile": fp.stem.replace("_fake_B", ""),
                    "ssim": m["ssim"], "pearson": m["pearson"],
                    "spearman": m["spearman"], "psnr": m["psnr"], "mae": m["mae"],
                    "gt_ppf": float(np.mean(real > thr)),
                    "pred_ppf": float(np.mean(pred > thr)),
                    "gt_mean": float(np.mean(real)),
                    "pred_mean": float(np.mean(pred)),
                })
    return pd.DataFrame(rows)


def _order(df: pd.DataFrame) -> list[str]:
    labels = list(dict.fromkeys(df["model"]))
    if OURS_LABEL in labels:
        labels = [OURS_LABEL] + [m for m in labels if m != OURS_LABEL]
    return labels


def _palette(labels: list[str]) -> list[str]:
    return [OURS_COLOR if m == OURS_LABEL else BASE_COLOR for m in labels]


METRIC_TITLES = {
    "ssim": "SSIM \u2191", "pearson": "Pearson r \u2191",
    "spearman": "Spearman \u03c1 \u2191", "psnr": "PSNR (dB) \u2191",
    "mae": "MAE \u2193",
}


def fig_metrics_box(df: pd.DataFrame, out_path: Path, dpi: int, channel_name: str,
                    metric_cols: list[str], show_points: bool = False) -> None:
    metrics = [(c, METRIC_TITLES.get(c, c)) for c in metric_cols]
    labels = _order(df)
    colors = _palette(labels)
    n = len(metrics)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.7 * ncol, 4.0 * nrow),
                             facecolor="white", squeeze=False)
    axflat = axes.ravel()
    for k, (col, title) in enumerate(metrics):
        ax = axflat[k]
        data = [df.loc[df["model"] == m, col].dropna().values for m in labels]
        bp = ax.boxplot(data, patch_artist=True, widths=0.6, showfliers=False,
                        medianprops=dict(color="black", linewidth=1.4))
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c); patch.set_alpha(0.85)
        if show_points:
            for i, m in enumerate(labels, start=1):
                y = df.loc[df["model"] == m, col].dropna().values
                x = np.random.normal(i, 0.05, size=len(y))
                ax.scatter(x, y, s=6, color="black", alpha=0.25, zorder=3)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
    for k in range(n, nrow * ncol):
        axflat[k].axis("off")
    fig.suptitle(f"CaMSC {channel_name} per-tile metrics across models",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out_path}")


def fig_collapse_scatter(df: pd.DataFrame, out_path: Path, dpi: int) -> None:
    labels = _order(df)
    n = len(labels)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    hi = float(max(df["gt_ppf"].max(), df["pred_ppf"].max()) * 1.05 + 1e-3)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.4 * nrow),
                             facecolor="white", squeeze=False)
    for idx, m in enumerate(labels):
        ax = axes[idx // ncol][idx % ncol]
        sub = df[df["model"] == m]
        c = OURS_COLOR if m == OURS_LABEL else BASE_COLOR
        ax.plot([0, hi], [0, hi], "--", color="gray", linewidth=1.0, zorder=1)
        ax.scatter(sub["gt_ppf"] * 100, sub["pred_ppf"] * 100, s=14, color=c,
                   alpha=0.7, edgecolor="none", zorder=2)
        track = sub["gt_ppf"].corr(sub["pred_ppf"])
        track_str = "n/a" if not np.isfinite(track) else f"{track:.2f}"
        ax.set_title(f"{m}  (tracking r={track_str})", fontsize=11,
                     fontweight="bold", color=c if m == OURS_LABEL else "black")
        ax.set_xlim(0, hi * 100); ax.set_ylim(0, hi * 100)
        ax.set_xlabel("GT WT1 PPF (%)"); ax.set_ylabel("Pred WT1 PPF (%)")
        ax.grid(alpha=0.3)
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle("WT1 coverage tracking: does predicted PPF follow GT per tile?",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out_path}")


def fig_ppf_bar(df: pd.DataFrame, out_path: Path, dpi: int) -> None:
    labels = _order(df)
    pred = [df.loc[df["model"] == m, "pred_ppf"].mean() * 100 for m in labels]
    err = [df.loc[df["model"] == m, "pred_ppf"].std() * 100 for m in labels]
    gt = float(df["gt_ppf"].mean() * 100)
    colors = _palette(labels)
    fig, ax = plt.subplots(figsize=(8.0, 5.0), facecolor="white")
    x = np.arange(len(labels))
    ax.bar(x, pred, yerr=err, capsize=4, color=colors, alpha=0.9, edgecolor="black", linewidth=0.6)
    ax.axhline(gt, color=GT_COLOR, linestyle="--", linewidth=1.6, label=f"Ground truth ({gt:.1f}%)")
    for xi, p in zip(x, pred):
        ax.text(xi, p + 0.2, f"{p:.1f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Mean WT1 positive-pixel-fraction (%)")
    ax.set_title("Sparse-marker collapse: baselines under-produce WT1 vs GT",
                 fontsize=13, fontweight="bold")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out_path}")


def fig_intensity_scatter(df: pd.DataFrame, out_path: Path, dpi: int) -> None:
    labels = _order(df)
    n = len(labels)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    hi = float(max(df["gt_mean"].max(), df["pred_mean"].max()) * 1.05 + 1e-3)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.4 * nrow),
                             facecolor="white", squeeze=False)
    for idx, m in enumerate(labels):
        ax = axes[idx // ncol][idx % ncol]
        sub = df[df["model"] == m]
        c = OURS_COLOR if m == OURS_LABEL else BASE_COLOR
        ax.plot([0, hi], [0, hi], "--", color="gray", linewidth=1.0, zorder=1)
        ax.scatter(sub["gt_mean"], sub["pred_mean"], s=14, color=c, alpha=0.7,
                   edgecolor="none", zorder=2)
        r = sub["gt_mean"].corr(sub["pred_mean"])
        ax.set_title(f"{m}  (r={r:.2f})", fontsize=11, fontweight="bold",
                     color=c if m == OURS_LABEL else "black")
        ax.set_xlim(0, hi); ax.set_ylim(0, hi)
        ax.set_xlabel("GT mean WT1"); ax.set_ylabel("Pred mean WT1")
        ax.grid(alpha=0.3)
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle("WT1 intensity recovery: predicted vs GT mean intensity per tile",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out_path}")


def write_summary(df: pd.DataFrame, out_path: Path) -> None:
    rows = []
    for m in _order(df):
        sub = df[df["model"] == m]
        rows.append({
            "model": m,
            "n_tiles": len(sub),
            "ssim_mean": sub["ssim"].mean(), "ssim_std": sub["ssim"].std(),
            "pearson_mean": sub["pearson"].mean(), "pearson_std": sub["pearson"].std(),
            "spearman_mean": sub["spearman"].mean(), "spearman_std": sub["spearman"].std(),
            "psnr_mean": sub["psnr"].mean(), "psnr_std": sub["psnr"].std(),
            "mae_mean": sub["mae"].mean(),
            "pred_ppf_mean_pct": sub["pred_ppf"].mean() * 100,
            "gt_ppf_mean_pct": sub["gt_ppf"].mean() * 100,
            "ppf_recovery_pct": sub["pred_ppf"].mean() / max(sub["gt_ppf"].mean(), 1e-9) * 100,
        })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Wrote {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="WT1-focused CaMSC quantitative figures")
    p.add_argument("--results-root", default="results")
    p.add_argument("--epoch", type=int, default=110)
    p.add_argument("--k-folds", type=int, default=5)
    p.add_argument("--model", action="append", default=[], help="Label=key (overrides defaults)")
    p.add_argument("--with-hoechst", action="store_true", help="also emit dense Hoechst metric boxplot")
    p.add_argument("--metrics", default="ssim,pearson,psnr",
                   help="comma list for boxplot panels (ssim,pearson,spearman,psnr,mae)")
    p.add_argument("--intensity-scatter", action="store_true",
                   help="also emit tile-mean intensity scatter (weak metric; off by default)")
    p.add_argument("--show-points", action="store_true",
                   help="overlay per-tile jitter dots on boxplots (off by default)")
    p.add_argument("--out-dir", default="figures/camsc/wt1")
    p.add_argument("--dpi", type=int, default=220)
    args = p.parse_args()
    metric_cols = [m.strip() for m in args.metrics.split(",") if m.strip()]

    apply_style()
    np.random.seed(0)
    results_root = Path(args.results_root).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.model:
        models = []
        for spec in args.model:
            if "=" not in spec:
                raise SystemExit(f"--model must be Label=key, got {spec}")
            lbl, key = spec.split("=", 1)
            models.append((lbl.strip(), key.strip()))
    else:
        models = DEFAULT_MODELS

    print(f"Discovering models under {results_root} (epoch {args.epoch}, {args.k_folds} folds):")
    model_dirs = discover(results_root, args.epoch, args.k_folds, models)
    if not model_dirs:
        raise SystemExit("No model dirs found. Check --results-root/--epoch/--k-folds.")

    print("Collecting WT1 per-tile metrics...")
    df = collect(model_dirs, WT1_IDX)
    if df.empty:
        raise SystemExit("No tiles scored.")
    df.to_csv(out_dir / "wt1_per_tile.csv", index=False)
    print(f"Wrote {out_dir / 'wt1_per_tile.csv'}  ({len(df)} tile-rows)")

    fig_metrics_box(df, out_dir / "fig_wt1_metrics_box.png", args.dpi, "WT1", metric_cols,
                    show_points=args.show_points)
    fig_collapse_scatter(df, out_dir / "fig_wt1_collapse_scatter.png", args.dpi)
    fig_ppf_bar(df, out_dir / "fig_wt1_ppf_bar.png", args.dpi)
    if args.intensity_scatter:
        fig_intensity_scatter(df, out_dir / "fig_wt1_intensity_scatter.png", args.dpi)
    write_summary(df, out_dir / "fig_wt1_summary.csv")

    if args.with_hoechst:
        print("Collecting Hoechst per-tile metrics (dense reference)...")
        dfh = collect(model_dirs, HOECHST_IDX)
        fig_metrics_box(dfh, out_dir / "fig_hoechst_metrics_box.png", args.dpi, "Hoechst", metric_cols,
                        show_points=args.show_points)


if __name__ == "__main__":
    main()
