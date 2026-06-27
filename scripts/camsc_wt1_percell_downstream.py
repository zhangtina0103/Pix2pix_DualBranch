#!/usr/bin/env python3
"""
CaMSC WT1 per-cell downstream validation.

Biological question: does the generated WT1 signal preserve which Hoechst-
segmented cells are WT1-high, not just pixel-level similarity?

For each tile, nuclei are segmented from the ground-truth Hoechst channel. WT1
mean intensity is measured in each nucleus for GT and prediction. A single global
WT1-positive threshold is estimated from pooled GT per-nucleus WT1 means
(global Otsu), then applied to both GT and predictions.

Outputs:
  wt1_percell_per_tile.csv
  wt1_percell_leaderboard.csv
  fig_wt1_percell_box.png
  fig_wt1_expr_fraction_error.png

Example:
  python scripts/camsc_wt1_percell_downstream.py \
    --results-root results --epoch 110 --k-folds 5 \
    --out-dir figures/camsc/downstream_wt1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.hemit_eval.downstream_biology import segment_nuclei  # noqa: E402

HOECHST_IDX = 0
WT1_IDX = 1
FONT = "Arial"
OURS_LABEL = "Ours"
OURS_COLOR = "#C62828"
BASE_COLOR = "#90A4AE"

DEFAULT_MODELS = [
    ("Ours", "fm_cross_attn_ft"),
    ("Pix2Pix", "pix2pix_ft"),
    ("CycleGAN", "cyclegan_ft"),
    ("CUT", "cut_ft"),
    ("ASP", "asp_ft"),
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


def _to_255(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img, dtype=np.float64)
    if arr.min() < 0:
        arr = (arr + 1.0) / 2.0 * 255.0
    elif arr.max() <= 1.0:
        arr = arr * 255.0
    return np.clip(arr, 0, 255)


def load_stack(path: Path) -> np.ndarray:
    arr = _to_255(np.asarray(Image.open(path)))
    if arr.ndim == 2 or arr.shape[-1] < 2:
        raise ValueError(f"{path}: expected >=2-channel CaMSC stack")
    return arr[..., :2]


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if a.size < 3:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt(np.sum(a * a) * np.sum(b * b))
    return float("nan") if denom <= 1e-12 else float(np.sum(a * b) / denom)


def discover(results_root: Path, epoch: int, k_folds: int,
             models: list[tuple[str, str]]) -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = {}
    for label_name, key in models:
        dirs = []
        for fold in range(k_folds):
            matches = sorted(results_root.glob(f"camsc_bf_{key}_fold{fold}*/test_{epoch}/images"))
            matches = [m for m in matches if any(m.glob("*_fake_B.tif"))]
            if matches:
                dirs.append(matches[0])
        if dirs:
            out[label_name] = dirs
            print(f"  [ok] {label_name:<10} {len(dirs)} fold(s)")
        else:
            print(f"  [MISS] {label_name:<10} camsc_bf_{key}_fold*/test_{epoch}/images",
                  file=sys.stderr)
    return out


def collect_gt_threshold(ref_dirs: list[Path], min_area: int) -> float:
    pooled = []
    for image_dir in ref_dirs:
        for real_path in sorted(image_dir.glob("*_real_B.tif")):
            real = load_stack(real_path)
            nuclei = segment_nuclei(real[..., HOECHST_IDX], min_area=min_area)
            lab = label(nuclei)
            if lab.max() == 0:
                continue
            props = regionprops(lab, intensity_image=real[..., WT1_IDX])
            pooled.extend(float(p.mean_intensity) for p in props)
    vals = np.asarray(pooled, dtype=np.float64)
    if vals.size == 0 or vals.max() <= vals.min():
        raise RuntimeError("Could not estimate WT1 threshold: no valid nuclei")
    return float(threshold_otsu(vals))


def score_models(model_dirs: dict[str, list[Path]], threshold: float,
                 min_area: int) -> pd.DataFrame:
    rows = []
    for model, dirs in model_dirs.items():
        for image_dir in dirs:
            for fake_path in sorted(image_dir.glob("*_fake_B.tif")):
                real_path = Path(str(fake_path).replace("_fake_B.tif", "_real_B.tif"))
                if not real_path.is_file():
                    continue
                real = load_stack(real_path)
                fake = load_stack(fake_path)
                nuclei = segment_nuclei(real[..., HOECHST_IDX], min_area=min_area)
                lab = label(nuclei)
                if lab.max() == 0:
                    continue
                real_props = regionprops(lab, intensity_image=real[..., WT1_IDX])
                fake_props = regionprops(lab, intensity_image=fake[..., WT1_IDX])
                real_means = np.asarray([p.mean_intensity for p in real_props], dtype=np.float64)
                fake_means = np.asarray([p.mean_intensity for p in fake_props], dtype=np.float64)
                real_pos = real_means >= threshold
                fake_pos = fake_means >= threshold
                n_nuclei = int(lab.max())
                real_frac = float(np.mean(real_pos))
                fake_frac = float(np.mean(fake_pos))
                rows.append({
                    "model": model,
                    "tile": fake_path.stem.replace("_fake_B", ""),
                    "n_nuclei": n_nuclei,
                    "wt1_percell_pearson": pearson(real_means, fake_means),
                    "wt1_expr_fraction_real": real_frac,
                    "wt1_expr_fraction_gen": fake_frac,
                    "wt1_expr_fraction_abs_err": abs(fake_frac - real_frac),
                    "wt1_count_real": int(real_pos.sum()),
                    "wt1_count_gen": int(fake_pos.sum()),
                    "wt1_count_abs_err": abs(int(fake_pos.sum()) - int(real_pos.sum())),
                })
    return pd.DataFrame(rows)


def order_models(labels: list[str]) -> list[str]:
    default = [m for m, _ in DEFAULT_MODELS]
    ordered = [m for m in default if m in labels]
    return ordered + [m for m in labels if m not in ordered]


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in order_models(list(dict.fromkeys(df["model"]))):
        sub = df[df["model"] == model]
        rows.append({
            "model": model,
            "n_tiles": len(sub),
            "n_nuclei_total": int(sub["n_nuclei"].sum()),
            "wt1_percell_pearson_mean": sub["wt1_percell_pearson"].mean(),
            "wt1_percell_pearson_std": sub["wt1_percell_pearson"].std(),
            "wt1_expr_fraction_abs_err_mean": sub["wt1_expr_fraction_abs_err"].mean(),
            "wt1_expr_fraction_abs_err_std": sub["wt1_expr_fraction_abs_err"].std(),
            "wt1_count_abs_err_mean": sub["wt1_count_abs_err"].mean(),
            "wt1_count_abs_err_std": sub["wt1_count_abs_err"].std(),
            "wt1_expr_fraction_real_mean": sub["wt1_expr_fraction_real"].mean(),
            "wt1_expr_fraction_gen_mean": sub["wt1_expr_fraction_gen"].mean(),
        })
    return pd.DataFrame(rows)


def colors(labels: list[str]) -> list[str]:
    return [OURS_COLOR if m == OURS_LABEL else BASE_COLOR for m in labels]


def boxplot(df: pd.DataFrame, col: str, ylabel: str, title: str, out_path: Path, dpi: int) -> None:
    labels = order_models(list(dict.fromkeys(df["model"])))
    data = [df.loc[df["model"] == m, col].dropna().values for m in labels]
    fig, ax = plt.subplots(figsize=(7.5, 4.6), facecolor="white")
    bp = ax.boxplot(data, patch_artist=True, widths=0.6, showfliers=False,
                    medianprops=dict(color="black", linewidth=1.4))
    for patch, c in zip(bp["boxes"], colors(labels)):
        patch.set_facecolor(c)
        patch.set_alpha(0.9)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="CaMSC WT1 per-cell downstream validation")
    p.add_argument("--results-root", default="results")
    p.add_argument("--epoch", type=int, default=110)
    p.add_argument("--k-folds", type=int, default=5)
    p.add_argument("--model", action="append", default=[], help="Label=key (overrides defaults)")
    p.add_argument("--min-area", type=int, default=36)
    p.add_argument("--out-dir", default="figures/camsc/downstream_wt1")
    p.add_argument("--dpi", type=int, default=220)
    args = p.parse_args()

    apply_style()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    models = []
    if args.model:
        for spec in args.model:
            if "=" not in spec:
                raise SystemExit(f"--model must be Label=key, got {spec}")
            label_name, key = spec.split("=", 1)
            models.append((label_name.strip(), key.strip()))
    else:
        models = DEFAULT_MODELS

    print(f"Discovering models under {args.results_root}...")
    model_dirs = discover(Path(args.results_root).expanduser(), args.epoch, args.k_folds, models)
    if not model_dirs:
        raise SystemExit("No model result dirs found")

    ref_label = next(iter(model_dirs))
    threshold = collect_gt_threshold(model_dirs[ref_label], args.min_area)
    print(f"WT1 global per-nucleus threshold (GT Otsu): {threshold:.2f}")
    df = score_models(model_dirs, threshold, args.min_area)
    if df.empty:
        raise SystemExit("No WT1 per-cell rows scored")

    df.to_csv(out_dir / "wt1_percell_per_tile.csv", index=False)
    leaderboard = summarize(df)
    leaderboard.to_csv(out_dir / "wt1_percell_leaderboard.csv", index=False)
    (out_dir / "wt1_percell_summary.json").write_text(
        json.dumps({"wt1_threshold": threshold, "n_rows": len(df)}, indent=2) + "\n",
        encoding="utf-8",
    )
    boxplot(
        df, "wt1_percell_pearson", "Per-cell WT1 Pearson r",
        "CaMSC WT1 per-cell intensity preservation", out_dir / "fig_wt1_percell_box.png", args.dpi,
    )
    boxplot(
        df, "wt1_expr_fraction_abs_err", "|predicted - GT| WT1+ cell fraction",
        "CaMSC WT1-positive cell fraction error", out_dir / "fig_wt1_expr_fraction_error.png",
        args.dpi,
    )

    print("\n=== WT1 per-cell downstream ===")
    for _, row in leaderboard.iterrows():
        print(
            f"{row['model']:<10} r={row['wt1_percell_pearson_mean']:.3f} "
            f"frac_err={row['wt1_expr_fraction_abs_err_mean']:.3f} "
            f"count_err={row['wt1_count_abs_err_mean']:.2f}"
        )


if __name__ == "__main__":
    main()
