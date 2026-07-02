#!/usr/bin/env python3
"""
CaMSC WT1 extended downstream validation.

Segments nuclei from GT Hoechst, then scores WT1 preservation with multiple
biologically motivated metrics (not only per-cell Pearson / ROC-AUC):

  • Per-cell Pearson / Spearman (all tiles)
  • WT1+ cell fraction & count calibration (MAE, MAE ratio)
  • WT1+ recall / precision (sensitivity to sparse positives)
  • Low-expression WT1+ recall (bottom tertile among GT WT1+ nuclei)
  • Nucleus-patch WT1 SSIM & masked MAE (spatial fidelity inside nuclei)
  • Subsets: all tiles | WT1-positive tiles | WT1-enriched tiles (top fraction)

Outputs (under --out-dir):
  wt1_percell_per_tile.csv
  wt1_percell_leaderboard.csv          # all tiles
  wt1_subset_leaderboard.csv           # all / wt1_positive / wt1_enriched
  wt1_cell_classification_leaderboard.csv
  wt1_percell_summary.json
  fig_wt1_percell_box.png
  fig_wt1_expr_fraction_error.png
  fig_wt1_cell_auc.png, fig_wt1_cell_f1.png
  fig_wt1_pos_recall.png
  fig_wt1_nucleus_ssim.png
  fig_wt1_enriched_percell.png

Example:
  python scripts/camsc_wt1_percell_downstream.py \\
    --results-root results --epoch 110 --k-folds 5 \\
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
from skimage.metrics import structural_similarity as ssim

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
EPS = 1e-3

DEFAULT_MODELS = [
    ("Ours", "fm_cross_attn_ft"),
    ("Pix2Pix", "pix2pix_ft"),
    ("CycleGAN", "cyclegan_ft"),
    ("CUT", "cut_ft"),
    ("ASP", "asp_ft"),
]

# Per-tile metrics aggregated in leaderboards (higher_is_better unless noted).
TILE_METRICS = [
    ("wt1_percell_pearson", True),
    ("wt1_percell_spearman", True),
    ("wt1_expr_fraction_abs_err", False),
    ("wt1_expr_fraction_mae_ratio", False),
    ("wt1_count_abs_err", False),
    ("wt1_pos_recall", True),
    ("wt1_pos_precision", True),
    ("wt1_pos_f1", True),
    ("wt1_low_expr_recall", True),
    ("nucleus_wt1_ssim_mean", True),
    ("nucleus_wt1_mae_mean", False),
]

METRIC_LABELS = {
    "wt1_percell_pearson": "Per-cell Pearson r",
    "wt1_percell_spearman": "Per-cell Spearman ρ",
    "wt1_expr_fraction_abs_err": "|pred − GT| WT1+ fraction",
    "wt1_expr_fraction_mae_ratio": "WT1+ fraction MAE ratio",
    "wt1_count_abs_err": "|pred − GT| WT1+ count",
    "wt1_pos_recall": "WT1+ recall",
    "wt1_pos_precision": "WT1+ precision",
    "wt1_pos_f1": "WT1+ F1",
    "wt1_low_expr_recall": "Low-expr WT1+ recall",
    "nucleus_wt1_ssim_mean": "Nucleus-patch WT1 SSIM",
    "nucleus_wt1_mae_mean": "Nucleus-mask WT1 MAE",
    "cell_roc_auc": "Cell-level ROC-AUC",
    "cell_wt1_pos_f1": "WT1+ F1 (pooled Otsu)",
    "cell_f1_best_threshold": "WT1+ F1 (best threshold)",
    "cell_wt1_pos_recall": "Pooled WT1+ recall",
    "cell_low_expr_recall": "Pooled low-expr recall",
}


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


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    try:
        from scipy.stats import spearmanr
        r, _ = spearmanr(a, b)
        return float(r)
    except Exception:
        ar = np.argsort(np.argsort(a))
        br = np.argsort(np.argsort(b))
        return pearson(ar.astype(np.float64), br.astype(np.float64))


def wt1_enrichment_threshold(mean_scores: np.ndarray, top_frac: float) -> float:
    scores = np.asarray(mean_scores, dtype=np.float64)
    if scores.size == 0:
        return float("nan")
    if not 0.0 < top_frac < 1.0:
        raise ValueError(f"top_frac must be in (0, 1), got {top_frac}")
    return float(np.percentile(scores, (1.0 - top_frac) * 100.0))


def _mae_ratio(p_real: float, p_gen: float, eps: float = EPS) -> float:
    if not np.isfinite(p_real) or not np.isfinite(p_gen):
        return float("nan")
    return float(abs(p_real - p_gen) / max(abs(p_real), eps))


def _pos_recall_precision(
    real_means: np.ndarray, fake_means: np.ndarray, threshold: float,
) -> tuple[float, float]:
    real_pos = real_means >= threshold
    fake_pos = fake_means >= threshold
    if not np.any(real_pos):
        recall = float("nan")
    else:
        recall = float(np.mean(fake_pos[real_pos]))
    if not np.any(fake_pos):
        precision = float("nan")
    else:
        precision = float(np.mean(real_pos[fake_pos]))
    return recall, precision


def _f1_from_recall_precision(recall: float, precision: float) -> float:
    if not np.isfinite(recall) or not np.isfinite(precision):
        return float("nan")
    denom = recall + precision
    if denom <= 0:
        return 0.0
    return float(2.0 * recall * precision / denom)


def _pooled_classification_at_threshold(
    real: np.ndarray, fake: np.ndarray, threshold: float,
) -> tuple[float, float, float]:
    """Recall, precision, F1 on all pooled nuclei at a fixed intensity threshold."""
    y = real >= threshold
    pred = fake >= threshold
    tp = int(np.sum(pred & y))
    fp = int(np.sum(pred & ~y))
    fn = int(np.sum(~pred & y))
    recall = float(tp / (tp + fn)) if (tp + fn) else float("nan")
    precision = float(tp / (tp + fp)) if (tp + fp) else float("nan")
    f1 = _f1_from_recall_precision(recall, precision)
    return recall, precision, f1


def _low_expr_recall(
    real_means: np.ndarray, fake_means: np.ndarray, threshold: float,
    low_pct: float = 33.33,
) -> float:
    """Recall on bottom tertile of GT WT1+ nuclei (weak-signal recovery)."""
    pos = real_means >= threshold
    if int(np.sum(pos)) < 3:
        return float("nan")
    pos_real = real_means[pos]
    pos_fake = fake_means[pos]
    low_thr = float(np.percentile(pos_real, low_pct))
    low_mask = pos_real <= low_thr
    if not np.any(low_mask):
        return float("nan")
    return float(np.mean(pos_fake[low_mask] >= threshold))


def _nucleus_wt1_structural(
    real_wt1: np.ndarray, fake_wt1: np.ndarray, lab: np.ndarray, min_area: int,
) -> tuple[float, float]:
    """Mean SSIM (bbox) and masked MAE over nuclei in a tile."""
    ssims: list[float] = []
    maes: list[float] = []
    for prop in regionprops(lab):
        if prop.area < min_area:
            continue
        mask = lab == prop.label
        ymin, xmin, ymax, xmax = prop.bbox
        pad = 3
        y0 = max(0, ymin - pad)
        x0 = max(0, xmin - pad)
        y1 = min(real_wt1.shape[0], ymax + pad)
        x1 = min(real_wt1.shape[1], xmax + pad)
        r = real_wt1[y0:y1, x0:x1].astype(np.float64)
        f = fake_wt1[y0:y1, x0:x1].astype(np.float64)
        m = mask[y0:y1, x0:x1]
        if not np.any(m):
            continue
        maes.append(float(np.mean(np.abs(r[m] - f[m]))))
        h, w = r.shape
        win = min(7, h, w)
        if win % 2 == 0:
            win -= 1
        if win < 3 or r.size < win * win:
            continue
        try:
            ssims.append(float(ssim(r, f, data_range=255.0, win_size=win)))
        except ValueError:
            pass
    return (
        float(np.mean(ssims)) if ssims else float("nan"),
        float(np.mean(maes)) if maes else float("nan"),
    )


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


def score_tile(
    real: np.ndarray, fake: np.ndarray, threshold: float, min_area: int,
) -> dict[str, float | int | bool]:
    nuclei = segment_nuclei(real[..., HOECHST_IDX], min_area=min_area)
    lab = label(nuclei)
    if lab.max() == 0:
        nan = float("nan")
        return {
            "n_nuclei": 0,
            "wt1_mean_real": float(np.mean(real[..., WT1_IDX])),
            "wt1_percell_pearson": nan,
            "wt1_percell_spearman": nan,
            "wt1_expr_fraction_real": nan,
            "wt1_expr_fraction_gen": nan,
            "wt1_expr_fraction_abs_err": nan,
            "wt1_expr_fraction_mae_ratio": nan,
            "wt1_count_real": 0,
            "wt1_count_gen": 0,
            "wt1_count_abs_err": 0,
            "wt1_pos_recall": nan,
            "wt1_pos_precision": nan,
            "wt1_pos_f1": nan,
            "wt1_low_expr_recall": nan,
            "nucleus_wt1_ssim_mean": nan,
            "nucleus_wt1_mae_mean": nan,
            "is_wt1_positive_tile": False,
            "is_wt1_enriched_tile": False,
        }

    real_props = regionprops(lab, intensity_image=real[..., WT1_IDX])
    fake_props = regionprops(lab, intensity_image=fake[..., WT1_IDX])
    real_means = np.asarray([p.mean_intensity for p in real_props], dtype=np.float64)
    fake_means = np.asarray([p.mean_intensity for p in fake_props], dtype=np.float64)
    real_pos = real_means >= threshold
    fake_pos = fake_means >= threshold
    n_nuclei = int(lab.max())
    real_frac = float(np.mean(real_pos))
    fake_frac = float(np.mean(fake_pos))
    recall, precision = _pos_recall_precision(real_means, fake_means, threshold)
    nuc_ssim, nuc_mae = _nucleus_wt1_structural(
        real[..., WT1_IDX], fake[..., WT1_IDX], lab, min_area,
    )
    return {
        "n_nuclei": n_nuclei,
        "wt1_mean_real": float(np.mean(real[..., WT1_IDX])),
        "wt1_percell_pearson": pearson(real_means, fake_means),
        "wt1_percell_spearman": spearman(real_means, fake_means),
        "wt1_expr_fraction_real": real_frac,
        "wt1_expr_fraction_gen": fake_frac,
        "wt1_expr_fraction_abs_err": abs(fake_frac - real_frac),
        "wt1_expr_fraction_mae_ratio": _mae_ratio(real_frac, fake_frac),
        "wt1_count_real": int(real_pos.sum()),
        "wt1_count_gen": int(fake_pos.sum()),
        "wt1_count_abs_err": abs(int(fake_pos.sum()) - int(real_pos.sum())),
        "wt1_pos_recall": recall,
        "wt1_pos_precision": precision,
        "wt1_pos_f1": _f1_from_recall_precision(recall, precision),
        "wt1_low_expr_recall": _low_expr_recall(real_means, fake_means, threshold),
        "nucleus_wt1_ssim_mean": nuc_ssim,
        "nucleus_wt1_mae_mean": nuc_mae,
        "is_wt1_positive_tile": int(real_pos.sum()) > 0,
        "is_wt1_enriched_tile": False,
    }


def score_models(
    model_dirs: dict[str, list[Path]], threshold: float, min_area: int,
    enrichment_top_frac: float,
) -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]]]:
    rows = []
    cells: dict[str, dict[str, list]] = {}
    for model, dirs in model_dirs.items():
        cells[model] = {"real_mean": [], "fake_mean": []}
        for image_dir in dirs:
            for fake_path in sorted(image_dir.glob("*_fake_B.tif")):
                real_path = Path(str(fake_path).replace("_fake_B.tif", "_real_B.tif"))
                if not real_path.is_file():
                    continue
                real = load_stack(real_path)
                fake = load_stack(fake_path)
                tile = fake_path.stem.replace("_fake_B", "")
                metrics = score_tile(real, fake, threshold, min_area)
                nuclei = segment_nuclei(real[..., HOECHST_IDX], min_area=min_area)
                lab = label(nuclei)
                if lab.max() > 0:
                    real_props = regionprops(lab, intensity_image=real[..., WT1_IDX])
                    fake_props = regionprops(lab, intensity_image=fake[..., WT1_IDX])
                    cells[model]["real_mean"].append(
                        np.asarray([p.mean_intensity for p in real_props], dtype=np.float64),
                    )
                    cells[model]["fake_mean"].append(
                        np.asarray([p.mean_intensity for p in fake_props], dtype=np.float64),
                    )
                rows.append({"model": model, "tile": tile, **metrics})

    df = pd.DataFrame(rows)
    if df.empty:
        pooled: dict[str, dict[str, np.ndarray]] = {}
    else:
        enrich_thr = wt1_enrichment_threshold(
            df.loc[df["model"] == df["model"].iloc[0], "wt1_mean_real"].to_numpy(),
            enrichment_top_frac,
        )
        df["is_wt1_enriched_tile"] = df["wt1_mean_real"] >= enrich_thr
        df["wt1_enrichment_threshold"] = enrich_thr
        pooled = {
            m: {
                "real_mean": np.concatenate(d["real_mean"]) if d["real_mean"] else np.array([]),
                "fake_mean": np.concatenate(d["fake_mean"]) if d["fake_mean"] else np.array([]),
            }
            for m, d in cells.items()
        }
    return df, pooled


def _roc_auc(y: np.ndarray, score: np.ndarray) -> float:
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    s = score[order]
    ranks = np.empty(score.size, dtype=np.float64)
    i = 0
    while i < s.size:
        j = i
        while j + 1 < s.size and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    sum_ranks_pos = float(np.sum(ranks[y == 1]))
    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _average_precision(y: np.ndarray, score: np.ndarray) -> float:
    if y.sum() == 0:
        return float("nan")
    order = np.argsort(-score, kind="mergesort")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / y.sum()
    recall_prev = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - recall_prev) * precision))


def _best_f1(y: np.ndarray, score: np.ndarray) -> tuple[float, float, float]:
    thresholds = np.unique(score[np.isfinite(score)])
    if thresholds.size == 0:
        return float("nan"), float("nan"), float("nan")
    best = (float("nan"), -1.0, -1.0)
    for thr in thresholds:
        pred = score >= thr
        tp = int(np.sum(pred & (y == 1)))
        fp = int(np.sum(pred & (y == 0)))
        fn = int(np.sum(~pred & (y == 1)))
        tn = int(np.sum(~pred & (y == 0)))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        bal = 0.5 * (recall + spec)
        if bal > best[2] or (np.isclose(bal, best[2]) and f1 > best[1]):
            best = (float(thr), float(f1), float(bal))
    return best


def cell_classification(pooled: dict[str, dict[str, np.ndarray]], threshold: float) -> pd.DataFrame:
    rows = []
    for model in order_models(list(pooled.keys())):
        real = pooled[model]["real_mean"]
        fake = pooled[model]["fake_mean"]
        if real.size == 0:
            continue
        y = (real >= threshold).astype(int)
        otsu_recall, otsu_precision, otsu_f1 = _pooled_classification_at_threshold(
            real, fake, threshold,
        )
        thr, f1_best, bal = _best_f1(y, fake)
        pooled_low = _low_expr_recall(real, fake, threshold)
        rows.append({
            "model": model,
            "n_cells": int(real.size),
            "n_wt1_pos_cells": int(y.sum()),
            "wt1_pos_cell_fraction": float(y.mean()),
            "cell_roc_auc": _roc_auc(y, fake),
            "cell_average_precision": _average_precision(y, fake),
            "cell_best_threshold": thr,
            "cell_f1_best_threshold": f1_best,
            "cell_balanced_accuracy": bal,
            # Pooled at global GT Otsu threshold (matches per-tile recall/precision definition).
            "cell_wt1_pos_recall": otsu_recall,
            "cell_wt1_pos_precision": otsu_precision,
            "cell_wt1_pos_f1": otsu_f1,
            "cell_low_expr_recall": pooled_low,
        })
    return pd.DataFrame(rows)


def build_paper_main_table(df: pd.DataFrame, cell_lb: pd.DataFrame) -> pd.DataFrame:
    """Main-text table: tile-mean recall/precision/F1/SSIM/Pearson + pooled Otsu F1."""
    rows = []
    for model in order_models(list(dict.fromkeys(df["model"]))):
        sub = df[df["model"] == model]
        cell = cell_lb[cell_lb["model"] == model]
        rows.append({
            "model": model,
            "n_tiles": len(sub),
            "wt1_recall_tile_mean": float(sub["wt1_pos_recall"].mean()),
            "wt1_precision_tile_mean": float(sub["wt1_pos_precision"].mean()),
            "wt1_f1_tile_mean": float(sub["wt1_pos_f1"].mean()),
            "nucleus_wt1_ssim_tile_mean": float(sub["nucleus_wt1_ssim_mean"].mean()),
            "percell_pearson_tile_mean": float(sub["wt1_percell_pearson"].mean()),
            "wt1_recall_pooled_otsu": float(cell["cell_wt1_pos_recall"].iloc[0]) if len(cell) else float("nan"),
            "wt1_precision_pooled_otsu": float(cell["cell_wt1_pos_precision"].iloc[0]) if len(cell) else float("nan"),
            "wt1_f1_pooled_otsu": float(cell["cell_wt1_pos_f1"].iloc[0]) if len(cell) else float("nan"),
            "cell_roc_auc": float(cell["cell_roc_auc"].iloc[0]) if len(cell) else float("nan"),
            "cell_f1_best_threshold": float(cell["cell_f1_best_threshold"].iloc[0]) if len(cell) else float("nan"),
        })
    return pd.DataFrame(rows)


def order_models(labels: list[str]) -> list[str]:
    default = [m for m, _ in DEFAULT_MODELS]
    ordered = [m for m in default if m in labels]
    return ordered + [m for m in labels if m not in ordered]


def summarize_tiles(df: pd.DataFrame, subset: str | None = None) -> pd.DataFrame:
    """Aggregate per-tile metrics; optional subset filter column."""
    if subset == "wt1_positive":
        sub_df = df[df["is_wt1_positive_tile"]]
        scope = "wt1_positive_tiles"
    elif subset == "wt1_enriched":
        sub_df = df[df["is_wt1_enriched_tile"]]
        scope = "wt1_enriched_tiles"
    else:
        sub_df = df
        scope = "all_tiles"

    rows = []
    for model in order_models(list(dict.fromkeys(df["model"]))):
        sub = sub_df[sub_df["model"] == model]
        row: dict[str, object] = {
            "model": model,
            "scope": scope,
            "n_tiles": len(sub),
            "n_nuclei_total": int(sub["n_nuclei"].sum()) if len(sub) else 0,
        }
        for col, _higher in TILE_METRICS:
            row[f"{col}_mean"] = float(sub[col].mean()) if len(sub) else float("nan")
            row[f"{col}_std"] = float(sub[col].std()) if len(sub) > 1 else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def build_subset_leaderboard(df: pd.DataFrame) -> pd.DataFrame:
    parts = [
        summarize_tiles(df, None),
        summarize_tiles(df, "wt1_positive"),
        summarize_tiles(df, "wt1_enriched"),
    ]
    return pd.concat(parts, ignore_index=True)


def colors(labels: list[str]) -> list[str]:
    return [OURS_COLOR if m == OURS_LABEL else BASE_COLOR for m in labels]


def boxplot(df: pd.DataFrame, col: str, ylabel: str, title: str, out_path: Path, dpi: int,
            subset_col: str | None = None, subset_val: bool | None = None) -> None:
    plot_df = df
    if subset_col is not None and subset_val is not None:
        plot_df = df[df[subset_col] == subset_val]
    labels = order_models(list(dict.fromkeys(plot_df["model"])))
    data = [plot_df.loc[plot_df["model"] == m, col].dropna().values for m in labels]
    fig, ax = plt.subplots(figsize=(7.5, 4.6), facecolor="white")
    bp = ax.boxplot(data, patch_artist=True, widths=0.6, showfliers=False,
                    medianprops=dict(color="black", linewidth=1.4))
    for patch, c in zip(bp["boxes"], colors(labels)):
        patch.set_facecolor(c)
        patch.set_alpha(0.9)
        lw = 2.0 if c == OURS_COLOR else 0.8
        patch.set_linewidth(lw)
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


def bar_metric(leaderboard: pd.DataFrame, col: str, ylabel: str, title: str,
               out_path: Path, dpi: int, chance: float | None = None,
               ylim_top: float = 1.05) -> None:
    labels = leaderboard["model"].tolist()
    vals = leaderboard[col].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7.0, 4.6), facecolor="white")
    bars = ax.bar(np.arange(len(labels)), vals, color=colors(labels),
                  edgecolor="black", linewidth=0.6)
    if chance is not None:
        ax.axhline(chance, color="gray", linestyle="--", linewidth=1.2,
                   label=f"chance ({chance:.2f})")
        ax.legend(frameon=False)
    for bar, val in zip(bars, vals):
        if np.isfinite(val):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.015, f"{val:.3f}",
                    ha="center", va="bottom", fontsize=10)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0, ylim_top)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out_path}")


def print_leaderboard_table(
    subset_lb: pd.DataFrame, cell_lb: pd.DataFrame, enrichment_top_frac: float,
) -> None:
    scopes = ["all_tiles", "wt1_positive_tiles", "wt1_enriched_tiles"]
    scope_titles = {
        "all_tiles": "All tiles",
        "wt1_positive_tiles": "WT1+ tiles (≥1 WT1+ nucleus)",
        "wt1_enriched_tiles": f"WT1-enriched (top {enrichment_top_frac:.0%} mean WT1)",
    }
    key_metrics = [
        "wt1_percell_pearson", "wt1_pos_recall", "wt1_pos_precision", "wt1_pos_f1",
        "wt1_low_expr_recall", "nucleus_wt1_ssim_mean",
        "wt1_expr_fraction_abs_err", "wt1_count_abs_err",
    ]
    for scope in scopes:
        block = subset_lb[subset_lb["scope"] == scope]
        if block.empty:
            continue
        print(f"\n=== {scope_titles[scope]} (n_tiles per model) ===")
        for _, row in block.iterrows():
            parts = [f"{row['model']:<10} n={int(row['n_tiles'])}"]
            for m in key_metrics:
                val = row.get(f"{m}_mean", float("nan"))
                if np.isfinite(val):
                    short = {
                        "wt1_percell_pearson": "pearson",
                        "wt1_pos_recall": "recall",
                        "wt1_pos_precision": "precision",
                        "wt1_pos_f1": "f1",
                        "wt1_low_expr_recall": "low_recall",
                        "nucleus_wt1_ssim_mean": "nuc_ssim",
                        "wt1_expr_fraction_abs_err": "frac_err",
                        "wt1_count_abs_err": "count_err",
                    }[m]
                    parts.append(f"{short}={val:.3f}")
            print("  " + "  ".join(parts))

    if not cell_lb.empty:
        print("\n=== Pooled nuclei @ global Otsu threshold ===")
        for _, row in cell_lb.iterrows():
            print(
                f"{row['model']:<10} recall={row['cell_wt1_pos_recall']:.3f} "
                f"precision={row['cell_wt1_pos_precision']:.3f} "
                f"F1={row['cell_wt1_pos_f1']:.3f} "
                f"AUC={row['cell_roc_auc']:.3f} "
                f"low_expr_recall={row['cell_low_expr_recall']:.3f}"
            )
        print("\n=== Pooled F1 at score-optimized threshold (supplement) ===")
        for _, row in cell_lb.iterrows():
            print(
                f"{row['model']:<10} F1_best={row['cell_f1_best_threshold']:.3f} "
                f"(thr={row['cell_best_threshold']:.1f})"
            )


def print_paper_main_table(paper: pd.DataFrame) -> None:
    print("\n=== MAIN PAPER TABLE (tile-mean recall/precision/F1/SSIM/Pearson) ===")
    header = (
        f"{'Model':<10} {'Recall':>7} {'Prec':>7} {'F1':>7} "
        f"{'NucSSIM':>8} {'Pearson':>8}"
    )
    print(header)
    print("-" * len(header))
    for _, row in paper.iterrows():
        def f(v: float) -> str:
            return f"{v:.3f}" if np.isfinite(v) else "—"
        print(
            f"{row['model']:<10} "
            f"{f(row['wt1_recall_tile_mean']):>7} "
            f"{f(row['wt1_precision_tile_mean']):>7} "
            f"{f(row['wt1_f1_tile_mean']):>7} "
            f"{f(row['nucleus_wt1_ssim_tile_mean']):>8} "
            f"{f(row['percell_pearson_tile_mean']):>8}"
        )


def main() -> None:
    p = argparse.ArgumentParser(description="CaMSC WT1 extended downstream validation")
    p.add_argument("--results-root", default="results")
    p.add_argument("--epoch", type=int, default=110)
    p.add_argument("--k-folds", type=int, default=5)
    p.add_argument("--model", action="append", default=[], help="Label=key (overrides defaults)")
    p.add_argument("--min-area", type=int, default=36)
    p.add_argument("--wt1-enrichment-top-frac", type=float, default=0.10,
                   help="Top fraction of tiles by mean GT WT1 for enriched subset")
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

    df, pooled = score_models(
        model_dirs, threshold, args.min_area, args.wt1_enrichment_top_frac,
    )
    if df.empty:
        raise SystemExit("No WT1 per-cell rows scored")

    enrich_thr = float(df["wt1_enrichment_threshold"].iloc[0])
    print(f"WT1 enrichment threshold (top {args.wt1_enrichment_top_frac:.0%}): {enrich_thr:.2f}")

    df.to_csv(out_dir / "wt1_percell_per_tile.csv", index=False)
    leaderboard = summarize_tiles(df, None)
    leaderboard.to_csv(out_dir / "wt1_percell_leaderboard.csv", index=False)

    subset_lb = build_subset_leaderboard(df)
    subset_lb.to_csv(out_dir / "wt1_subset_leaderboard.csv", index=False)

    cell_lb = cell_classification(pooled, threshold)
    cell_lb.to_csv(out_dir / "wt1_cell_classification_leaderboard.csv", index=False)

    paper_table = build_paper_main_table(df, cell_lb)
    paper_table.to_csv(out_dir / "wt1_paper_main_table.csv", index=False)

    summary = {
        "wt1_threshold": threshold,
        "wt1_enrichment_top_frac": args.wt1_enrichment_top_frac,
        "wt1_enrichment_threshold": enrich_thr,
        "n_rows": len(df),
        "n_wt1_positive_tiles": int(df.groupby("model")["is_wt1_positive_tile"].sum().iloc[0]),
        "n_wt1_enriched_tiles": int(df.groupby("model")["is_wt1_enriched_tile"].sum().iloc[0]),
    }
    (out_dir / "wt1_percell_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8",
    )

    # Figures — original + new
    boxplot(
        df, "wt1_percell_pearson", METRIC_LABELS["wt1_percell_pearson"],
        "CaMSC WT1 per-cell Pearson (all tiles)",
        out_dir / "fig_wt1_percell_box.png", args.dpi,
    )
    boxplot(
        df, "wt1_expr_fraction_abs_err", METRIC_LABELS["wt1_expr_fraction_abs_err"],
        "CaMSC WT1+ cell fraction error (all tiles)",
        out_dir / "fig_wt1_expr_fraction_error.png", args.dpi,
    )
    boxplot(
        df, "wt1_pos_recall", METRIC_LABELS["wt1_pos_recall"],
        "CaMSC WT1+ recall per tile",
        out_dir / "fig_wt1_pos_recall.png", args.dpi,
    )
    boxplot(
        df, "wt1_pos_precision", METRIC_LABELS["wt1_pos_precision"],
        "CaMSC WT1+ precision per tile",
        out_dir / "fig_wt1_pos_precision.png", args.dpi,
    )
    boxplot(
        df, "wt1_pos_f1", METRIC_LABELS["wt1_pos_f1"],
        "CaMSC WT1+ F1 per tile (global Otsu threshold)",
        out_dir / "fig_wt1_pos_f1.png", args.dpi,
    )
    boxplot(
        df, "wt1_low_expr_recall", METRIC_LABELS["wt1_low_expr_recall"],
        "CaMSC low-expression WT1+ recall per tile",
        out_dir / "fig_wt1_low_expr_recall.png", args.dpi,
    )
    boxplot(
        df, "nucleus_wt1_ssim_mean", METRIC_LABELS["nucleus_wt1_ssim_mean"],
        "CaMSC nucleus-patch WT1 SSIM per tile",
        out_dir / "fig_wt1_nucleus_ssim.png", args.dpi,
    )
    boxplot(
        df, "wt1_percell_pearson", METRIC_LABELS["wt1_percell_pearson"],
        f"CaMSC WT1 per-cell Pearson (WT1-enriched top {args.wt1_enrichment_top_frac:.0%})",
        out_dir / "fig_wt1_enriched_percell.png", args.dpi,
        subset_col="is_wt1_enriched_tile", subset_val=True,
    )
    boxplot(
        df, "wt1_count_abs_err", METRIC_LABELS["wt1_count_abs_err"],
        f"CaMSC |pred−GT| WT1+ count (WT1-enriched top {args.wt1_enrichment_top_frac:.0%})",
        out_dir / "fig_wt1_enriched_count_err.png", args.dpi,
        subset_col="is_wt1_enriched_tile", subset_val=True,
    )

    if not cell_lb.empty:
        bar_metric(
            cell_lb, "cell_roc_auc", METRIC_LABELS["cell_roc_auc"],
            "CaMSC WT1+ cell classification (AUC)",
            out_dir / "fig_wt1_cell_auc.png", args.dpi, chance=0.5,
        )
        bar_metric(
            cell_lb, "cell_wt1_pos_f1", METRIC_LABELS["wt1_pos_f1"],
            "CaMSC WT1+ F1 (pooled, global Otsu threshold)",
            out_dir / "fig_wt1_cell_f1_otsu.png", args.dpi,
        )
        bar_metric(
            cell_lb, "cell_wt1_pos_recall", METRIC_LABELS["cell_wt1_pos_recall"],
            "CaMSC pooled WT1+ recall",
            out_dir / "fig_wt1_pooled_recall.png", args.dpi,
        )
        bar_metric(
            cell_lb, "cell_low_expr_recall", METRIC_LABELS["cell_low_expr_recall"],
            "CaMSC pooled low-expression WT1+ recall",
            out_dir / "fig_wt1_pooled_low_expr_recall.png", args.dpi,
        )

    print_leaderboard_table(subset_lb, cell_lb, args.wt1_enrichment_top_frac)
    print_paper_main_table(paper_table)


if __name__ == "__main__":
    main()
