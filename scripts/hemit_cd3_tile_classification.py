#!/usr/bin/env python3
"""
HEMIT CD3-positive tile classification downstream task.

Biological question: can a generated mIF tile identify whether the region is
CD3-positive / immune-infiltrated?

GT label:
  Segment nuclei from real DAPI, then label a tile positive if the GT CD3 channel
  contains at least --min-positive-cells CD3+ nuclei using the existing downstream
  Otsu/percentile nucleus-positive rule.

Prediction score:
  Predicted CD3+ nucleus fraction measured on the same GT DAPI nuclei. This is a
  continuous score, so we report ROC-AUC and average precision. We also report
  F1 at the threshold that maximizes Youden's J on that model's tiles.

Input can be a manifest CSV with columns model,srcdir or repeated --model
arguments:
  --model "Ours=results/hemit_fm_cross_attn_scratch_512/test_80/images"

Outputs:
  cd3_tile_classification_per_tile.csv
  cd3_tile_classification_leaderboard.csv
  fig_cd3_tile_auc.png

Example:
  python scripts/hemit_cd3_tile_classification.py \
    --model "Ours=results/hemit_fm_cross_attn_scratch_512/test_80/images" \
    --model "Pix2Pix=results/hemit_pix2pix_resnet9_512/test_80/images" \
    --out-dir figures/hemit/downstream_cd3_tile
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.hemit_eval_path import setup_hemit_eval_path  # noqa: E402

setup_hemit_eval_path()

from hemit_eval.downstream_biology import (  # noqa: E402
    _marker_positive_nuclei_count,
    segment_nuclei,
)
from hemit_eval.image_io import list_fake_files, load_pair, resolve_image_dir  # noqa: E402

FONT = "Arial"
OURS_LABEL = "Ours"
OURS_COLOR = "#C62828"
BASE_COLOR = "#90A4AE"


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


def parse_models(args) -> list[tuple[str, Path]]:
    models: list[tuple[str, Path]] = []
    if args.manifest:
        with Path(args.manifest).expanduser().open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if not row.get("model") or not row.get("srcdir"):
                    raise SystemExit("Manifest must contain model,srcdir columns")
                models.append((row["model"], Path(row["srcdir"]).expanduser()))
    for spec in args.model or []:
        if "=" not in spec:
            raise SystemExit(f"--model must be Label=srcdir, got {spec}")
        label, src = spec.split("=", 1)
        models.append((label.strip(), Path(src.strip()).expanduser()))
    if not models:
        raise SystemExit("Provide --manifest or at least one --model Label=srcdir")
    return models


def score_dir(
    label_name: str,
    srcdir: Path,
    *,
    marker_percentile: float,
    min_positive_cells: int,
) -> list[dict]:
    image_dir = resolve_image_dir(srcdir)
    rows = []
    for fake_path in list_fake_files(image_dir):
        real, fake, base = load_pair(fake_path)
        nuclei = segment_nuclei(real[..., 0])
        n_nuclei = int(np.max(nuclei.astype(np.uint8)))  # overwritten below if labels exist
        gt_count = _marker_positive_nuclei_count(
            real[..., 1], nuclei, marker_percentile=marker_percentile,
        )
        pred_count = _marker_positive_nuclei_count(
            fake[..., 1], nuclei, marker_percentile=marker_percentile,
        )
        try:
            from skimage.measure import label as sk_label
            n_nuclei = int(sk_label(nuclei).max())
        except Exception:
            pass
        pred_fraction = pred_count / n_nuclei if n_nuclei else 0.0
        rows.append({
            "model": label_name,
            "file_name": base,
            "n_nuclei": n_nuclei,
            "gt_cd3_count": gt_count,
            "pred_cd3_count": pred_count,
            "gt_cd3_positive_tile": int(gt_count >= min_positive_cells),
            "pred_cd3_fraction_score": pred_fraction,
            "pred_cd3_count_score": pred_count,
        })
    return rows


def _best_threshold_f1(y: np.ndarray, score: np.ndarray) -> tuple[float, float, float]:
    thresholds = np.unique(score[np.isfinite(score)])
    if thresholds.size == 0:
        return float("nan"), float("nan"), float("nan")
    best = (float("nan"), -1.0, float("nan"))  # threshold, f1, balanced_acc
    for thr in thresholds:
        pred = score >= thr
        tp = int(np.sum((pred == 1) & (y == 1)))
        fp = int(np.sum((pred == 1) & (y == 0)))
        fn = int(np.sum((pred == 0) & (y == 1)))
        tn = int(np.sum((pred == 0) & (y == 0)))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        sens = recall
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        bal = 0.5 * (sens + spec)
        if bal > best[2] or (np.isclose(bal, best[2]) and f1 > best[1]):
            best = (float(thr), float(f1), float(bal))
    return best


def _roc_auc(y: np.ndarray, score: np.ndarray) -> float:
    """ROC-AUC via the rank (Mann-Whitney U) formulation, with tie handling."""
    pos = score[y == 1]
    neg = score[y == 0]
    n_pos, n_neg = pos.size, neg.size
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(score.size, dtype=np.float64)
    s = score[order]
    i = 0
    while i < s.size:
        j = i
        while j + 1 < s.size and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0  # average rank for ties (1-indexed)
        i = j + 1
    sum_ranks_pos = float(np.sum(ranks[y == 1]))
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _average_precision(y: np.ndarray, score: np.ndarray) -> float:
    """AP = area under precision-recall (step), matching sklearn's definition."""
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


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in list(dict.fromkeys(df["model"])):
        sub = df[df["model"] == model].copy()
        y = sub["gt_cd3_positive_tile"].to_numpy(dtype=int)
        score = sub["pred_cd3_fraction_score"].to_numpy(dtype=float)
        if len(np.unique(y)) < 2:
            auc = ap = float("nan")
        else:
            auc = _roc_auc(y, score)
            ap = _average_precision(y, score)
        thr, f1, bal = _best_threshold_f1(y, score)
        rows.append({
            "model": model,
            "n_tiles": len(sub),
            "n_positive_tiles": int(y.sum()),
            "positive_tile_fraction": float(y.mean()) if y.size else float("nan"),
            "roc_auc": auc,
            "average_precision": ap,
            "best_threshold_pred_fraction": thr,
            "f1_at_best_threshold": f1,
            "balanced_accuracy_at_best_threshold": bal,
            "pred_cd3_count_mae": float(np.mean(np.abs(
                sub["pred_cd3_count_score"] - sub["gt_cd3_count"]
            ))),
        })
    return pd.DataFrame(rows)


def plot_auc(leaderboard: pd.DataFrame, out_path: Path, dpi: int) -> None:
    labels = leaderboard["model"].tolist()
    vals = leaderboard["roc_auc"].to_numpy(dtype=float)
    colors = [OURS_COLOR if m == OURS_LABEL else BASE_COLOR for m in labels]
    fig, ax = plt.subplots(figsize=(7.0, 4.6), facecolor="white")
    bars = ax.bar(np.arange(len(labels)), vals, color=colors, edgecolor="black", linewidth=0.6)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1.2)
    for bar, val in zip(bars, vals):
        if np.isfinite(val):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.015, f"{val:.2f}",
                    ha="center", va="bottom", fontsize=10)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("ROC-AUC")
    ax.set_title("CD3-positive tile classification (immune-infiltrated vs negative)",
                 fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="HEMIT CD3-positive tile classification")
    p.add_argument("--model", action="append", default=[], help="Label=srcdir")
    p.add_argument("--manifest", default=None, help="CSV with model,srcdir columns")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--marker-percentile", type=float, default=60)
    p.add_argument("--min-positive-cells", type=int, default=1)
    p.add_argument("--dpi", type=int, default=220)
    args = p.parse_args()

    apply_style()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for label_name, srcdir in parse_models(args):
        print(f"Scoring {label_name}: {srcdir}")
        all_rows.extend(score_dir(
            label_name, srcdir,
            marker_percentile=args.marker_percentile,
            min_positive_cells=args.min_positive_cells,
        ))
    df = pd.DataFrame(all_rows)
    if df.empty:
        raise SystemExit("No tiles scored")
    df.to_csv(out_dir / "cd3_tile_classification_per_tile.csv", index=False)
    leaderboard = summarize(df)
    leaderboard.to_csv(out_dir / "cd3_tile_classification_leaderboard.csv", index=False)
    plot_auc(leaderboard, out_dir / "fig_cd3_tile_auc.png", args.dpi)

    print("\n=== CD3-positive tile classification ===")
    for _, row in leaderboard.iterrows():
        print(
            f"{row['model']:<12} AUC={row['roc_auc']:.3f} "
            f"AP={row['average_precision']:.3f} "
            f"F1={row['f1_at_best_threshold']:.3f} "
            f"balAcc={row['balanced_accuracy_at_best_threshold']:.3f}"
        )


if __name__ == "__main__":
    main()
