#!/usr/bin/env python3
"""
Figures for marker sparsity / density definitions.

Reads the per-tile CSVs written by:
  - scripts/compute_marker_sparsity.py        (HEMIT + CaMSC pixel-area PPF)
  - scripts/compute_cell_expression_fraction.py (CaMSC cell-level expression)

and renders a 3-panel boxplot figure that tells the full methodological story:
  (a) HEMIT pixel-area PPF: CD3 sparse vs DAPI/panCK dense.
  (b) CaMSC pixel-area PPF: ambiguous (Hoechst nuclei cover little area).
  (c) CaMSC cell-level expression fraction: WT1 sparse vs Hoechst (100%).

Example:
  python scripts/plot_marker_sparsity_figures.py \
    --hemit-ppf  results/marker_sparsity_hemit/marker_sparsity_per_tile.csv \
    --camsc-ppf  results/marker_sparsity_camsc/marker_sparsity_per_tile.csv \
    --camsc-expr results/cell_expression_camsc/cell_expression_per_tile.csv \
    --out figures/camsc/fig_marker_sparsity.png
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

FONT = "Arial"
SPARSE_COLOR = "#C62828"   # red — sparse markers
DENSE_COLOR = "#2E7D32"    # green — dense markers
BOX_FACE = "#ECEFF1"
REF_COLOR = "#1565C0"      # blue — 100% reference


def apply_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [FONT, "Helvetica", "DejaVu Sans"],
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def load_column(csv_path: Path, col: str, scale: float = 100.0) -> list[float]:
    vals: list[float] = []
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            v = row.get(col)
            if v is None or v == "" or str(v).lower() == "nan":
                continue
            x = float(v)
            if math.isfinite(x):
                vals.append(x * scale)
    return vals


def _box(ax, data, labels, colors, *, title, ylabel, ymax=None):
    import inspect
    kw = "tick_labels" if "tick_labels" in inspect.signature(ax.boxplot).parameters else "labels"
    bp = ax.boxplot(
        data, patch_artist=True, widths=0.55, showfliers=False,
        medianprops=dict(color="#E65100", linewidth=1.6),
        whiskerprops=dict(linewidth=0.9),
        capprops=dict(linewidth=0.9),
        boxprops=dict(linewidth=0.9),
        **{kw: labels},
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)
        patch.set_edgecolor("#37474F")
    # jittered points
    rng = np.random.default_rng(42)
    for i, d in enumerate(data, start=1):
        if not d:
            continue
        x = rng.normal(i, 0.05, size=len(d))
        ax.scatter(x, d, s=6, color="#455A64", alpha=0.25, zorder=3, linewidths=0)
    # mean ± std annotation
    for i, d in enumerate(data, start=1):
        if not d:
            continue
        mu = float(np.mean(d))
        sd = float(np.std(d, ddof=1)) if len(d) > 1 else 0.0
        top = max(d)
        ax.text(i, top + (ymax or max(map(max, data))) * 0.02,
                f"{mu:.1f}±{sd:.1f}", ha="center", va="bottom",
                fontsize=8.5, color="#263238")
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel(ylabel)
    if ymax is not None:
        ax.set_ylim(0, ymax)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)


def main() -> None:
    p = argparse.ArgumentParser(description="Marker sparsity / density figures")
    p.add_argument("--hemit-ppf", default="results/marker_sparsity_hemit/marker_sparsity_per_tile.csv")
    p.add_argument("--camsc-ppf", default="results/marker_sparsity_camsc/marker_sparsity_per_tile.csv")
    p.add_argument("--camsc-expr", default="results/cell_expression_camsc/cell_expression_per_tile.csv")
    p.add_argument("--scheme", default="ppf_global_otsu",
                   choices=["ppf_global_otsu", "ppf_tile_otsu"],
                   help="Which PPF column to plot")
    p.add_argument("--out", default="figures/camsc/fig_marker_sparsity.png")
    p.add_argument("--dpi", type=int, default=200)
    args = p.parse_args()

    apply_style()
    hemit_ppf = Path(args.hemit_ppf).expanduser()
    camsc_ppf = Path(args.camsc_ppf).expanduser()
    camsc_expr = Path(args.camsc_expr).expanduser()

    panels = []  # (kind, ...)
    if hemit_ppf.is_file():
        data = [
            load_column(hemit_ppf, f"cd3_{args.scheme}"),
            load_column(hemit_ppf, f"dapi_{args.scheme}"),
            load_column(hemit_ppf, f"panck_{args.scheme}"),
        ]
        n = len(data[0])
        panels.append((
            "box", data, ["CD3", "DAPI", "panCK"],
            [SPARSE_COLOR, DENSE_COLOR, DENSE_COLOR],
            f"HEMIT tissue — pixel-area PPF (n={n})", "Positive pixel fraction (%)", None,
        ))
    else:
        print(f"[skip] {hemit_ppf} not found")

    if camsc_ppf.is_file():
        data = [
            load_column(camsc_ppf, f"wt1_{args.scheme}"),
            load_column(camsc_ppf, f"hoechst_{args.scheme}"),
        ]
        n = len(data[0])
        panels.append((
            "box", data, ["WT1", "Hoechst"],
            ["#EF6C00", "#EF6C00"],
            f"CaMSC — pixel-area PPF (ambiguous, n={n})", "Positive pixel fraction (%)", None,
        ))
    else:
        print(f"[skip] {camsc_ppf} not found")

    expr_panel = None
    if camsc_expr.is_file():
        wt1 = load_column(camsc_expr, "wt1_expr_fraction")
        expr_panel = ("expr", wt1)
    else:
        print(f"[skip] {camsc_expr} not found")

    n_panels = len(panels) + (1 if expr_panel else 0)
    if n_panels == 0:
        raise SystemExit("No input CSVs found — run compute_* scripts first")

    fig, axes = plt.subplots(1, n_panels, figsize=(4.6 * n_panels, 4.6))
    if n_panels == 1:
        axes = [axes]
    ai = 0
    for kind, data, labels, colors, title, ylabel, ymax in panels:
        _box(axes[ai], data, labels, colors, title=title, ylabel=ylabel, ymax=ymax)
        ai += 1

    if expr_panel:
        ax = axes[ai]
        wt1 = expr_panel[1]
        _box(ax, [wt1], ["WT1"], [SPARSE_COLOR],
             title=f"CaMSC — cell-level expression (n={len(wt1)})",
             ylabel="Nuclei expressing marker (%)", ymax=105)
        ax.axhline(100, color=REF_COLOR, linewidth=1.6, linestyle="--", zorder=1)
        ax.text(1.0, 100, " Hoechst = 100% (all nuclei)", color=REF_COLOR,
                va="bottom", ha="center", fontsize=9, fontweight="bold")

    fig.suptitle("Quantitative definition of sparse vs dense markers",
                 fontsize=13, fontweight="bold", fontfamily=FONT)
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
