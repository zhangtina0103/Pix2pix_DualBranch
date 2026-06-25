#!/usr/bin/env python3
"""Build HEMIT paper figures: quantitative plots + qualitative comparisons."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

# ---------------------------------------------------------------------------
# Style — Arial, publication labels
# ---------------------------------------------------------------------------

FONT = "Arial"

TITLES = {
    "benchmark_bars": "Quantitative Benchmark on the HEMIT Test Set (Epoch 80, n = 945)",
    "benchmark_box": "Per-Tile Score Distributions on the HEMIT Test Set",
    "per_marker": "Per-Marker Pearson Correlation on the HEMIT Test Set",
    "ablation": "Ablation Study: Per-Tile Metric Distributions (n = 945)",
    "downstream": "Downstream Biological Validation (Per-Cell Analysis, n = 945)",
    "qual_zoom": "Qualitative Comparison of Virtual Staining Methods on HEMIT",
    "qual_comparison": "Qualitative Comparison of Virtual Staining Methods on HEMIT",
    "qual_detail": "Representative Virtual Staining Result (Ground Truth vs. Ours)",
}

COL_LABELS = [
    "H&E", "Ground Truth", "Ours", "Pix2Pix", "CUT", "CycleGAN", "ASP", "DiffVS",
]

BORDER_HE = "#D32F2F"
BORDER_MIF = "#F9A825"

OURS_COLOR = "#2E7D32"
BASELINE_COLOR = "#78909C"
BOX_FACE = "#ECEFF1"

ZOOM_TILES = [
    "[18778,52957]_patch_0_8",
    "[19129,51780]_patch_2_8",
    "[10382,50252]_patch_0_4",
]

COMPARISON_TILES = [
    "[18778,52957]_patch_0_8",
    "[19129,51780]_patch_2_8",
    "[10382,50252]_patch_0_4",
    "[19129,51780]_patch_4_7",
]

# GT vs Ours detail panel — visible DAPI + close match to ground truth
DETAIL_TILE = "[19129,51780]_patch_2_8"

MODEL_KEYS = [
    "hemit_fm_cross_attn_scratch_512",
    "hemit_pix2pix_resnet9_512",
    "hemit_cut_joint_512",
    "hemit_cyclegan_joint_512",
    "hemit_asp_joint_512",
    "hemit_dvst",
]

ZOOM_BOX = 72
N_ZOOM = 4

BENCHMARK_MODELS = [
    ("pix2pix", "Pix2Pix"),
    ("cut", "CUT"),
    ("cyclegan", "CycleGAN"),
    ("asp", "ASP"),
    ("dvst", "DiffVS"),
    ("cross_attn", "Ours"),
]

ABLATION_CSV_MAP: list[tuple[str, list[str] | None]] = [
    ("Vanilla FM", ["vanilla.csv"]),
    ("+ cross-attn + vel", None),
    ("+ focal γ=1", ["focal_loss.csv", "focal_loss (1).csv"]),
    ("+ CD3 focal", ["cd3_focal.csv"]),
    ("+ seg cond.", ["score-seg.csv", "score-segonly80.csv"]),
]

DOWNSTREAM_MODELS = [
    ("cyclegan", "CycleGAN"),
    ("pix2pix", "Pix2Pix"),
    ("cut", "CUT"),
    ("asp", "ASP"),
    ("vanilla_fm", "Vanilla FM"),
    ("cross_attn", "Ours"),
]

CROSS_ATTN_CANDIDATES = ["score-3.csv", "score-2.csv", "cross_attn.csv", "score-cross-attn.csv"]


def apply_plot_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [FONT, "Helvetica", "DejaVu Sans"],
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 13,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


# ---------------------------------------------------------------------------
# Benchmark data loaders
# ---------------------------------------------------------------------------

def load_summary_metric(summary_path: Path, metric: str) -> tuple[float, float]:
    with summary_path.open() as f:
        for row in csv.DictReader(f):
            if row.get("scope") == "average" and row.get("channel") == "mean" and row.get("metric") == metric:
                return float(row["mean"]), float(row["std"])
    raise KeyError(f"{metric} not found in {summary_path}")


def load_benchmark_summary(benchmark_dir: Path) -> list[tuple[str, str, float, float, float, float, float, float]]:
    """(key, display, r_mean, r_std, ssim_mean, ssim_std, psnr_mean, psnr_std)."""
    rows: list[tuple[str, str, float, float, float, float, float, float]] = []
    for key, display in BENCHMARK_MODELS:
        p = benchmark_dir / key / "extended_metrics_summary.csv"
        if not p.is_file():
            print(f"  missing benchmark summary: {p}")
            continue
        rm, rs = load_summary_metric(p, "pearson")
        sm, ss = load_summary_metric(p, "ssim")
        pm, ps = load_summary_metric(p, "psnr")
        rows.append((key, display, rm, rs, sm, ss, pm, ps))
    return rows


def load_per_marker_from_benchmark(benchmark_dir: Path) -> dict[str, tuple[float, float, float]]:
    """display name -> (DAPI r, CD3 r, panCK r) means."""
    out: dict[str, tuple[float, float, float]] = {}
    for key, display in [("asp", "ASP"), ("cut", "CUT"), ("cross_attn", "Ours")]:
        p = benchmark_dir / key / "extended_metrics_summary.csv"
        if not p.is_file():
            continue
        with p.open() as f:
            rows = list(csv.DictReader(f))
        ch = {}
        for row in rows:
            if row.get("scope") == "channel" and row.get("metric") == "pearson":
                ch[row["channel"]] = float(row["mean"])
        if len(ch) >= 3:
            out[display] = (ch["dapi"], ch["cd3"], ch["panck"])
    return out


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def load_mif_array(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path))
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[-1] > 3:
        arr = arr[..., :3]
    arr = arr.astype(np.float64)
    if arr.min() < 0:
        arr = (arr + 1.0) / 2.0 * 255.0
    elif arr.max() <= 1.0:
        arr = arr * 255.0
    return np.clip(arr, 0, 255)


def tif_to_mif_rgb(path: Path, size: int = 512) -> np.ndarray:
    """panCK→R, CD3→G, DAPI→B — same linear mapping as post_process.tif_composite."""
    arr = load_mif_array(path)
    dapi, cd3, panck = arr[..., 0], arr[..., 1], arr[..., 2]
    rgb = np.zeros((*arr.shape[:2], 3), dtype=np.uint8)
    rgb[..., 0] = panck.astype(np.uint8)
    rgb[..., 1] = cd3.astype(np.uint8)
    rgb[..., 2] = dapi.astype(np.uint8)
    if rgb.shape[0] != size or rgb.shape[1] != size:
        rgb = np.asarray(Image.fromarray(rgb).resize((size, size), Image.BILINEAR))
    return rgb


def load_he(path: Path, size: int = 512) -> np.ndarray:
    arr = np.asarray(Image.open(path))
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0:
            arr = ((arr + 1.0) / 2.0 * 255.0).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[-1] > 3:
        arr = arr[..., :3]
    if arr.shape[0] != size or arr.shape[1] != size:
        arr = np.asarray(Image.fromarray(arr).resize((size, size), Image.BILINEAR))
    return arr


def _stretch_marker_channel(ch: np.ndarray) -> np.ndarray:
    """Display-only stretch for single-marker panels (detail figure); keeps background black."""
    ch = ch.astype(np.float64)
    active = ch > 0
    if not np.any(active):
        return ch
    lo = float(np.percentile(ch[active], 2.0))
    hi = float(np.percentile(ch[active], 99.5))
    if hi <= lo:
        hi = lo + 1.0
    norm = np.clip((ch - lo) / (hi - lo), 0, 1)
    return np.where(ch <= lo, 0.0, norm * 255.0)


def marker_to_rgb(ch: np.ndarray, marker: str, enhance: bool = False) -> np.ndarray:
    ch = ch.astype(np.float64)
    if enhance:
        ch = _stretch_marker_channel(ch)
    ch = np.clip(ch, 0, 255).astype(np.uint8)
    rgb = np.zeros((*ch.shape, 3), dtype=np.uint8)
    if marker == "dapi":
        rgb[..., 2] = ch
    elif marker == "cd3":
        rgb[..., 1] = ch
    else:
        rgb[..., 0] = ch
    return rgb


def find_tile_path(tiles_dir: Path, pattern: str) -> Path | None:
    p = tiles_dir / pattern
    return p if p.is_file() else None


def tile_paths(tiles_dir: Path, tile: str) -> list[Path | None]:
    paths: list[Path | None] = [
        find_tile_path(tiles_dir, f"GT__{tile}_real_A.tif"),
        find_tile_path(tiles_dir, f"GT__{tile}_real_B.tif"),
    ]
    for key in MODEL_KEYS:
        if key == "hemit_dvst":
            paths.append(find_tile_path(tiles_dir, f"hemit_dvst__{tile}_fake_B.tif"))
        else:
            paths.append(find_tile_path(tiles_dir, f"{key}__{tile}_fake_B.tif"))
    return paths


def auto_zoom_boxes(cd3: np.ndarray, n: int = N_ZOOM, box: int = ZOOM_BOX) -> list[tuple[int, int, int, int]]:
    h, w = cd3.shape
    score = cd3.astype(np.float64).copy()
    boxes: list[tuple[int, int, int, int]] = []
    for _ in range(n):
        if score.max() <= 0:
            break
        y, x = np.unravel_index(int(np.argmax(score)), score.shape)
        x0 = int(np.clip(x - box // 2, 0, w - box))
        y0 = int(np.clip(y - box // 2, 0, h - box))
        boxes.append((x0, y0, box, box))
        y1, y2 = max(0, y0 - box), min(h, y0 + 2 * box)
        x1, x2 = max(0, x0 - box), min(w, x0 + 2 * box)
        score[y1:y2, x1:x2] = 0
    return boxes


def crop_box(img: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x, y, bw, bh = box
    return img[y : y + bh, x : x + bw]


def set_panel_border(ax, color: str, linewidth: float = 1.4) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(color)
        spine.set_linewidth(linewidth)


def draw_main_panel(ax, img: np.ndarray, boxes: list[tuple[int, int, int, int]], border_color: str) -> None:
    ax.imshow(img, aspect="auto", interpolation="bilinear")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("auto")
    set_panel_border(ax, border_color)
    for i, (x, y, bw, bh) in enumerate(boxes, start=1):
        rect = Rectangle((x, y), bw, bh, linewidth=1.0, edgecolor=BORDER_MIF, facecolor="none")
        ax.add_patch(rect)
        ax.text(
            x + 2, y + 11, str(i), color=BORDER_MIF, fontsize=8, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.08", facecolor="black", alpha=0.5, linewidth=0),
        )


def build_qualitative_zoom_figure(tiles_dir: Path, out_path: Path, dpi: int = 200) -> bool:
    """HEMIT Fig. 4 style: main panel spans same 4-column width as zoom strip below."""
    n_cols = len(COL_LABELS)
    n_tiles = len(ZOOM_TILES)

    fig = plt.figure(figsize=(2.05 * n_cols, 2.2 * n_tiles + 0.35), facecolor="white")
    outer = gridspec.GridSpec(
        n_tiles, n_cols, figure=fig,
        hspace=0.04, wspace=0.022,
        top=0.94, bottom=0.07, left=0.01, right=0.99,
    )

    any_ok = False
    col_bottom_axes: list = []

    for ti, tile in enumerate(ZOOM_TILES):
        paths = tile_paths(tiles_dir, tile)
        if paths[1] is None:
            continue
        gt_arr = load_mif_array(paths[1])
        boxes = auto_zoom_boxes(gt_arr[..., 1])
        if len(boxes) < N_ZOOM:
            continue
        any_ok = True

        images: list[np.ndarray] = []
        for ci, pth in enumerate(paths):
            if pth is None:
                images.append(np.zeros((512, 512, 3), dtype=np.uint8))
            elif ci == 0:
                images.append(load_he(pth))
            else:
                images.append(tif_to_mif_rgb(pth))

        for ci, img in enumerate(images):
            border = BORDER_HE if ci == 0 else BORDER_MIF

            # Main row spans all 4 zoom columns → identical total width
            cell_gs = gridspec.GridSpecFromSubplotSpec(
                2, N_ZOOM, subplot_spec=outer[ti, ci],
                height_ratios=[4.0, 1.0], hspace=0.012, wspace=0.003,
            )
            ax_main = fig.add_subplot(cell_gs[0, :])
            draw_main_panel(ax_main, img, boxes, border)

            last_zoom_ax = None
            for zi, box in enumerate(boxes):
                ax_z = fig.add_subplot(cell_gs[1, zi])
                ax_z.imshow(crop_box(img, box), interpolation="nearest", aspect="auto")
                ax_z.set_xticks([])
                ax_z.set_yticks([])
                ax_z.set_aspect("auto")
                set_panel_border(ax_z, border, linewidth=1.0)
                last_zoom_ax = ax_z

            if ti == n_tiles - 1 and last_zoom_ax is not None:
                col_bottom_axes.append(last_zoom_ax)

    if not any_ok:
        plt.close(fig)
        print(f"SKIP zoom figure: no tiles in {tiles_dir}")
        return False

    for ax, label in zip(col_bottom_axes, COL_LABELS):
        pos = ax.get_position()
        fig.text(
            pos.x0 + pos.width / 2, pos.y0 - 0.004, label,
            ha="center", va="top", fontsize=11, fontweight="bold", fontfamily=FONT,
        )

    fig.suptitle(TITLES["qual_zoom"], fontsize=13, fontweight="bold", y=0.98, fontfamily=FONT)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.06, facecolor="white")
    plt.close(fig)
    print(f"Wrote {out_path}")
    return True


def build_qualitative_comparison_figure(tiles_dir: Path, out_path: Path, dpi: int = 200) -> bool:
    """Full-patch grid (no zoom) — same brightness/borders as zoom figure."""
    n_cols = len(COL_LABELS)
    n_tiles = len(COMPARISON_TILES)

    fig = plt.figure(figsize=(2.05 * n_cols, 2.1 * n_tiles + 0.3), facecolor="white")
    outer = gridspec.GridSpec(
        n_tiles, n_cols, figure=fig,
        hspace=0.04, wspace=0.022,
        top=0.94, bottom=0.07, left=0.01, right=0.99,
    )

    any_ok = False
    col_bottom_axes: list = []

    for ti, tile in enumerate(COMPARISON_TILES):
        paths = tile_paths(tiles_dir, tile)
        if paths[1] is None:
            continue
        any_ok = True

        for ci, pth in enumerate(paths):
            border = BORDER_HE if ci == 0 else BORDER_MIF
            ax = fig.add_subplot(outer[ti, ci])
            if pth is None:
                img = np.zeros((512, 512, 3), dtype=np.uint8)
            elif ci == 0:
                img = load_he(pth)
            else:
                img = tif_to_mif_rgb(pth)
            ax.imshow(img, aspect="auto", interpolation="bilinear")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("auto")
            set_panel_border(ax, border)
            if ti == n_tiles - 1:
                col_bottom_axes.append(ax)

    if not any_ok:
        plt.close(fig)
        print(f"SKIP comparison figure: no tiles in {tiles_dir}")
        return False

    for ax, label in zip(col_bottom_axes, COL_LABELS):
        pos = ax.get_position()
        fig.text(
            pos.x0 + pos.width / 2, pos.y0 - 0.004, label,
            ha="center", va="top", fontsize=11, fontweight="bold", fontfamily=FONT,
        )

    fig.suptitle(TITLES["qual_comparison"], fontsize=13, fontweight="bold", y=0.98, fontfamily=FONT)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.06, facecolor="white")
    plt.close(fig)
    print(f"Wrote {out_path}")
    return True


def build_single_tile_detail(
    tiles_dir: Path,
    out_path: Path,
    tile: str | None = None,
    dpi: int = 200,
    *,
    tight: bool = False,
) -> bool:
    tile = tile or DETAIL_TILE
    gt_a = find_tile_path(tiles_dir, f"GT__{tile}_real_A.tif")
    gt_b = find_tile_path(tiles_dir, f"GT__{tile}_real_B.tif")
    ours = find_tile_path(tiles_dir, f"hemit_fm_cross_attn_scratch_512__{tile}_fake_B.tif")
    if not all([gt_a, gt_b, ours]):
        print(f"SKIP detail figure: missing files for {tile}")
        return False

    he = load_he(gt_a)
    gt_arr = load_mif_array(gt_b)
    pr_arr = load_mif_array(ours)
    gt_comp = tif_to_mif_rgb(gt_b)
    pr_comp = tif_to_mif_rgb(ours)

    cols = ["H&E", "mIF Composite", "DAPI", "CD3", "panCK"]
    wspace = 0.006 if tight else 0.05
    fig_w = 12.0 if tight else 14.0
    fig = plt.figure(figsize=(fig_w, 5.4), facecolor="white")
    gs = gridspec.GridSpec(2, 5, figure=fig, hspace=0.08, wspace=wspace, top=0.86, bottom=0.08, left=0.08, right=0.99)

    rows_data = [
        ("Ground Truth", he, gt_comp, gt_arr),
        ("Ours (FM + Cross-Attention)", he, pr_comp, pr_arr),
    ]
    markers = [None, None, "dapi", "cd3", "panck"]

    for j, (row_label, he_img, comp, arr) in enumerate(rows_data):
        chans = [he_img, comp, arr[..., 0], arr[..., 1], arr[..., 2]]
        for i, m in enumerate(markers):
            ax = fig.add_subplot(gs[j, i])
            ax.axis("off")
            if m is None:
                ax.imshow(chans[i])
            else:
                ax.imshow(marker_to_rgb(chans[i], m, enhance=True))
            border = BORDER_HE if i == 0 else (BORDER_MIF if m else BORDER_MIF)
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor(border)
                spine.set_linewidth(1.2)
            if j == 0:
                ax.set_title(cols[i], fontsize=11, fontweight="bold", pad=3 if tight else 5, fontfamily=FONT)
        fig.text(
            0.02, 0.76 - j * 0.40, row_label,
            fontsize=11, fontweight="bold", va="center", rotation=90, fontfamily=FONT,
        )

    fig.suptitle(TITLES["qual_detail"], fontsize=13, fontweight="bold", fontfamily=FONT)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pad = 0.02 if tight else 0.06
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=pad, facecolor="white")
    plt.close(fig)
    print(f"Wrote {out_path}")
    return True


# ---------------------------------------------------------------------------
# Quantitative plots
# ---------------------------------------------------------------------------

def _save(fig, path: Path, dpi: int = 200) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {path}")


def load_metric_column(csv_path: Path, col: str) -> list[float]:
    vals: list[float] = []
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            v = row.get(col)
            if v is None or v == "" or str(v).lower() == "nan":
                continue
            x = float(v)
            if math.isfinite(x):
                vals.append(x)
    return vals


def _summary_stats(vals: list[float]) -> tuple[float, float, float]:
    if not vals:
        return float("nan"), float("nan"), float("nan")
    mu = float(sum(vals) / len(vals))
    sd = float(math.sqrt(sum((v - mu) ** 2 for v in vals) / max(len(vals) - 1, 1))) if len(vals) > 1 else 0.0
    med = float(sorted(vals)[len(vals) // 2])
    return mu, sd, med


def sync_cross_attn_benchmark(score_csv: Path, benchmark_dir: Path) -> None:
    """Write cross_attn/ benchmark CSVs from post_process score.csv (e.g. cross-attn+vel)."""
    with score_csv.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows in {score_csv}")

    out_dir = benchmark_dir / "cross_attn"
    out_dir.mkdir(parents=True, exist_ok=True)

    tile_cols = list(rows[0].keys())
    per_tile = out_dir / "extended_metrics_per_tile.csv"
    with per_tile.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=tile_cols)
        w.writeheader()
        w.writerows(rows)

    specs = [
        ("channel", "dapi", "ssim", "dapi_ssim", True),
        ("channel", "dapi", "pearson", "dapi_pearson", True),
        ("channel", "dapi", "psnr", "dapi_psnr", True),
        ("channel", "cd3", "ssim", "cd3_ssim", True),
        ("channel", "cd3", "pearson", "cd3_pearson", True),
        ("channel", "cd3", "psnr", "cd3_psnr", True),
        ("channel", "panck", "ssim", "panck_ssim", True),
        ("channel", "panck", "pearson", "panck_pearson", True),
        ("channel", "panck", "psnr", "panck_psnr", True),
        ("average", "mean", "ssim", "average_ssim", True),
        ("average", "mean", "pearson", "average_pearson", True),
        ("average", "mean", "psnr", "average_psnr", True),
    ]
    summary_rows = []
    n = len(rows)
    for scope, channel, metric, col, higher in specs:
        vals = []
        for row in rows:
            v = row.get(col)
            if v is None or v == "" or str(v).lower() == "nan":
                continue
            x = float(v)
            if math.isfinite(x):
                vals.append(x)
        mu, sd, med = _summary_stats(vals)
        summary_rows.append({
            "scope": scope,
            "channel": channel,
            "metric": metric,
            "n": str(len(vals) or n),
            "mean": f"{mu:.6f}",
            "std": f"{sd:.6f}",
            "median": f"{med:.6f}",
            "ci_low": "",
            "ci_high": "",
            "higher_is_better": str(higher),
        })

    summary_path = out_dir / "extended_metrics_summary.csv"
    fields = ["scope", "channel", "metric", "n", "mean", "std", "median", "ci_low", "ci_high", "higher_is_better"]
    with summary_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(summary_rows)
    print(f"Updated cross_attn benchmark from {score_csv.name} → {out_dir}")


def styled_boxplot(ax, data, labels, *, highlight: str | None = None, show_outliers: bool = False) -> None:
    import inspect
    kw = "tick_labels" if "tick_labels" in inspect.signature(ax.boxplot).parameters else "labels"
    bp = ax.boxplot(
        data, patch_artist=True, widths=0.55,
        showfliers=show_outliers,
        medianprops=dict(color="#E65100", linewidth=1.4),
        whiskerprops=dict(linewidth=0.9),
        capprops=dict(linewidth=0.9),
        boxprops=dict(linewidth=0.9),
        **{kw: labels},
    )
    for i, patch in enumerate(bp["boxes"]):
        is_ours = highlight is not None and labels[i] == highlight
        patch.set_facecolor(OURS_COLOR if is_ours else BOX_FACE)
        patch.set_alpha(0.85 if is_ours else 0.95)
        patch.set_edgecolor("#37474F")


def resolve_cross_attn_csv(metrics_dir: Path, explicit: Path | None) -> Path | None:
    if explicit is not None and explicit.is_file():
        return explicit
    for name in CROSS_ATTN_CANDIDATES:
        p = metrics_dir / name
        if p.is_file():
            return p
    return None


def resolve_ablation_csv(metrics_dir: Path, candidates: list[str]) -> Path | None:
    for name in candidates:
        p = metrics_dir / name
        if p.is_file():
            return p
    return None


def sync_ablation_metrics(metrics_dir: Path, ablation_dir: Path) -> None:
    """Copy known ablation score.csv files into ablation_dir for self-contained rebuilds."""
    ablation_dir.mkdir(parents=True, exist_ok=True)
    names: set[str] = set()
    for _, cands in ABLATION_CSV_MAP:
        if cands:
            names.update(cands)
    names.update(CROSS_ATTN_CANDIDATES)
    for name in sorted(names):
        src = metrics_dir / name
        if src.is_file():
            dst = ablation_dir / name
            if not dst.exists() or src.stat().st_mtime > dst.stat().st_mtime:
                dst.write_bytes(src.read_bytes())
                print(f"  synced ablation metric: {name}")


def build_benchmark_bars(benchmark_dir: Path, out_path: Path) -> None:
    summary = load_benchmark_summary(benchmark_dir)
    if not summary:
        print("SKIP benchmark bars: no summary CSVs")
        return

    models = [r[1] for r in summary]
    specs = [
        ("Average Pearson r", [(r[2], r[3]) for r in summary], 0.48, None),
        ("Average SSIM", [(r[4], r[5]) for r in summary], 0.68, None),
        ("Average PSNR (dB)", [(r[6], r[7]) for r in summary], 24.0, None),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5))
    fig.subplots_adjust(top=0.82, bottom=0.18, wspace=0.32)
    x = np.arange(len(models))
    colors = [OURS_COLOR if m == "Ours" else BASELINE_COLOR for m in models]

    for ax, (ylabel, data, ymin, _) in zip(axes, specs):
        means = [d[0] for d in data]
        stds = [d[1] for d in data]
        ax.bar(x, means, yerr=stds, capsize=4, color=colors, edgecolor="white", linewidth=0.6, width=0.72,
               error_kw=dict(lw=1.0, capthick=1.0))
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=35, ha="right")
        ax.set_ylabel(ylabel, fontweight="bold")
        ymax = max(means[i] + stds[i] for i in range(len(means)))
        pad = (ymax - ymin) * 0.18
        ax.set_ylim(ymin, ymax + pad)
        ax.grid(axis="y", alpha=0.22)
        for bar, m, s in zip(ax.patches[: len(means)], means, stds):
            ax.text(
                bar.get_x() + bar.get_width() / 2, m + s + pad * 0.08,
                f"{m:.3f}", ha="center", va="bottom", fontsize=9,
            )

    fig.suptitle(TITLES["benchmark_bars"], fontweight="bold", y=0.98, fontfamily=FONT)
    _save(fig, out_path)


def build_benchmark_boxplots(benchmark_dir: Path, out_path: Path, show_outliers: bool = False) -> None:
    series: list[tuple[str, Path]] = []
    for key, display in BENCHMARK_MODELS:
        p = benchmark_dir / key / "extended_metrics_per_tile.csv"
        if p.is_file():
            series.append((display, p))
    if len(series) < 2:
        print("SKIP benchmark boxplots: need per-tile CSVs in benchmark-dir")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.subplots_adjust(top=0.82, bottom=0.22, wspace=0.28)
    for ax, col, ylab, ylim in zip(
        axes,
        ["average_pearson", "average_ssim"],
        ["Average Pearson r", "Average SSIM"],
        [(0.0, 1.0), (0.55, 1.0)],
    ):
        data, labels = [], []
        for label, path in series:
            vals = load_metric_column(path, col)
            if vals:
                data.append(vals)
                labels.append(label)
        styled_boxplot(ax, data, labels, highlight="Ours", show_outliers=show_outliers)
        ax.set_ylabel(ylab, fontweight="bold")
        ax.set_ylim(ylim)
        ax.tick_params(axis="x", rotation=28)
        ax.grid(axis="y", alpha=0.22)

    fig.suptitle(TITLES["benchmark_box"], fontweight="bold", y=0.98, fontfamily=FONT)
    _save(fig, out_path)


def build_per_marker_bars(benchmark_dir: Path, out_path: Path) -> None:
    per_marker = load_per_marker_from_benchmark(benchmark_dir)
    if not per_marker:
        print("SKIP per-marker bars")
        return

    models = list(per_marker.keys())
    markers = ["DAPI", "CD3", "panCK"]
    x = np.arange(len(markers))
    width = 0.25
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    fig.subplots_adjust(top=0.82, bottom=0.15)

    for i, model in enumerate(models):
        vals = per_marker[model]
        offset = (i - 1) * width
        color = OURS_COLOR if model == "Ours" else BASELINE_COLOR
        bars = ax.bar(x + offset, vals, width, label=model, color=color, edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(markers)
    ax.set_ylabel("Pearson r", fontweight="bold")
    ax.set_ylim(0.48, 1.02)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.22)
    fig.suptitle(TITLES["per_marker"], fontweight="bold", y=0.98, fontfamily=FONT)
    _save(fig, out_path)


def build_ablation_boxplots(metrics_dir: Path, out_path: Path, cross_attn_csv: Path | None, show_outliers: bool = False) -> None:
    paths: list[tuple[str, Path]] = []
    for label, cands in ABLATION_CSV_MAP:
        if label.startswith("+ cross-attn"):
            if cross_attn_csv is not None and cross_attn_csv.is_file():
                paths.append((label, cross_attn_csv))
            else:
                print(f"  ablation missing: {label}")
            continue
        if not cands:
            continue
        p = resolve_ablation_csv(metrics_dir, cands)
        if p is not None:
            paths.append((label, p))
            print(f"  ablation {label}: {p.name}")
        else:
            print(f"  ablation missing: {label} (tried {', '.join(cands)})")
    if len(paths) < 2:
        print("SKIP ablation boxplots")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.subplots_adjust(top=0.82, bottom=0.22, wspace=0.28)
    for ax, col, ylab, ylim in zip(
        axes,
        ["average_pearson", "average_ssim"],
        ["Average Pearson r", "Average SSIM"],
        [(0.0, 1.0), (0.55, 1.0)],
    ):
        data, labels = [], []
        for label, path in paths:
            vals = load_metric_column(path, col)
            if vals:
                data.append(vals)
                labels.append(label.replace("+ ", ""))
        styled_boxplot(
            ax, data, labels,
            highlight=next((l.replace("+ ", "") for l, _ in paths if "cross" in l.lower()), None),
            show_outliers=show_outliers,
        )
        ax.set_ylabel(ylab, fontweight="bold")
        ax.set_ylim(ylim)
        ax.tick_params(axis="x", rotation=22)
        ax.grid(axis="y", alpha=0.22)

    fig.suptitle(TITLES["ablation"], fontweight="bold", y=0.98, fontfamily=FONT)
    _save(fig, out_path)


def build_downstream_boxplots(metrics_dir: Path, out_path: Path, show_outliers: bool = False) -> None:
    base = metrics_dir / "percell_downstream"
    if not base.is_dir():
        print("SKIP downstream boxplots")
        return

    loaded: list[tuple[str, Path]] = []
    for key, label in DOWNSTREAM_MODELS:
        p = base / key / "percell_per_tile.csv"
        if p.is_file():
            loaded.append((label, p))
    if len(loaded) < 2:
        print("SKIP downstream boxplots")
        return

    metrics = [
        ("cd3_percell_pearson", "CD3 Per-Cell Pearson r"),
        ("panck_percell_pearson", "panCK Per-Cell Pearson r"),
        ("coexp_abs_err", "Co-Expression Error (Lower Is Better)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.5))
    fig.subplots_adjust(top=0.82, bottom=0.22, wspace=0.32)
    for ax, (col, ylab) in zip(axes, metrics):
        data, labels = [], []
        for label, path in loaded:
            vals = load_metric_column(path, col)
            if len(vals) >= 10:
                data.append(vals)
                labels.append(label)
        if not data:
            continue
        styled_boxplot(ax, data, labels, highlight="Ours", show_outliers=show_outliers)
        ax.set_ylabel(ylab, fontweight="bold", fontsize=10)
        ax.tick_params(axis="x", rotation=28)
        ax.grid(axis="y", alpha=0.22)

    fig.suptitle(TITLES["downstream"], fontweight="bold", y=0.98, fontfamily=FONT)
    _save(fig, out_path)


def main() -> None:
    apply_plot_style()

    p = argparse.ArgumentParser(description="Build HEMIT paper figures")
    p.add_argument("--tiles-dir", type=Path, default=Path.home() / "Downloads/new_tiles-2")
    p.add_argument("--metrics-dir", type=Path, default=Path.home() / "Downloads")
    p.add_argument("--benchmark-dir", type=Path, default=None,
                   help="main/ folder with per-model extended_metrics_* (from benchmark zip)")
    p.add_argument("--cross-attn-csv", type=Path, default=None)
    p.add_argument("--out", type=Path, default=Path("figures/hemit"))
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--show-outliers", action="store_true")
    p.add_argument("--detail-tile", type=str, default=None,
                   help=f"Tile stem for fig_qualitative_detail (default: {DETAIL_TILE})")
    p.add_argument("--skip-quant", action="store_true")
    p.add_argument("--skip-qual", action="store_true")
    args = p.parse_args()

    repo = Path(__file__).resolve().parents[1]
    benchmark_dir = args.benchmark_dir or repo / "data" / "hemit_benchmark" / "main"
    out = args.out
    ablation_dir = out / "metrics" / "ablation"
    print(f"Syncing ablation CSVs → {ablation_dir}")
    sync_ablation_metrics(args.metrics_dir, ablation_dir)
    metrics_dir = ablation_dir if any(ablation_dir.glob("*.csv")) else args.metrics_dir

    cross = resolve_cross_attn_csv(metrics_dir, args.cross_attn_csv)
    if cross is None and args.cross_attn_csv is None:
        cross = resolve_cross_attn_csv(args.metrics_dir, None)
    if cross:
        print(f"Using cross-attn CSV: {cross}")
        try:
            sync_cross_attn_benchmark(cross, benchmark_dir)
        except Exception as exc:
            print(f"WARN: could not sync cross_attn benchmark: {exc}")
    print(f"Using benchmark dir: {benchmark_dir}")

    if not args.skip_quant:
        build_benchmark_bars(benchmark_dir, out / "fig_benchmark_bars.png")
        build_benchmark_boxplots(benchmark_dir, out / "fig_benchmark_boxplots.png", args.show_outliers)
        build_per_marker_bars(benchmark_dir, out / "fig_per_marker_bars.png")
        build_ablation_boxplots(metrics_dir, out / "fig_ablation_boxplots.png", cross, args.show_outliers)
        build_downstream_boxplots(args.metrics_dir, out / "fig_downstream_boxplots.png", args.show_outliers)

    if not args.skip_qual:
        build_qualitative_zoom_figure(args.tiles_dir, out / "fig_qualitative_zoom.png", dpi=args.dpi)
        build_qualitative_comparison_figure(args.tiles_dir, out / "fig_qualitative_comparison.png", dpi=args.dpi)
        build_single_tile_detail(
            args.tiles_dir, out / "fig_qualitative_detail.png",
            tile=args.detail_tile, dpi=args.dpi,
        )
        build_single_tile_detail(
            args.tiles_dir, out / "fig_qualitative_detail_matched.png",
            tile=args.detail_tile, dpi=args.dpi, tight=True,
        )


if __name__ == "__main__":
    main()
