#!/usr/bin/env python3
"""Supplement / appendix figures: ablation side-by-side, error maps, workflow, per-marker scatter."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

from skimage.measure import label, regionprops

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_hemit_paper_figures import (
    BORDER_HE,
    BORDER_MIF,
    DETAIL_TILE,
    FONT,
    apply_plot_style,
    crop_box,
    find_tile_path,
    load_he,
    load_mif_array,
    marker_to_rgb,
    tif_to_mif_rgb,
)

from hemit_eval.downstream_biology import segment_nuclei

VANILLA_KEY = "hemit_vanilla_fm_joint_perc_512"
CROSS_ATTN_KEY = "hemit_fm_cross_attn_scratch_512"
# Patch with largest visible Vanilla vs Cross-Attn gap (CD3 / DAPI) among local tiles
ABLATION_TILE = "[10382,50252]_patch_0_4"
ABLATION_ZOOM_BOX = 80
ABLATION_N_ZOOM = 3
ZOOM_YELLOW = "#FBC02D"

# Tile filename prefix → display label (must match test.py output names)
SCATTER_MODELS: list[tuple[str, str]] = [
    ("hemit_pix2pix_resnet9_512", "Pix2Pix"),
    ("hemit_cut_joint_512", "CUT"),
    ("hemit_cyclegan_joint_512", "CycleGAN"),
    ("hemit_asp_joint_512", "ASP"),
    ("hemit_dvst", "DiffVS"),
    ("hemit_vanilla_fm_joint_perc_512", "Vanilla FM"),
    ("hemit_fm_cross_attn_scratch_512", "Ours"),
]
SCATTER_COLORS = {
    "Ours": "#2E7D32",
    "Vanilla FM": "#78909C",
}
SCATTER_DEFAULT_COLOR = "#1565C0"
MARKER_SPECS = (
    ("dapi", "DAPI", 0),
    ("cd3", "CD3", 1),
    ("panck", "panCK", 2),
)


def _border(ax, color: str, linewidth: float = 1.2) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(color)
        spine.set_linewidth(linewidth)


def _mean_marker_pearson(csv_path: Path) -> tuple[float, float, float]:
    dapi, cd3, panck = [], [], []
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            dapi.append(float(row["dapi_pearson"]))
            cd3.append(float(row["cd3_pearson"]))
            panck.append(float(row["panck_pearson"]))
    return float(np.mean(dapi)), float(np.mean(cd3)), float(np.mean(panck))


def build_ablation_sidebyside(
    tiles_dir: Path,
    out_path: Path,
    tile: str,
    dpi: int = 200,
) -> bool:
    """Appendix: H&E | Vanilla FM | + Cross-Attn | GT — composite + per-marker rows."""
    he_p = find_tile_path(tiles_dir, f"GT__{tile}_real_A.tif")
    gt_p = find_tile_path(tiles_dir, f"GT__{tile}_real_B.tif")
    van_p = find_tile_path(tiles_dir, f"{VANILLA_KEY}__{tile}_fake_B.tif")
    ours_p = find_tile_path(tiles_dir, f"{CROSS_ATTN_KEY}__{tile}_fake_B.tif")
    if not all([he_p, gt_p, van_p, ours_p]):
        print(f"SKIP ablation side-by-side: missing files for {tile}")
        return False

    he = load_he(he_p)
    gt_arr = load_mif_array(gt_p)
    van_arr = load_mif_array(van_p)
    ours_arr = load_mif_array(ours_p)

    cols = ["H&E", "Vanilla FM", "+ Cross-Attn", "Ground Truth"]
    row_labels = ["mIF Composite", "DAPI", "CD3", "panCK"]
    markers = [None, "dapi", "cd3", "panck"]
    arrs = [gt_arr, van_arr, ours_arr, gt_arr]  # col0 uses he for composite row only
    comp_paths = [gt_p, van_p, ours_p, gt_p]

    fig = plt.figure(figsize=(11.0, 10.0), facecolor="white")
    outer = gridspec.GridSpec(
        1, 2, figure=fig, width_ratios=[0.032, 1],
        left=0.008, right=0.978, top=0.90, bottom=0.06, wspace=0.006,
    )
    label_gs = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=outer[0], hspace=0.03)
    img_gs = gridspec.GridSpecFromSubplotSpec(4, 4, subplot_spec=outer[1], hspace=0.03, wspace=0.035)

    for i, rl in enumerate(row_labels):
        ax_lbl = fig.add_subplot(label_gs[i, 0])
        ax_lbl.axis("off")
        ax_lbl.text(
            1.0, 0.5, rl, ha="right", va="center", rotation=90,
            fontsize=9.5, fontweight="bold", fontfamily=FONT, transform=ax_lbl.transAxes,
        )

    for j in range(4):
        for i, m in enumerate(markers):
            ax = fig.add_subplot(img_gs[i, j])
            ax.axis("off")
            ax.margins(0)
            if j == 0:
                if i == 0:
                    ax.imshow(he, aspect="auto", interpolation="nearest")
                elif m is None:
                    ax.imshow(tif_to_mif_rgb(gt_p), aspect="auto", interpolation="nearest")
                else:
                    ch_i = {"dapi": 0, "cd3": 1, "panck": 2}[m]
                    ax.imshow(marker_to_rgb(gt_arr[..., ch_i], m, enhance=True), aspect="auto", interpolation="nearest")
            else:
                arr = arrs[j]
                if m is None:
                    ax.imshow(tif_to_mif_rgb(comp_paths[j]), aspect="auto", interpolation="nearest")
                else:
                    ch_i = {"dapi": 0, "cd3": 1, "panck": 2}[m]
                    ax.imshow(marker_to_rgb(arr[..., ch_i], m, enhance=True), aspect="auto", interpolation="nearest")
            _border(ax, BORDER_HE if (j == 0 and i == 0) else BORDER_MIF)
            if i == 0:
                ax.set_title(cols[j], fontsize=10.5, fontweight="bold", pad=2, fontfamily=FONT)

    fig.suptitle(
        "Ablation: Vanilla Flow Matching vs. H&E Cross-Attention (Representative Patch)",
        fontsize=13, fontweight="bold", fontfamily=FONT,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.06, facecolor="white")
    plt.close(fig)
    print(f"Wrote {out_path}")
    return True


def _ablation_advantage_score(
    gt: np.ndarray, van: np.ndarray, ours: np.ndarray, marker_ch: int,
) -> np.ndarray:
    """Per-pixel score: positive where +Cross-Attn is closer to GT than Vanilla FM."""
    g = gt[..., marker_ch].astype(np.float64)
    v = van[..., marker_ch].astype(np.float64)
    o = ours[..., marker_ch].astype(np.float64)
    advantage = np.maximum(0.0, np.abs(g - v) - np.abs(g - o))
    if (g > 0).any():
        thr = float(np.percentile(g[g > 0], 35))
        advantage = advantage * (g >= thr)
    return advantage


def _find_ablation_zoom_boxes(
    gt: np.ndarray,
    van: np.ndarray,
    ours: np.ndarray,
    marker_ch: int,
    *,
    n: int = ABLATION_N_ZOOM,
    box: int = ABLATION_ZOOM_BOX,
) -> list[tuple[int, int, int, int]]:
    score = _ablation_advantage_score(gt, van, ours, marker_ch).copy()
    h, w = score.shape
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
        score[y1:y2, x1:x2] = 0.0
    return boxes


def _pick_best_ablation_tile(tiles_dir: Path) -> str | None:
    """Tile where Cross-Attn advantage over Vanilla is largest (CD3 + panCK)."""
    best_tile: str | None = None
    best_total = -1.0
    for van_p in sorted(tiles_dir.glob(f"{VANILLA_KEY}__*_fake_B.tif")):
        tile = van_p.name[len(VANILLA_KEY) + 2 : -len("_fake_B.tif")]
        gt_p = find_tile_path(tiles_dir, f"GT__{tile}_real_B.tif")
        ours_p = find_tile_path(tiles_dir, f"{CROSS_ATTN_KEY}__{tile}_fake_B.tif")
        if not gt_p or not ours_p:
            continue
        gt = load_mif_array(gt_p)
        van = load_mif_array(van_p)
        ours = load_mif_array(ours_p)
        total = (
            _ablation_advantage_score(gt, van, ours, 1).sum()
            + _ablation_advantage_score(gt, van, ours, 2).sum()
        )
        if total > best_total:
            best_total = total
            best_tile = tile
    return best_tile


def _draw_zoom_boxes(ax, boxes: list[tuple[int, int, int, int]]) -> None:
    for i, (x, y, bw, bh) in enumerate(boxes, start=1):
        ax.add_patch(Rectangle(
            (x, y), bw, bh, linewidth=1.8, edgecolor=ZOOM_YELLOW, facecolor="none",
        ))
        ax.text(
            x + 3, y + 13, str(i), color=ZOOM_YELLOW, fontsize=9, fontweight="bold",
            fontfamily=FONT,
            bbox=dict(boxstyle="round,pad=0.12", facecolor="black", alpha=0.55, linewidth=0),
        )


def build_ablation_zoom_deep(
    tiles_dir: Path,
    out_path: Path,
    tile: str | None = None,
    *,
    markers: tuple[str, ...] = ("cd3", "panck"),
    dpi: int = 200,
) -> bool:
    """Separate deep-dive: yellow boxes where +Cross-Attn beats Vanilla FM (CD3 / panCK zooms)."""
    tile = tile or _pick_best_ablation_tile(tiles_dir) or ABLATION_TILE
    gt_p = find_tile_path(tiles_dir, f"GT__{tile}_real_B.tif")
    van_p = find_tile_path(tiles_dir, f"{VANILLA_KEY}__{tile}_fake_B.tif")
    ours_p = find_tile_path(tiles_dir, f"{CROSS_ATTN_KEY}__{tile}_fake_B.tif")
    if not all([gt_p, van_p, ours_p]):
        print(f"SKIP ablation zoom deep: missing files for {tile}")
        return False

    gt_arr = load_mif_array(gt_p)
    van_arr = load_mif_array(van_p)
    ours_arr = load_mif_array(ours_p)
    col_labels = ["Vanilla FM", "+ Cross-Attn", "Ground Truth"]
    col_arrs = [van_arr, ours_arr, gt_arr]

    marker_blocks: list[tuple[str, int, list[tuple[int, int, int, int]], list[np.ndarray]]] = []
    for mname in markers:
        ch = {"dapi": 0, "cd3": 1, "panck": 2}[mname]
        boxes = _find_ablation_zoom_boxes(gt_arr, van_arr, ours_arr, ch)
        if not boxes:
            continue
        imgs = [marker_to_rgb(a[..., ch], mname, enhance=True) for a in col_arrs]
        marker_blocks.append((mname.upper(), ch, boxes, imgs))

    if not marker_blocks:
        print("SKIP ablation zoom deep: no advantage regions found")
        return False

    n_blocks = len(marker_blocks)
    n_zoom = len(marker_blocks[0][2])
    height_ratios = ([3.2] + [1.15] * n_zoom) * n_blocks
    n_rows = len(height_ratios)
    fig_h = 2.2 * n_blocks + 1.2
    fig = plt.figure(figsize=(9.2, fig_h), facecolor="white")
    outer = gridspec.GridSpec(
        n_rows, 3, figure=fig,
        height_ratios=height_ratios,
        hspace=0.06, wspace=0.04,
        top=0.91, bottom=0.05, left=0.08, right=0.98,
    )

    for bi, (mtitle, _ch, boxes, imgs) in enumerate(marker_blocks):
        base = bi * (1 + n_zoom)
        for j, (label, img) in enumerate(zip(col_labels, imgs)):
            ax = fig.add_subplot(outer[base, j])
            ax.imshow(img, aspect="auto", interpolation="nearest")
            ax.axis("off")
            ax.margins(0)
            _draw_zoom_boxes(ax, boxes)
            _border(ax, BORDER_MIF)
            if bi == 0:
                ax.set_title(label, fontsize=11, fontweight="bold", pad=3, fontfamily=FONT)
            if j == 0:
                ax.text(
                    -0.14, 0.5, mtitle, transform=ax.transAxes,
                    fontsize=10, fontweight="bold", va="center", ha="right",
                    rotation=90, fontfamily=FONT,
                )

        for zi, box in enumerate(boxes):
            zrow = base + 1 + zi
            for j, img in enumerate(imgs):
                ax = fig.add_subplot(outer[zrow, j])
                ax.imshow(crop_box(img, box), aspect="auto", interpolation="nearest")
                ax.axis("off")
                ax.margins(0)
                _border(ax, BORDER_MIF, linewidth=1.0)
                if j == 0:
                    ax.text(
                        -0.14, 0.5, f"Zoom {zi + 1}", transform=ax.transAxes,
                        fontsize=9, fontweight="bold", va="center", ha="right",
                        rotation=90, fontfamily=FONT,
                    )

    fig.suptitle(
        "Ablation Deep-Dive: Regions Where Cross-Attention Outperforms Vanilla FM",
        fontsize=12, fontweight="bold", fontfamily=FONT,
    )
    fig.text(
        0.5, 0.965,
        f"Yellow boxes = largest GT error reduction (Vanilla → Cross-Attn)  |  tile {tile}",
        ha="center", fontsize=9, fontfamily=FONT, color="#455A64",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.05, facecolor="white")
    plt.close(fig)
    print(f"Wrote {out_path} (tile={tile}, {n_zoom} zooms × {n_blocks} markers)")
    return True


def build_error_maps(tiles_dir: Path, out_path: Path, tile: str, dpi: int = 200) -> bool:
    gt_p = find_tile_path(tiles_dir, f"GT__{tile}_real_B.tif")
    ours_p = find_tile_path(tiles_dir, f"{CROSS_ATTN_KEY}__{tile}_fake_B.tif")
    if not gt_p or not ours_p:
        print("SKIP error maps")
        return False

    gt = load_mif_array(gt_p)
    pr = load_mif_array(ours_p)
    names = ["DAPI", "CD3", "panCK"]
    fig, axes = plt.subplots(3, 3, figsize=(9, 9), facecolor="white")
    fig.subplots_adjust(top=0.88, hspace=0.08, wspace=0.05)
    for i in range(3):
        g, p = gt[..., i], pr[..., i]
        err = np.abs(g - p)
        vmax = max(float(np.percentile(err[err > 0], 99)) if (err > 0).any() else 1.0, 1.0)
        for j, (img, title) in enumerate([
            (marker_to_rgb(g, names[i].lower(), enhance=True), "Ground Truth"),
            (marker_to_rgb(p, names[i].lower(), enhance=True), "Ours"),
            (err, f"|Error| (max={vmax:.0f})"),
        ]):
            ax = axes[i, j]
            ax.axis("off")
            if j < 2:
                ax.imshow(img)
            else:
                ax.imshow(err, cmap="hot", vmin=0, vmax=vmax)
            if i == 0:
                ax.set_title(title, fontsize=10, fontweight="bold", fontfamily=FONT)
            if j == 0:
                ax.set_ylabel(names[i], fontsize=10, fontweight="bold", fontfamily=FONT)
    fig.suptitle("Per-Marker Absolute Error (Ground Truth vs. Ours)", fontsize=12, fontweight="bold", fontfamily=FONT)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out_path}")
    return True


def _resolve_tiles_dir(tiles_dir: Path) -> Path:
    tiles_dir = tiles_dir.expanduser().resolve()
    if list(tiles_dir.glob("*_fake_B.tif")):
        return tiles_dir
    images = tiles_dir / "images"
    if images.is_dir() and list(images.glob("*_fake_B.tif")):
        return images
    return tiles_dir


def _tile_from_fake(fake_name: str, model_key: str) -> str:
    prefix = f"{model_key}__"
    suffix = "_fake_B.tif"
    if not fake_name.startswith(prefix) or not fake_name.endswith(suffix):
        raise ValueError(f"Unexpected fake tile name: {fake_name}")
    return fake_name[len(prefix) : -len(suffix)]


def _gt_path_for_fake(tiles_dir: Path, fake_p: Path, model_key: str) -> Path | None:
    tile = _tile_from_fake(fake_p.name, model_key)
    for cand in (
        tiles_dir / f"GT__{tile}_real_B.tif",
        tiles_dir / f"{model_key}__{tile}_real_B.tif",
        fake_p.parent / f"{fake_p.name[:-len('_fake_B.tif')]}_real_B.tif",
    ):
        if cand.is_file():
            return cand
    return None


def _iter_gt_pred_pairs(tiles_dir: Path, model_key: str):
    tiles_dir = _resolve_tiles_dir(tiles_dir)
    fakes = sorted(tiles_dir.glob(f"{model_key}__*_fake_B.tif"))
    for fake_p in fakes:
        gt_p = _gt_path_for_fake(tiles_dir, fake_p, model_key)
        if gt_p is None:
            continue
        yield gt_p, fake_p


def _collect_pixel_intensities(
    tiles_dir: Path,
    model_key: str,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Pool per-pixel intensities (GT vs predicted) — matches post_process Pearson."""
    pools: dict[str, list[np.ndarray]] = {m: [] for m, _, _ in MARKER_SPECS}
    preds: dict[str, list[np.ndarray]] = {m: [] for m, _, _ in MARKER_SPECS}
    n_tiles = 0
    for gt_p, fake_p in _iter_gt_pred_pairs(tiles_dir, model_key):
        gt = load_mif_array(gt_p)
        pr = load_mif_array(fake_p)
        n_tiles += 1
        for marker, _, ch in MARKER_SPECS:
            pools[marker].append(gt[..., ch].ravel())
            preds[marker].append(pr[..., ch].ravel())

    if n_tiles == 0:
        print(f"  no tile pairs for model {model_key}")
        return {m: (np.array([]), np.array([])) for m, _, _ in MARKER_SPECS}

    print(f"  pooled per-pixel intensities from {n_tiles} tiles")
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for marker, _, _ in MARKER_SPECS:
        if pools[marker]:
            out[marker] = (np.concatenate(pools[marker]), np.concatenate(preds[marker]))
        else:
            out[marker] = (np.array([]), np.array([]))
    return out


def _collect_per_cell_intensities(
    tiles_dir: Path,
    model_key: str,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Pool per-nucleus mean intensities (GT vs predicted) across all tiles."""
    pools: dict[str, list[tuple[float, float]]] = {m: [] for m, _, _ in MARKER_SPECS}
    n_tiles = 0
    for gt_p, fake_p in _iter_gt_pred_pairs(tiles_dir, model_key):
        gt = load_mif_array(gt_p)
        pr = load_mif_array(fake_p)
        nuclei = segment_nuclei(gt[..., 0])
        labeled = label(nuclei)
        if labeled.max() == 0:
            continue
        n_tiles += 1
        for prop in regionprops(labeled):
            mask = labeled == prop.label
            for marker, _, ch in MARKER_SPECS:
                pools[marker].append((float(gt[mask, ch].mean()), float(pr[mask, ch].mean())))

    print(f"  pooled per-cell intensities from {n_tiles} tiles")
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for marker, _, _ in MARKER_SPECS:
        pairs = pools[marker]
        if pairs:
            out[marker] = (
                np.array([a for a, _ in pairs], dtype=np.float64),
                np.array([b for _, b in pairs], dtype=np.float64),
            )
        else:
            out[marker] = (np.array([]), np.array([]))
    return out


def _subsample_xy(
    x: np.ndarray, y: np.ndarray, max_points: int, seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if x.size <= max_points:
        return x, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(x.size, size=max_points, replace=False)
    return x[idx], y[idx]


def _corr_annotation(r: float, p: float) -> str:
    if not np.isfinite(r):
        return "r = n/a"
    lines = [f"r = {r:.3f}"]
    if np.isfinite(p) and p < 0.05:
        lines.append("p < 0.001" if p < 0.001 else f"p = {p:.3g}")
    return "\n".join(lines)


def _collect_intensities(
    tiles_dir: Path, model_key: str, mode: str,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    if mode == "percell":
        return _collect_per_cell_intensities(tiles_dir, model_key)
    return _collect_pixel_intensities(tiles_dir, model_key)


def _plot_scatter_panel(
    ax,
    x_full: np.ndarray,
    y_full: np.ndarray,
    *,
    color: str,
    max_plot_points: int,
    seed: int,
    title: str | None = None,
    show_xlabel: bool = True,
    show_ylabel: bool = True,
) -> tuple[float, float, int]:
    from scipy.stats import pearsonr

    if title:
        ax.set_title(title, fontweight="bold", fontfamily=FONT, fontsize=10)
    if x_full.size < 3:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return float("nan"), float("nan"), 0

    x, y = _subsample_xy(x_full, y_full, max_plot_points, seed)
    ax.scatter(x, y, s=10, alpha=0.32, color=color, edgecolors="none", rasterized=True)
    lo = float(min(x_full.min(), y_full.min()))
    hi = float(max(x_full.max(), y_full.max()))
    pad = max((hi - lo) * 0.04, 1.0)
    lo -= pad
    hi += pad
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.0, alpha=0.55)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    r, p = pearsonr(x_full, y_full)
    ax.text(
        0.05, 0.95, _corr_annotation(r, p),
        transform=ax.transAxes, va="top", ha="left", fontsize=8, fontfamily=FONT,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#B0BEC5", alpha=0.9),
    )
    ax.set_aspect("equal", adjustable="box")
    if show_xlabel:
        ax.set_xlabel("GT intensity", fontfamily=FONT, fontsize=9)
    else:
        ax.set_xlabel("")
    if show_ylabel:
        ax.set_ylabel("Pred intensity", fontfamily=FONT, fontsize=9)
    else:
        ax.set_ylabel("")
    ax.grid(alpha=0.2)
    ax.tick_params(labelsize=8)
    return float(r), float(p), int(x_full.size)


def write_percell_scatter_summary(
    tiles_dir: Path,
    out_csv: Path,
    models: list[tuple[str, str]],
    markers: list[str],
    mode: str = "percell",
) -> Path:
    """Write per-model Pearson r / p / n from tile TIFFs (same source as scatter plots)."""
    from scipy.stats import pearsonr

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model_key", "model_label", "marker", "mode", "n", "pearson_r", "p_value"])
        for key, label in models:
            pools = _collect_intensities(tiles_dir, key, mode)
            for marker in markers:
                x, y = pools.get(marker, (np.array([]), np.array([])))
                if x.size >= 3:
                    r, p = pearsonr(x, y)
                    w.writerow([key, label, marker, mode, x.size, f"{r:.6f}", f"{p:.6g}"])
                else:
                    w.writerow([key, label, marker, mode, x.size, "", ""])
    print(f"Wrote {out_csv}")
    return out_csv


def build_per_marker_pearson_scatter(
    tiles_dir: Path,
    out_path: Path,
    *,
    model_key: str = CROSS_ATTN_KEY,
    model_label: str = "Ours",
    mode: str = "percell",
    markers: list[str] | None = None,
    max_plot_points: int = 25000,
    seed: int = 42,
    dpi: int = 200,
) -> bool:
    """Scatter: GT vs predicted intensity per marker, y=x line, r (+ p if p < 0.05)."""
    marker_filter = markers or ["cd3", "panck"]
    specs = [s for s in MARKER_SPECS if s[0] in marker_filter]
    if not specs:
        print("SKIP per-marker pearson scatter: no markers selected")
        return False

    pools = _collect_intensities(tiles_dir, model_key, mode)
    if all(pools[m][0].size == 0 for m, _, _ in specs):
        print(f"SKIP per-marker pearson scatter: no data for {model_key}")
        return False

    color = SCATTER_COLORS.get(model_label, SCATTER_DEFAULT_COLOR)
    ncols = len(specs)
    fig, axes = plt.subplots(1, ncols, figsize=(3.8 * ncols, 3.8), facecolor="white", squeeze=False)
    fig.subplots_adjust(top=0.82, bottom=0.16, wspace=0.28)
    for ax, (marker, title, _) in zip(axes[0], specs):
        _plot_scatter_panel(
            ax, pools[marker][0], pools[marker][1],
            color=color, max_plot_points=max_plot_points, seed=seed, title=title,
        )

    mode_note = "per-cell" if mode == "percell" else "per-pixel"
    fig.suptitle(
        f"Per-Cell Intensity Correlation ({model_label}, {mode_note})",
        fontsize=12, fontweight="bold", fontfamily=FONT,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out_path}")
    return True


def build_per_marker_scatter_comparison(
    tiles_dir: Path,
    out_path: Path,
    models: list[tuple[str, str]] | None = None,
    *,
    mode: str = "percell",
    markers: list[str] | None = None,
    max_plot_points: int = 8000,
    seed: int = 42,
    dpi: int = 200,
) -> bool:
    """Grid: rows = markers, cols = models — all computed from tile TIFF pairs."""
    from scipy.stats import pearsonr  # noqa: F401 — used in _plot_scatter_panel

    models = models or SCATTER_MODELS
    marker_filter = markers or ["cd3", "panck"]
    specs = [s for s in MARKER_SPECS if s[0] in marker_filter]
    if not specs:
        return False

    available = [(k, lab) for k, lab in models if list(_resolve_tiles_dir(tiles_dir).glob(f"{k}__*_fake_B.tif"))]
    if not available:
        print("SKIP scatter comparison: no model tiles found")
        return False

    nrows, ncols = len(specs), len(available)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(2.9 * ncols, 3.0 * nrows),
        facecolor="white",
        squeeze=False,
    )
    fig.subplots_adjust(top=0.90, bottom=0.10, left=0.07, right=0.99, hspace=0.35, wspace=0.28)

    for j, (key, label) in enumerate(available):
        pools = _collect_intensities(tiles_dir, key, mode)
        color = SCATTER_COLORS.get(label, SCATTER_DEFAULT_COLOR)
        for i, (marker, mtitle, _) in enumerate(specs):
            ax = axes[i, j]
            _plot_scatter_panel(
                ax, pools[marker][0], pools[marker][1],
                color=color, max_plot_points=max_plot_points, seed=seed + j,
                title=label if i == 0 else None,
                show_xlabel=i == nrows - 1,
                show_ylabel=j == 0,
            )
            if j == 0:
                ax.text(
                    -0.35, 0.5, mtitle, transform=ax.transAxes,
                    fontsize=10, fontweight="bold", va="center", ha="right", rotation=90, fontfamily=FONT,
                )

    mode_note = "per-cell" if mode == "percell" else "per-pixel"
    fig.suptitle(
        f"Per-Cell Intensity Correlation — All Methods ({mode_note})",
        fontsize=12, fontweight="bold", fontfamily=FONT,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out_path}")
    return True


def build_per_marker_pearson_scatter_all(
    tiles_dir: Path,
    out_dir: Path,
    models: list[tuple[str, str]] | None = None,
    *,
    mode: str = "percell",
    markers: list[str] | None = None,
    dpi: int = 200,
) -> list[Path]:
    """One scatter PNG per model + summary CSV."""
    models = models or SCATTER_MODELS
    marker_list = markers or ["cd3", "panck"]
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for key, label in models:
        slug = key.replace("hemit_", "").replace("_512", "")
        out = out_dir / f"fig_percell_scatter_{slug}.png"
        if build_per_marker_pearson_scatter(
            tiles_dir, out, model_key=key, model_label=label,
            mode=mode, markers=marker_list, dpi=dpi,
        ):
            saved.append(out)
    write_percell_scatter_summary(
        tiles_dir, out_dir / "percell_scatter_summary.csv", models, marker_list, mode=mode,
    )
    return saved


def build_per_marker_pearson_lines(metrics_dir: Path, out_path: Path) -> None:
    """Line plot: Pearson r per marker for Vanilla vs Cross-Attn vs Cross-Attn+Vel."""
    series: list[tuple[str, Path, str]] = [
        ("Vanilla FM", metrics_dir / "vanilla.csv", "#78909C"),
        ("+ Cross-Attn", metrics_dir / "score-2.csv", "#1565C0"),
        ("+ Cross-Attn + Vel", metrics_dir / "score-3.csv", "#2E7D32"),
    ]
    markers = ["DAPI", "CD3", "panCK"]
    x = np.arange(len(markers))

    fig, ax = plt.subplots(figsize=(7, 4.2), facecolor="white")
    fig.subplots_adjust(top=0.82, bottom=0.14)

    for label, path, color in series:
        if not path.is_file():
            print(f"  missing {path}")
            continue
        vals = _mean_marker_pearson(path)
        ax.plot(x, vals, marker="o", linewidth=2.2, markersize=8, label=label, color=color)
        for xi, v in zip(x, vals):
            ax.text(xi, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=9, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(markers)
    ax.set_ylabel("Mean Pearson r (n = 945)", fontweight="bold", fontfamily=FONT)
    ax.set_ylim(0.45, 1.02)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="lower right")
    fig.suptitle(
        "Per-Marker Pearson Correlation Across Model Variants",
        fontsize=12, fontweight="bold", fontfamily=FONT,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out_path}")


def build_workflow_panels(tiles_dir: Path, out_path: Path, tile: str, dpi: int = 200) -> bool:
    """Workflow strip with real H&E, GT channels, and Ours prediction."""
    he_p = find_tile_path(tiles_dir, f"GT__{tile}_real_A.tif")
    gt_p = find_tile_path(tiles_dir, f"GT__{tile}_real_B.tif")
    ours_p = find_tile_path(tiles_dir, f"{CROSS_ATTN_KEY}__{tile}_fake_B.tif")
    if not all([he_p, gt_p, ours_p]):
        print("SKIP workflow panels")
        return False

    gt_arr = load_mif_array(gt_p)
    panels = [
        ("1. Input H&E", load_he(he_p), False),
        ("2. GT DAPI", marker_to_rgb(gt_arr[..., 0], "dapi", enhance=True), True),
        ("2. GT CD3", marker_to_rgb(gt_arr[..., 1], "cd3", enhance=True), True),
        ("2. GT panCK", marker_to_rgb(gt_arr[..., 2], "panck", enhance=True), True),
        ("2. GT Composite", tif_to_mif_rgb(gt_p), True),
        ("4. Virtual mIF (Ours)", tif_to_mif_rgb(ours_p), True),
    ]

    fig = plt.figure(figsize=(16, 3.2), facecolor="white")
    gs = gridspec.GridSpec(1, len(panels), figure=fig, wspace=0.04, top=0.78, bottom=0.05)
    for i, (title, img, mif) in enumerate(panels):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(img)
        ax.axis("off")
        _border(ax, BORDER_HE if i == 0 else BORDER_MIF)
        ax.set_title(title, fontsize=9, fontweight="bold", pad=4, fontfamily=FONT)
    fig.suptitle("H&E-to-mIF Virtual Staining Workflow (Real Test Patch)", fontsize=12, fontweight="bold", fontfamily=FONT)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out_path}")
    return True


def build_vanilla_vs_ours_strip(tiles_dir: Path, out_path: Path, tile: str, dpi: int = 200) -> bool:
    """Impact panel: H&E | GT | Vanilla | Ours (composite)."""
    paths = {
        "H&E": find_tile_path(tiles_dir, f"GT__{tile}_real_A.tif"),
        "Ground Truth": find_tile_path(tiles_dir, f"GT__{tile}_real_B.tif"),
        "Vanilla FM": find_tile_path(tiles_dir, f"{VANILLA_KEY}__{tile}_fake_B.tif"),
        "Ours (+ Cross-Attn)": find_tile_path(tiles_dir, f"{CROSS_ATTN_KEY}__{tile}_fake_B.tif"),
    }
    if any(v is None for v in paths.values()):
        print("SKIP vanilla vs ours strip")
        return False

    fig, axes = plt.subplots(1, 4, figsize=(12, 3.4), facecolor="white")
    fig.subplots_adjust(top=0.82, wspace=0.05)
    for ax, (title, p) in zip(axes, paths.items()):
        ax.axis("off")
        if title == "H&E":
            ax.imshow(load_he(p))
            _border(ax, BORDER_HE)
        else:
            ax.imshow(tif_to_mif_rgb(p))
            _border(ax, BORDER_MIF)
        ax.set_title(title, fontsize=10, fontweight="bold", fontfamily=FONT)
    fig.suptitle(
        "Impact of H&E Cross-Attention (Composite mIF)",
        fontsize=12, fontweight="bold", fontfamily=FONT,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out_path}")
    return True


def main() -> None:
    apply_plot_style()
    p = argparse.ArgumentParser()
    p.add_argument("--tiles-dir", type=Path, default=Path.home() / "Downloads/new_tiles-2")
    p.add_argument("--metrics-dir", type=Path, default=Path.home() / "Downloads")
    p.add_argument("--out", type=Path, default=Path("figures/hemit/hemit 2"))
    p.add_argument("--tile", type=str, default=DETAIL_TILE, help="Tile for workflow / error maps / vanilla strip")
    p.add_argument("--ablation-tile", type=str, default=ABLATION_TILE, help="Tile for ablation side-by-side")
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--model-key", type=str, default=CROSS_ATTN_KEY, help="Tile prefix for scatter correlation")
    p.add_argument(
        "--correlation-mode", choices=("pixels", "percell"), default="percell",
        help="percell = per-nucleus means (default); pixels = per-pixel pooled",
    )
    p.add_argument(
        "--markers", type=str, default="cd3,panck",
        help="Comma-separated markers to plot (default: cd3,panck — no DAPI)",
    )
    p.add_argument(
        "--scatter-all-models", action="store_true",
        help="Also build per-model scatters + all-methods grid + summary CSV",
    )
    args = p.parse_args()

    out = args.out
    tile = args.tile
    marker_list = [m.strip().lower() for m in args.markers.split(",") if m.strip()]
    build_ablation_sidebyside(args.tiles_dir, out / "fig_ablation_sidebyside.png", args.ablation_tile, args.dpi)
    build_ablation_zoom_deep(
        args.tiles_dir, out / "fig_ablation_zoom_deep.png",
        tile=args.ablation_tile, dpi=args.dpi,
    )
    build_error_maps(args.tiles_dir, out / "fig_error_maps.png", tile, args.dpi)
    build_per_marker_pearson_scatter(
        args.tiles_dir, out / "fig_per_marker_pearson_scatter.png",
        model_key=args.model_key, mode=args.correlation_mode,
        markers=marker_list, dpi=args.dpi,
    )
    if args.scatter_all_models:
        scatter_dir = out / "percell_scatter"
        build_per_marker_scatter_comparison(
            args.tiles_dir, out / "fig_percell_scatter_all_models.png",
            mode=args.correlation_mode, markers=marker_list, dpi=args.dpi,
        )
        build_per_marker_pearson_scatter_all(
            args.tiles_dir, scatter_dir,
            mode=args.correlation_mode, markers=marker_list, dpi=args.dpi,
        )
    build_workflow_panels(args.tiles_dir, out / "fig_workflow_real.png", tile, args.dpi)
    build_vanilla_vs_ours_strip(args.tiles_dir, out / "fig_vanilla_vs_crossattn.png", tile, args.dpi)


if __name__ == "__main__":
    main()
