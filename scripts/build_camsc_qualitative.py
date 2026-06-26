#!/usr/bin/env python3
"""
Qualitative CaMSC figures: brightfield -> Hoechst + WT1 virtual staining.

CaMSC target channels (prepare_camsc_bf.py): [Hoechst, WT1, pad] as R/G/B.
Render convention: Hoechst -> blue, WT1 -> green (pad ignored).

Outputs (when inputs available):
  fig_camsc_detail.png          GT vs Ours, full tile: BF | Composite | Hoechst | WT1
  fig_camsc_comparison.png      rows=tiles, cols=BF|GT|models (full-tile composite)
  fig_camsc_zoom_composite.png  rows=tiles, cols=BF|GT|models (ZOOMED composite)
  fig_camsc_zoom_wt1.png        rows=tiles, cols=BF|GT|models (ZOOMED WT1 only)

Model result dirs are auto-discovered under --results-root by fold/epoch, so you
don't need to know the exact "_512_aug" / "_512_aug_512" suffixes. Missing
per-tile files render as a labeled GRAY panel (not black) so you can tell
"file not found" from "model output is truly empty/collapsed".

Examples
--------
Auto-discover all finetuned models for fold 0 @ epoch 110:
  python scripts/build_camsc_qualitative.py --results-root results --fold 0 --epoch 110 \
    --out-dir figures/camsc

Explicit dirs (override discovery):
  python scripts/build_camsc_qualitative.py \
    --model "Ours=results/camsc_bf_fm_cross_attn_ft_fold0_512_aug_512/test_110/images" \
    --model "Pix2Pix=results/camsc_bf_pix2pix_ft_fold0_512_aug/test_110/images" \
    --out-dir figures/camsc
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

FONT = "Arial"
BORDER_BF = "#5D4037"     # brown — brightfield
BORDER_FL = "#F9A825"     # amber — fluorescence
BORDER_ZOOM = "#FFFFFF"

# (display label, result-name key) — auto-discovery order = column order
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
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


# ---------------------------------------------------------------------------
# IO / rendering
# ---------------------------------------------------------------------------

def _to_255(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    if arr.min() < 0:
        arr = (arr + 1.0) / 2.0 * 255.0
    elif arr.max() <= 1.0:
        arr = arr * 255.0
    return np.clip(arr, 0, 255)


def load_bf(path: Path, size: int = 512) -> np.ndarray:
    arr = _to_255(np.asarray(Image.open(path))).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[-1] > 3:
        arr = arr[..., :3]
    if arr.shape[0] != size or arr.shape[1] != size:
        arr = np.asarray(Image.fromarray(arr).resize((size, size), Image.BILINEAR))
    return arr


def load_camsc_array(path: Path) -> np.ndarray:
    """Return (H,W,2) float [0,255]: [Hoechst, WT1]."""
    arr = _to_255(np.asarray(Image.open(path)))
    if arr.ndim == 2:
        arr = np.stack([arr, arr], axis=-1)
    return arr[..., :2]


def _stretch(ch: np.ndarray) -> np.ndarray:
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


def marker_rgb(ch: np.ndarray, color: str, enhance: bool = True) -> np.ndarray:
    ch = _stretch(ch) if enhance else ch.astype(np.float64)
    ch = np.clip(ch, 0, 255).astype(np.uint8)
    rgb = np.zeros((*ch.shape, 3), dtype=np.uint8)
    rgb[..., {"red": 0, "green": 1, "blue": 2}[color]] = ch
    return rgb


def composite_rgb(arr2: np.ndarray, enhance: bool = True) -> np.ndarray:
    """Hoechst -> blue, WT1 -> green."""
    h = marker_rgb(arr2[..., 0], "blue", enhance).astype(np.uint16)
    w = marker_rgb(arr2[..., 1], "green", enhance).astype(np.uint16)
    return np.clip(h + w, 0, 255).astype(np.uint8)


def missing_panel(size: int = 512) -> np.ndarray:
    """Gray placeholder so 'file not found' != 'collapsed black output'."""
    return np.full((size, size, 3), 60, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Tile + crop helpers
# ---------------------------------------------------------------------------

def fake_stem(p: Path) -> str:
    return p.name[:-len("_fake_B.tif")]


def list_tiles(images_dir: Path) -> list[str]:
    return sorted(fake_stem(p) for p in images_dir.glob("*_fake_B.tif"))


def pick_tiles(images_dir: Path, n: int, explicit: list[str] | None) -> list[str]:
    if explicit:
        return explicit
    scored = []
    for stem in list_tiles(images_dir):
        real_b = images_dir / f"{stem}_real_B.tif"
        if not real_b.is_file():
            continue
        arr = load_camsc_array(real_b)
        # informative = strong WT1 AND visible nuclei
        scored.append((float(np.mean(arr[..., 1]) + 0.3 * np.mean(arr[..., 0])), stem))
    scored.sort(reverse=True)
    return [s for _, s in scored[:n]]


def auto_zoom_box(channel: np.ndarray, box: int) -> tuple[int, int, int, int]:
    """Center a box on the densest region of `channel`."""
    h, w = channel.shape
    box = min(box, h, w)
    try:
        from scipy.ndimage import uniform_filter
        score = uniform_filter(channel.astype(np.float64), size=box, mode="constant")
    except Exception:
        score = channel.astype(np.float64)
    y, x = np.unravel_index(int(np.argmax(score)), score.shape)
    x0 = int(np.clip(x - box // 2, 0, w - box))
    y0 = int(np.clip(y - box // 2, 0, h - box))
    return x0, y0, box, box


def crop(img: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x, y, bw, bh = box
    return img[y:y + bh, x:x + bw]


def _style_axis(ax, border: str, lw: float = 1.2) -> None:
    ax.axis("off")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(border)
        spine.set_linewidth(lw)


def find_bf(images_dir: Path, stem: str, fallback_dirs: list[Path]) -> Path | None:
    for d in [images_dir, *fallback_dirs]:
        for cand in (f"{stem}_real_A.tif", f"{stem}_real_A.png", f"{stem}.tif"):
            p = d / cand
            if p.is_file():
                return p
    return None


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_model_dirs(results_root: Path, fold: int, epoch: int,
                        models: list[tuple[str, str]]) -> list[tuple[str, Path]]:
    found: list[tuple[str, Path]] = []
    for label, key in models:
        matches = sorted(results_root.glob(f"camsc_bf_{key}_fold{fold}*/test_{epoch}/images"))
        matches = [m for m in matches if any(m.glob("*_fake_B.tif"))]
        if matches:
            found.append((label, matches[0]))
            print(f"  [ok] {label:<10} -> {matches[0]}")
        else:
            print(f"  [MISS] {label:<10} no dir: "
                  f"{results_root}/camsc_bf_{key}_fold{fold}*/test_{epoch}/images", file=sys.stderr)
    return found


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def build_detail(ours_dir: Path, out_path: Path, stem: str, dpi: int,
                 fallback_bf: list[Path], zoom_box: int) -> bool:
    gt_p = ours_dir / f"{stem}_real_B.tif"
    pr_p = ours_dir / f"{stem}_fake_B.tif"
    if not (gt_p.is_file() and pr_p.is_file()):
        print(f"SKIP detail: missing GT/pred for {stem}", file=sys.stderr)
        return False
    bf_p = find_bf(ours_dir, stem, fallback_bf)
    bf = load_bf(bf_p) if bf_p else missing_panel()
    gt = load_camsc_array(gt_p)
    pr = load_camsc_array(pr_p)
    box = auto_zoom_box(gt[..., 1], zoom_box)

    cols = ["Brightfield", "Composite", "Hoechst", "WT1", "WT1 (zoom)"]
    rows = [("Ground Truth", gt), ("Ours (FM + Cross-Attn)", pr)]

    fig = plt.figure(figsize=(13.5, 5.6), facecolor="white")
    gs = gridspec.GridSpec(2, 5, figure=fig, hspace=0.08, wspace=0.05,
                           top=0.86, bottom=0.05, left=0.085, right=0.99)
    for j, (row_label, arr) in enumerate(rows):
        comp = composite_rgb(arr)
        imgs = [bf, comp, marker_rgb(arr[..., 0], "blue"),
                marker_rgb(arr[..., 1], "green"),
                crop(marker_rgb(arr[..., 1], "green"), box)]
        for i, img in enumerate(imgs):
            ax = fig.add_subplot(gs[j, i])
            ax.imshow(img, interpolation="nearest" if i == 4 else "bilinear")
            _style_axis(ax, BORDER_BF if i == 0 else (BORDER_ZOOM if i == 4 else BORDER_FL))
            if i == 3:  # draw zoom box on full WT1
                x, y, bw, bh = box
                ax.add_patch(Rectangle((x, y), bw, bh, linewidth=1.2,
                                       edgecolor=BORDER_ZOOM, facecolor="none"))
            if j == 0:
                ax.set_title(cols[i], fontsize=11, fontweight="bold", pad=5, fontfamily=FONT)
        fig.text(0.03, 0.66 - j * 0.42, row_label, fontsize=11, fontweight="bold",
                 va="center", rotation=90, fontfamily=FONT)

    fig.suptitle("CaMSC Brightfield \u2192 Hoechst + WT1 (Ground Truth vs. Ours)",
                 fontsize=13, fontweight="bold", fontfamily=FONT)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)
    print(f"Wrote {out_path}")
    return True


def _grid_figure(models, tiles, out_path, dpi, fallback_bf, *,
                 mode: str, zoom_box: int, title: str) -> bool:
    """mode: 'full_comp' | 'zoom_comp' | 'zoom_wt1'."""
    if not models or not tiles:
        return False
    ref_dir = models[0][1]
    cols = ["Brightfield", "GT"] + [lbl for lbl, _ in models]
    n_rows, n_cols = len(tiles), len(cols)
    zoom = mode != "full_comp"

    fig = plt.figure(figsize=(2.1 * n_cols, 2.25 * n_rows + 0.3), facecolor="white")
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.05, wspace=0.04,
                           top=0.93, bottom=0.02, left=0.025, right=0.99)

    def render(arr2: np.ndarray, box) -> np.ndarray:
        if mode == "zoom_wt1":
            img = marker_rgb(arr2[..., 1], "green")
        else:
            img = composite_rgb(arr2)
        return crop(img, box) if zoom else img

    for r, stem in enumerate(tiles):
        gt_p = ref_dir / f"{stem}_real_B.tif"
        gt_arr = load_camsc_array(gt_p) if gt_p.is_file() else None
        box = auto_zoom_box(gt_arr[..., 1], zoom_box) if (zoom and gt_arr is not None) else (0, 0, 512, 512)

        bf_p = find_bf(ref_dir, stem, fallback_bf)
        bf = load_bf(bf_p) if bf_p else missing_panel()
        bf_show = crop(bf, box) if zoom else bf
        gt_show = render(gt_arr, box) if gt_arr is not None else missing_panel(box[2] if zoom else 512)

        panels = [(bf_show, BORDER_BF), (gt_show, BORDER_FL)]
        for _, mdir in models:
            fp = mdir / f"{stem}_fake_B.tif"
            if fp.is_file():
                panels.append((render(load_camsc_array(fp), box), BORDER_FL))
            else:
                print(f"  [missing] {fp}", file=sys.stderr)
                panels.append((missing_panel(box[2] if zoom else 512), "#B71C1C"))

        for c, (img, border) in enumerate(panels):
            ax = fig.add_subplot(gs[r, c])
            ax.imshow(img, interpolation="nearest" if zoom else "bilinear")
            _style_axis(ax, border)
            if r == 0:
                ax.set_title(cols[c], fontsize=11, fontweight="bold", pad=5, fontfamily=FONT)

    fig.suptitle(title, fontsize=13, fontweight="bold", fontfamily=FONT)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)
    print(f"Wrote {out_path}")
    return True


def main() -> None:
    p = argparse.ArgumentParser(description="CaMSC qualitative figures (detail + zoom across models)")
    p.add_argument("--results-root", default="results")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--epoch", type=int, default=110)
    p.add_argument("--model", action="append", default=[],
                   help="Label=images_dir (repeatable); overrides auto-discovery")
    p.add_argument("--kfold-root", default="",
                   help="CaMSC k-fold root for BF fallback (e.g. .../camsc_bf_kfold_aug)")
    p.add_argument("--tiles", nargs="*", default=None, help="Explicit tile stems")
    p.add_argument("--n-tiles", type=int, default=4)
    p.add_argument("--zoom-box", type=int, default=170, help="Zoom crop size in px (of 512)")
    p.add_argument("--out-dir", default="figures/camsc")
    p.add_argument("--dpi", type=int, default=220)
    args = p.parse_args()

    apply_style()
    out_dir = Path(args.out_dir).expanduser()
    results_root = Path(args.results_root).expanduser()

    if args.model:
        models = []
        for spec in args.model:
            if "=" not in spec:
                raise SystemExit(f"--model must be Label=dir, got {spec}")
            label, d = spec.split("=", 1)
            models.append((label.strip(), Path(d.strip()).expanduser()))
    else:
        print(f"Auto-discovering CaMSC results under {results_root} (fold {args.fold}, epoch {args.epoch}):")
        models = discover_model_dirs(results_root, args.fold, args.epoch, DEFAULT_MODELS)

    if not models:
        raise SystemExit("No model result dirs found. Check --results-root/--fold/--epoch or pass --model.")

    # BF fallback dirs (test.py may not save real_A for CaMSC)
    fallback_bf: list[Path] = []
    if args.kfold_root:
        fallback_bf.append(Path(args.kfold_root).expanduser() / f"fold{args.fold}" / "testA")

    ref_dir = models[0][1]
    tiles = pick_tiles(ref_dir, args.n_tiles, args.tiles)
    if not tiles:
        raise SystemExit(f"No tiles found in {ref_dir}")
    print(f"Tiles: {tiles}")

    # 1) Detail (GT vs Ours) — best single tile
    build_detail(ref_dir, out_dir / "fig_camsc_detail.png", tiles[0], args.dpi,
                 fallback_bf, args.zoom_box)

    # 2) Full-tile comparison
    _grid_figure(models, tiles, out_dir / "fig_camsc_comparison.png", args.dpi, fallback_bf,
                 mode="full_comp", zoom_box=args.zoom_box,
                 title="CaMSC Virtual Staining: BF \u2192 Hoechst + WT1 (Composite)")

    # 3) Zoom composite across models
    _grid_figure(models, tiles, out_dir / "fig_camsc_zoom_composite.png", args.dpi, fallback_bf,
                 mode="zoom_comp", zoom_box=args.zoom_box,
                 title="CaMSC Zoom: Hoechst + WT1 Composite (per-row zoom on WT1-dense region)")

    # 4) Zoom WT1-only across models — shows baseline collapse
    _grid_figure(models, tiles, out_dir / "fig_camsc_zoom_wt1.png", args.dpi, fallback_bf,
                 mode="zoom_wt1", zoom_box=args.zoom_box,
                 title="CaMSC Zoom: WT1 channel only (sparse marker; baselines collapse)")


if __name__ == "__main__":
    main()
