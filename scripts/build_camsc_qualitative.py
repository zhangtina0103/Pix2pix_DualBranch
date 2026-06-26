#!/usr/bin/env python3
"""
Qualitative CaMSC figures: brightfield -> Hoechst + WT1 virtual staining.

CaMSC target channels (from prepare_camsc_bf.py): [Hoechst, WT1, pad] as R/G/B.
Rendering convention here: Hoechst -> blue, WT1 -> green (pad channel ignored).

Two figure modes (both run if inputs are available):

1) DETAIL (GT vs Ours): rows = Ground Truth, Ours; cols = BF | composite | Hoechst | WT1.
   Needs --ours-dir (a test_<epoch>/images folder with *_real_A/_real_B/_fake_B.tif).

2) COMPARISON (multi-model): rows = tiles; cols = BF | GT | each model composite.
   Provide one or more --model "Label=/path/to/test_<epoch>/images".

Tile selection: --tiles stem1 stem2 ... ; otherwise auto-pick the tiles with the
most WT1 signal (most informative for a sparse marker).

Examples
--------
Detail (GT vs Ours):
  python scripts/build_camsc_qualitative.py \
    --ours-dir results/camsc_bf_fm_cross_attn_ft_fold0_512_aug_512/test_110/images \
    --out-dir figures/camsc

Multi-model comparison:
  python scripts/build_camsc_qualitative.py \
    --model "Ours=results/camsc_bf_fm_cross_attn_ft_fold0_512_aug_512/test_110/images" \
    --model "Pix2Pix=results/camsc_bf_pix2pix_ft_fold0_512_aug/test_110/images" \
    --model "CUT=results/camsc_bf_cut_ft_fold0_512_aug/test_110/images" \
    --model "ASP=results/camsc_bf_asp_ft_fold0_512_aug/test_110/images" \
    --out-dir figures/camsc
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from PIL import Image

FONT = "Arial"
BORDER_BF = "#5D4037"     # brown — brightfield
BORDER_FL = "#F9A825"     # amber — fluorescence


def apply_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [FONT, "Helvetica", "DejaVu Sans"],
        "font.size": 11,
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


def load_bf(path: Path, size: int = 512) -> np.ndarray:
    arr = np.asarray(Image.open(path))
    arr = _to_255(arr).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[-1] > 3:
        arr = arr[..., :3]
    if arr.shape[0] != size or arr.shape[1] != size:
        arr = np.asarray(Image.fromarray(arr).resize((size, size), Image.BILINEAR))
    return arr


def load_camsc_array(path: Path) -> np.ndarray:
    """Return (H,W,2) float [0,255]: [Hoechst, WT1]."""
    arr = np.asarray(Image.open(path))
    arr = _to_255(arr)
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
    if color == "blue":
        rgb[..., 2] = ch
    elif color == "green":
        rgb[..., 1] = ch
    else:
        rgb[..., 0] = ch
    return rgb


def composite_rgb(arr2: np.ndarray, enhance: bool = True) -> np.ndarray:
    """Hoechst -> blue, WT1 -> green."""
    hoechst = marker_rgb(arr2[..., 0], "blue", enhance)
    wt1 = marker_rgb(arr2[..., 1], "green", enhance)
    return np.clip(hoechst.astype(np.uint16) + wt1.astype(np.uint16), 0, 255).astype(np.uint8)


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
        wt1 = load_camsc_array(real_b)[..., 1]
        scored.append((float(np.mean(wt1)), stem))
    scored.sort(reverse=True)
    return [s for _, s in scored[:n]]


def _style_axis(ax, border: str) -> None:
    ax.axis("off")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(border)
        spine.set_linewidth(1.2)


def find_bf(images_dir: Path, stem: str) -> Path | None:
    for cand in (f"{stem}_real_A.tif", f"{stem}_real_A.png"):
        p = images_dir / cand
        if p.is_file():
            return p
    return None


def build_detail(ours_dir: Path, out_path: Path, tiles: list[str], dpi: int) -> bool:
    stem = tiles[0]
    bf_p = find_bf(ours_dir, stem)
    gt_p = ours_dir / f"{stem}_real_B.tif"
    pr_p = ours_dir / f"{stem}_fake_B.tif"
    if not (gt_p.is_file() and pr_p.is_file()):
        print(f"SKIP detail: missing GT/pred for {stem} in {ours_dir}")
        return False

    bf = load_bf(bf_p) if bf_p else np.zeros((512, 512, 3), np.uint8)
    gt = load_camsc_array(gt_p)
    pr = load_camsc_array(pr_p)

    cols = ["Brightfield", "Composite", "Hoechst", "WT1"]
    rows = [("Ground Truth", gt), ("Ours (FM + Cross-Attn)", pr)]

    fig = plt.figure(figsize=(11.0, 5.6), facecolor="white")
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.08, wspace=0.05,
                           top=0.86, bottom=0.05, left=0.09, right=0.99)
    for j, (row_label, arr) in enumerate(rows):
        imgs = [bf, composite_rgb(arr), marker_rgb(arr[..., 0], "blue"), marker_rgb(arr[..., 1], "green")]
        for i, img in enumerate(imgs):
            ax = fig.add_subplot(gs[j, i])
            ax.imshow(img)
            _style_axis(ax, BORDER_BF if i == 0 else BORDER_FL)
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


def build_comparison(models: list[tuple[str, Path]], out_path: Path,
                     tiles: list[str], dpi: int) -> bool:
    if not models:
        return False
    ref_dir = models[0][1]
    cols = ["Brightfield", "GT"] + [label for label, _ in models]
    n_rows, n_cols = len(tiles), len(cols)

    fig = plt.figure(figsize=(2.1 * n_cols, 2.3 * n_rows), facecolor="white")
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.06, wspace=0.04,
                           top=0.92, bottom=0.02, left=0.02, right=0.99)
    for r, stem in enumerate(tiles):
        bf_p = find_bf(ref_dir, stem)
        bf = load_bf(bf_p) if bf_p else np.zeros((512, 512, 3), np.uint8)
        gt_p = ref_dir / f"{stem}_real_B.tif"
        gt = composite_rgb(load_camsc_array(gt_p)) if gt_p.is_file() else np.zeros((512, 512, 3), np.uint8)
        panels = [(bf, BORDER_BF), (gt, BORDER_FL)]
        for _, mdir in models:
            fp = mdir / f"{stem}_fake_B.tif"
            img = composite_rgb(load_camsc_array(fp)) if fp.is_file() else np.zeros((512, 512, 3), np.uint8)
            panels.append((img, BORDER_FL))
        for c, (img, border) in enumerate(panels):
            ax = fig.add_subplot(gs[r, c])
            ax.imshow(img)
            _style_axis(ax, border)
            if r == 0:
                ax.set_title(cols[c], fontsize=11, fontweight="bold", pad=5, fontfamily=FONT)

    fig.suptitle("CaMSC Virtual Staining: Brightfield \u2192 Hoechst + WT1 (Composite)",
                 fontsize=13, fontweight="bold", fontfamily=FONT)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)
    print(f"Wrote {out_path}")
    return True


def main() -> None:
    p = argparse.ArgumentParser(description="CaMSC qualitative figures")
    p.add_argument("--ours-dir", default="",
                   help="images dir for GT-vs-Ours detail (test_<epoch>/images)")
    p.add_argument("--model", action="append", default=[],
                   help="Label=images_dir for multi-model comparison (repeatable)")
    p.add_argument("--tiles", nargs="*", default=None, help="Explicit tile stems")
    p.add_argument("--n-tiles", type=int, default=4, help="Auto-picked tiles for comparison")
    p.add_argument("--out-dir", default="figures/camsc")
    p.add_argument("--dpi", type=int, default=200)
    args = p.parse_args()

    apply_style()
    out_dir = Path(args.out_dir).expanduser()

    models: list[tuple[str, Path]] = []
    for spec in args.model:
        if "=" not in spec:
            raise SystemExit(f"--model must be Label=dir, got {spec}")
        label, d = spec.split("=", 1)
        models.append((label.strip(), Path(d.strip()).expanduser()))

    # DETAIL
    ours_dir = Path(args.ours_dir).expanduser() if args.ours_dir else (
        models[0][1] if models else None)
    if ours_dir and ours_dir.is_dir():
        det_tiles = pick_tiles(ours_dir, 1, args.tiles)
        if det_tiles:
            build_detail(ours_dir, out_dir / "fig_camsc_detail.png", det_tiles, args.dpi)
        else:
            print(f"[skip] detail: no tiles found in {ours_dir}")
    else:
        print("[skip] detail: provide --ours-dir or at least one --model")

    # COMPARISON
    if models:
        cmp_tiles = pick_tiles(models[0][1], args.n_tiles, args.tiles)
        if cmp_tiles:
            build_comparison(models, out_dir / "fig_camsc_comparison.png", cmp_tiles, args.dpi)
        else:
            print(f"[skip] comparison: no tiles found in {models[0][1]}")
    else:
        print("[skip] comparison: pass --model Label=dir (>=1)")


if __name__ == "__main__":
    main()
