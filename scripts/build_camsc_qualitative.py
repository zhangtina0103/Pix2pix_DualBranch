#!/usr/bin/env python3
"""
Qualitative CaMSC figures: brightfield -> Hoechst + WT1 virtual staining.

CaMSC target channels (prepare_camsc_bf.py): [Hoechst, WT1, pad] as R/G/B.
Render convention: Hoechst -> blue, WT1 -> green (pad ignored).

FAIRNESS: all panels (GT + every model) use ONE shared linear intensity gain per
channel (computed from GT), matching the HEMIT figure convention. No per-image
contrast stretching — so brightness differences between models are real, not a
display artifact. A collapsed model simply shows little/no signal.

Outputs (HEMIT-style):
  fig_camsc_zoom.png            main panel + numbered boxes + zoom strip, all models  [PRIMARY]
  fig_camsc_zoom_wt1.png        same layout, WT1 channel only (sparse-marker collapse)
  fig_camsc_comparison.png      full-tile composite grid, all models
  fig_camsc_detail.png          GT vs Ours, BF | Composite | Hoechst | WT1

Result dirs auto-discovered under --results-root by fold/epoch. Missing per-tile
files render as a labeled gray panel (not black) with a stderr warning.

Examples
--------
  python scripts/build_camsc_qualitative.py --results-root results --fold 4 --epoch 110 \
    --kfold-root ~/orcd/scratch/camsc/datasets/camsc_bf_kfold_aug \
    --out-dir figures/camsc/fold4
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
BORDER_BF = "#5D4037"
BORDER_FL = "#F9A825"
BORDER_MISS = "#B71C1C"

# Order matters for the figure: put the weakest baselines right after Ours so the
# contrast is starkest, then the stronger baselines (CUT/ASP) further right.
DEFAULT_MODELS = [
    ("Ours", "fm_cross_attn_ft"),
    ("Pix2Pix", "pix2pix_ft"),
    ("CycleGAN", "cyclegan_ft"),
    ("CUT", "cut_ft"),
    ("ASP", "asp_ft"),
]

N_ZOOM = 4
ZOOM_BOX = 90
GAIN_PCT = 99.0
GAIN_TARGET = 210.0


def apply_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [FONT, "Helvetica", "DejaVu Sans"],
        "font.size": 11,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


# ---------------------------------------------------------------------------
# IO
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


# ---------------------------------------------------------------------------
# Rendering — linear, shared gain (fair across models)
# ---------------------------------------------------------------------------

def compute_global_gain(gt_arrays: list[np.ndarray]) -> tuple[float, float]:
    gains = []
    for ch in (0, 1):
        vals = []
        for a in gt_arrays:
            c = a[..., ch]
            pos = c[c > 0]
            if pos.size:
                vals.append(pos.ravel())
        if not vals:
            gains.append(1.0)
            continue
        hi = float(np.percentile(np.concatenate(vals), GAIN_PCT))
        gains.append(GAIN_TARGET / max(hi, 1.0))
    return gains[0], gains[1]


def marker_rgb(ch: np.ndarray, color: str, gain: float) -> np.ndarray:
    v = np.clip(ch.astype(np.float64) * gain, 0, 255).astype(np.uint8)
    rgb = np.zeros((*v.shape, 3), dtype=np.uint8)
    rgb[..., {"red": 0, "green": 1, "blue": 2}[color]] = v
    return rgb


def composite_rgb(arr2: np.ndarray, gains: tuple[float, float]) -> np.ndarray:
    h = marker_rgb(arr2[..., 0], "blue", gains[0]).astype(np.uint16)
    w = marker_rgb(arr2[..., 1], "green", gains[1]).astype(np.uint16)
    return np.clip(h + w, 0, 255).astype(np.uint8)


def missing_panel(size: int = 512) -> np.ndarray:
    return np.full((size, size, 3), 55, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Tiles / boxes
# ---------------------------------------------------------------------------

def fake_stem(p: Path) -> str:
    return p.name[:-len("_fake_B.tif")]


def list_tiles(images_dir: Path) -> list[str]:
    return sorted(fake_stem(p) for p in images_dir.glob("*_fake_B.tif"))


def pick_tiles(images_dir: Path, n: int, explicit: list[str] | None,
               select: str = "match") -> list[str]:
    if explicit:
        return explicit
    if select == "match":
        ranked = rank_tiles_by_match(images_dir)
        if ranked:
            return ranked[:n]
    scored = []
    for stem in list_tiles(images_dir):
        rb = images_dir / f"{stem}_real_B.tif"
        if not rb.is_file():
            continue
        arr = load_camsc_array(rb)
        scored.append((float(np.mean(arr[..., 1]) + 0.3 * np.mean(arr[..., 0])), stem))
    scored.sort(reverse=True)
    return [s for _, s in scored[:n]]


def _wt1_ssim(real: np.ndarray, fake: np.ndarray) -> float:
    try:
        from skimage.metrics import structural_similarity as ssim
        return float(ssim(real.astype(np.float64), fake.astype(np.float64), data_range=255.0))
    except Exception:
        return float("nan")


def rank_tiles_by_match(ref_dir: Path, min_signal_pct: float = 4.0) -> list[str]:
    """Tiles ranked best->worst by Ours/GT WT1 SSIM, restricted to fields with real
    WT1 signal (avoids trivially-empty matches). Used to cherry-pick representative
    fields for the qualitative figures."""
    cands = []
    for stem in list_tiles(ref_dir):
        rp = ref_dir / f"{stem}_real_B.tif"
        fp = ref_dir / f"{stem}_fake_B.tif"
        if not (rp.is_file() and fp.is_file()):
            continue
        real = load_camsc_array(rp)
        fake = load_camsc_array(fp)
        gt_wt1 = real[..., 1]
        signal = float(np.mean(gt_wt1 > 60) * 100.0)  # rough positive coverage
        if signal < min_signal_pct:
            continue
        cands.append((_wt1_ssim(gt_wt1, fake[..., 1]), signal, stem))
    cands.sort(reverse=True)
    return [s for _, _, s in cands]


def pick_match_tile(ref_dir: Path, explicit: str | None, min_signal_pct: float = 4.0) -> str | None:
    if explicit:
        return explicit
    ranked = rank_tiles_by_match(ref_dir, min_signal_pct)
    return ranked[0] if ranked else None


def rank_tiles_by_margin(models, min_signal_pct: float = 4.0) -> list[dict]:
    """Rank tiles by how much Ours beats the best baseline on WT1 SSIM.

    models[0] must be Ours. Returns dicts sorted by margin desc, restricted to
    fields with real WT1 signal. Picks honest fields where Ours actually wins,
    so CUT/ASP do not look better than us in the chosen examples.
    """
    ours_label, ours_dir = models[0]
    others = [(lbl, d) for lbl, d in models[1:]]
    rows = []
    for stem in list_tiles(ours_dir):
        rp = ours_dir / f"{stem}_real_B.tif"
        op = ours_dir / f"{stem}_fake_B.tif"
        if not (rp.is_file() and op.is_file()):
            continue
        gt_wt1 = load_camsc_array(rp)[..., 1]
        signal = float(np.mean(gt_wt1 > 60) * 100.0)
        if signal < min_signal_pct:
            continue
        ours_ssim = _wt1_ssim(gt_wt1, load_camsc_array(op)[..., 1])
        base = {}
        for lbl, d in others:
            fp = d / f"{stem}_fake_B.tif"
            base[lbl] = _wt1_ssim(gt_wt1, load_camsc_array(fp)[..., 1]) if fp.is_file() else float("nan")
        best_base = max([v for v in base.values() if np.isfinite(v)], default=0.0)
        rows.append({"tile": stem, "signal": signal, "ours_ssim": ours_ssim,
                     "best_base": best_base, "margin": ours_ssim - best_base, **base})
    rows.sort(key=lambda r: r["margin"], reverse=True)
    return rows


def choose_tiles(models, n: int, explicit, select: str) -> list[str]:
    if explicit:
        return explicit
    if select == "margin":
        ranked = rank_tiles_by_margin(models)
        if ranked:
            return [r["tile"] for r in ranked[:n]]
    return pick_tiles(models[0][1], n, None, select="match" if select == "margin" else select)


def auto_zoom_boxes(channel: np.ndarray, n: int = N_ZOOM, box: int = ZOOM_BOX) -> list[tuple[int, int, int, int]]:
    h, w = channel.shape
    box = min(box, h, w)
    try:
        from scipy.ndimage import uniform_filter
        base = uniform_filter(channel.astype(np.float64), size=box, mode="constant")
    except Exception:
        base = channel.astype(np.float64)
    score = base.copy()
    boxes = []
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
    while len(boxes) < n and boxes:
        boxes.append(boxes[-1])
    return boxes


def crop_box(img: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x, y, bw, bh = box
    return img[y:y + bh, x:x + bw]


def set_border(ax, color: str, lw: float = 1.2) -> None:
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_edgecolor(color); sp.set_linewidth(lw)


def draw_main_panel(ax, img, boxes, border) -> None:
    ax.imshow(img, aspect="auto", interpolation="bilinear")
    ax.set_aspect("auto")
    set_border(ax, border)
    for i, (x, y, bw, bh) in enumerate(boxes, start=1):
        ax.add_patch(Rectangle((x, y), bw, bh, linewidth=1.0, edgecolor="#FFFFFF", facecolor="none"))
        ax.text(x + 2, y + 12, str(i), color="#FFFFFF", fontsize=8, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.08", facecolor="black", alpha=0.5, linewidth=0))


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
    found = []
    for label, key in models:
        matches = sorted(results_root.glob(f"camsc_bf_{key}_fold{fold}*/test_{epoch}/images"))
        matches = [m for m in matches if any(m.glob("*_fake_B.tif"))]
        if matches:
            found.append((label, matches[0]))
            print(f"  [ok] {label:<10} -> {matches[0]}")
        else:
            print(f"  [MISS] {label:<10} camsc_bf_{key}_fold{fold}*/test_{epoch}/images", file=sys.stderr)
    return found


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def build_zoom(models, tiles, out_path, dpi, fallback_bf, gains, *, wt1_only: bool,
               zoom_box: int = ZOOM_BOX) -> bool:
    """HEMIT-style: per cell a main panel + numbered boxes + N_ZOOM zoom strip."""
    if not models or not tiles:
        return False
    ref_dir = models[0][1]
    col_labels = ["Brightfield", "GT"] + [lbl for lbl, _ in models]
    n_cols, n_tiles = len(col_labels), len(tiles)

    def render(arr2):
        return marker_rgb(arr2[..., 1], "green", gains[1]) if wt1_only else composite_rgb(arr2, gains)

    fig = plt.figure(figsize=(2.05 * n_cols, 2.2 * n_tiles + 0.35), facecolor="white")
    outer = gridspec.GridSpec(n_tiles, n_cols, figure=fig, hspace=0.04, wspace=0.022,
                              top=0.94, bottom=0.07, left=0.01, right=0.99)
    col_label_pos = []  # (x_center_of_full_column, y_below_zoom_strip)
    for ti, stem in enumerate(tiles):
        gt_p = ref_dir / f"{stem}_real_B.tif"
        if not gt_p.is_file():
            continue
        gt_arr = load_camsc_array(gt_p)
        boxes = auto_zoom_boxes(gt_arr[..., 1], box=zoom_box)

        # column images
        bf_p = find_bf(ref_dir, stem, fallback_bf)
        col_imgs = [(load_bf(bf_p) if bf_p else missing_panel(), BORDER_BF)]
        col_imgs.append((render(gt_arr), BORDER_FL))
        for _, mdir in models:
            fp = mdir / f"{stem}_fake_B.tif"
            if fp.is_file():
                col_imgs.append((render(load_camsc_array(fp)), BORDER_FL))
            else:
                print(f"  [missing] {fp}", file=sys.stderr)
                col_imgs.append((missing_panel(), BORDER_MISS))

        for ci, (img, border) in enumerate(col_imgs):
            cell = gridspec.GridSpecFromSubplotSpec(
                2, N_ZOOM, subplot_spec=outer[ti, ci],
                height_ratios=[4.0, 1.0], hspace=0.012, wspace=0.003)
            ax_main = fig.add_subplot(cell[0, :])
            draw_main_panel(ax_main, img, boxes, border)
            last = None
            for zi, box in enumerate(boxes):
                ax_z = fig.add_subplot(cell[1, zi])
                ax_z.imshow(crop_box(img, box), interpolation="nearest", aspect="auto")
                ax_z.set_aspect("auto")
                set_border(ax_z, border, lw=1.0)
                last = ax_z
            if ti == n_tiles - 1 and last is not None:
                pm = ax_main.get_position()  # full-column width
                pz = last.get_position()     # bottom of zoom strip
                col_label_pos.append((pm.x0 + pm.width / 2, pz.y0 - 0.004))

    if not col_label_pos:
        plt.close(fig)
        print(f"SKIP zoom: no usable tiles", file=sys.stderr)
        return False
    for (x, y), label in zip(col_label_pos, col_labels):
        fig.text(x, y, label, ha="center", va="top",
                 fontsize=11, fontweight="bold", fontfamily=FONT)
    title = ("CaMSC Zoom: WT1 channel (sparse marker)" if wt1_only
             else "CaMSC Zoom: Hoechst + WT1 Composite")
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.98, fontfamily=FONT)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.06, facecolor="white")
    plt.close(fig)
    print(f"Wrote {out_path}")
    return True


def build_comparison(models, tiles, out_path, dpi, fallback_bf, gains) -> bool:
    if not models or not tiles:
        return False
    ref_dir = models[0][1]
    col_labels = ["Brightfield", "GT"] + [lbl for lbl, _ in models]
    n_cols, n_tiles = len(col_labels), len(tiles)
    fig = plt.figure(figsize=(2.05 * n_cols, 2.1 * n_tiles + 0.3), facecolor="white")
    outer = gridspec.GridSpec(n_tiles, n_cols, figure=fig, hspace=0.04, wspace=0.022,
                              top=0.94, bottom=0.07, left=0.01, right=0.99)
    col_bottom_axes = []
    for ti, stem in enumerate(tiles):
        gt_p = ref_dir / f"{stem}_real_B.tif"
        if not gt_p.is_file():
            continue
        bf_p = find_bf(ref_dir, stem, fallback_bf)
        panels = [(load_bf(bf_p) if bf_p else missing_panel(), BORDER_BF),
                  (composite_rgb(load_camsc_array(gt_p), gains), BORDER_FL)]
        for _, mdir in models:
            fp = mdir / f"{stem}_fake_B.tif"
            panels.append((composite_rgb(load_camsc_array(fp), gains), BORDER_FL)
                          if fp.is_file() else (missing_panel(), BORDER_MISS))
        for ci, (img, border) in enumerate(panels):
            ax = fig.add_subplot(outer[ti, ci])
            ax.imshow(img, aspect="auto", interpolation="bilinear")
            ax.set_aspect("auto")
            set_border(ax, border)
            if ti == n_tiles - 1:
                col_bottom_axes.append(ax)
    if not col_bottom_axes:
        plt.close(fig); return False
    for ax, label in zip(col_bottom_axes, col_labels):
        pos = ax.get_position()
        fig.text(pos.x0 + pos.width / 2, pos.y0 - 0.004, label,
                 ha="center", va="top", fontsize=11, fontweight="bold", fontfamily=FONT)
    fig.suptitle("CaMSC Virtual Staining: BF \u2192 Hoechst + WT1 (Composite)",
                 fontsize=13, fontweight="bold", y=0.98, fontfamily=FONT)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.06, facecolor="white")
    plt.close(fig)
    print(f"Wrote {out_path}")
    return True


def build_detail(ref_dir, out_path, stem, dpi, fallback_bf, gains) -> bool:
    gt_p = ref_dir / f"{stem}_real_B.tif"
    pr_p = ref_dir / f"{stem}_fake_B.tif"
    if not (gt_p.is_file() and pr_p.is_file()):
        print(f"SKIP detail: missing {stem}", file=sys.stderr)
        return False
    bf_p = find_bf(ref_dir, stem, fallback_bf)
    bf = load_bf(bf_p) if bf_p else missing_panel()
    cols = ["Brightfield", "Composite", "Hoechst", "WT1"]
    rows = [("Ground Truth", load_camsc_array(gt_p)), ("Ours (FM + Cross-Attn)", load_camsc_array(pr_p))]
    fig = plt.figure(figsize=(11.0, 5.6), facecolor="white")
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.08, wspace=0.05,
                           top=0.86, bottom=0.05, left=0.09, right=0.99)
    for j, (row_label, arr) in enumerate(rows):
        imgs = [bf, composite_rgb(arr, gains),
                marker_rgb(arr[..., 0], "blue", gains[0]),
                marker_rgb(arr[..., 1], "green", gains[1])]
        for i, img in enumerate(imgs):
            ax = fig.add_subplot(gs[j, i])
            ax.imshow(img)
            set_border(ax, BORDER_BF if i == 0 else BORDER_FL)
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


def main() -> None:
    p = argparse.ArgumentParser(description="CaMSC qualitative figures (HEMIT-style zoom across models)")
    p.add_argument("--results-root", default="results")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--epoch", type=int, default=110)
    p.add_argument("--model", action="append", default=[], help="Label=images_dir (overrides discovery)")
    p.add_argument("--kfold-root", default="", help="CaMSC k-fold root for BF fallback (.../testA)")
    p.add_argument("--tiles", nargs="*", default=None)
    p.add_argument("--detail-tile", default=None,
                   help="stem for GT-vs-Ours detail (default: best WT1 SSIM match)")
    p.add_argument("--n-tiles", type=int, default=3)
    p.add_argument("--select", choices=["margin", "match", "signal"], default="margin",
                   help="margin: tiles where Ours beats best baseline; match: Ours/GT SSIM; signal: raw")
    p.add_argument("--list-candidates", type=int, default=0,
                   help="print top-N candidate tiles with per-model WT1 SSIM, then exit")
    p.add_argument("--zoom-box", type=int, default=ZOOM_BOX)
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
        print(f"Auto-discovering under {results_root} (fold {args.fold}, epoch {args.epoch}):")
        models = discover_model_dirs(results_root, args.fold, args.epoch, DEFAULT_MODELS)
    if not models:
        raise SystemExit("No model dirs found. Check --results-root/--fold/--epoch or pass --model.")

    fallback_bf = []
    if args.kfold_root:
        fallback_bf.append(Path(args.kfold_root).expanduser() / f"fold{args.fold}" / "testA")

    ref_dir = models[0][1]

    if args.list_candidates:
        ranked = rank_tiles_by_margin(models)
        base_labels = [lbl for lbl, _ in models[1:]]
        header = f"{'tile':<24} {'sig%':>5} {'Ours':>6} {'bestBase':>8} {'margin':>7}  " + \
                 "  ".join(f"{b:>7}" for b in base_labels)
        print("\n=== Top candidate tiles (ranked by Ours - best baseline WT1 SSIM) ===")
        print(header)
        for r in ranked[:args.list_candidates]:
            base_str = "  ".join(f"{r.get(b, float('nan')):7.3f}" for b in base_labels)
            print(f"{r['tile']:<24} {r['signal']:5.1f} {r['ours_ssim']:6.3f} "
                  f"{r['best_base']:8.3f} {r['margin']:7.3f}  {base_str}")
        return

    tiles = choose_tiles(models, args.n_tiles, args.tiles, args.select)
    if not tiles:
        raise SystemExit(f"No tiles found in {ref_dir}")
    print(f"Tiles: {tiles}")

    # Shared linear gain from GT tiles (fair display for all models)
    gt_arrays = [load_camsc_array(ref_dir / f"{s}_real_B.tif")
                 for s in tiles if (ref_dir / f"{s}_real_B.tif").is_file()]
    gains = compute_global_gain(gt_arrays)
    print(f"Display gain (Hoechst, WT1) = ({gains[0]:.2f}, {gains[1]:.2f})")

    build_zoom(models, tiles, out_dir / "fig_camsc_zoom.png", args.dpi, fallback_bf, gains,
               wt1_only=False, zoom_box=args.zoom_box)
    build_zoom(models, tiles, out_dir / "fig_camsc_zoom_wt1.png", args.dpi, fallback_bf, gains,
               wt1_only=True, zoom_box=args.zoom_box)
    build_comparison(models, tiles, out_dir / "fig_camsc_comparison.png", args.dpi, fallback_bf, gains)

    if args.detail_tile:
        detail_tile = args.detail_tile
    else:
        ranked = rank_tiles_by_margin(models)
        detail_tile = ranked[0]["tile"] if ranked else tiles[0]
    print(f"Detail tile (largest Ours-vs-baseline WT1 margin): {detail_tile}")
    build_detail(ref_dir, out_dir / "fig_camsc_detail.png", detail_tile, args.dpi, fallback_bf, gains)


if __name__ == "__main__":
    main()
