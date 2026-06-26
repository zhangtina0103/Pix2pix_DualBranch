#!/usr/bin/env python3
"""
Quantify marker sparsity as Positive Pixel Fraction (PPF) on ground-truth tiles.

PPF = fraction of pixels in a marker channel above a positivity threshold,
averaged over tiles. This turns the qualitative "sparse vs dense" label into a
reproducible, data-derived property, using the same Otsu convention as the rest
of the HEMIT eval pipeline (skimage.threshold_otsu on [0,255] images).

Two thresholds are reported per marker:
  - ppf_tile_otsu   : per-tile Otsu threshold (adaptive; matches downstream_biology.py)
  - ppf_global_otsu : single Otsu threshold per marker, computed on pooled GT
                      intensities, then applied to every tile. Preferred for a
                      stable sparse/dense definition (one threshold per marker).

Examples
--------
HEMIT (945 GT test tiles, after running MODE=test for any model):
  python scripts/compute_marker_sparsity.py \
    --gt-glob "results/hemit_fm_cross_attn_scratch_512/test_80/images/*_real_B.tif" \
    --channels dapi,cd3,panck \
    --out-dir results/marker_sparsity_hemit

CaMSC (50 GT fields pooled across folds):
  python scripts/compute_marker_sparsity.py \
    --gt-glob "~/orcd/scratch/camsc/datasets/camsc_bf_kfold_aug/fold*/testB/*.tif" \
    --channels hoechst,wt1 \
    --out-dir results/marker_sparsity_camsc
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path

import numpy as np
from skimage.filters import threshold_otsu
from skimage.io import imread


def _to_float255(img: np.ndarray) -> np.ndarray:
    """Match scripts/hemit_eval/image_io._to_float255 normalization."""
    arr = np.asarray(img, dtype=np.float32)
    if arr.max() <= 1.0:
        arr = ((arr + 1.0) / 2.0) * 255.0
    return np.clip(arr, 0.0, 255.0)


def _safe_otsu(values: np.ndarray) -> float:
    """Otsu threshold; returns nan for flat/empty channels."""
    v = values[np.isfinite(values)]
    if v.size == 0 or float(v.max()) <= float(v.min()):
        return float("nan")
    try:
        return float(threshold_otsu(v))
    except Exception:
        return float("nan")


def _ppf(channel: np.ndarray, thr: float) -> float:
    if not np.isfinite(thr):
        return 0.0
    return float(np.mean(channel >= thr))


def load_gt_stacks(paths: list[Path], n_channels: int) -> list[np.ndarray]:
    stacks = []
    for p in paths:
        arr = _to_float255(imread(p))
        if arr.ndim == 2:
            arr = arr[..., None]
        if arr.shape[-1] < n_channels:
            raise ValueError(
                f"{p.name}: has {arr.shape[-1]} channels, need {n_channels}"
            )
        stacks.append(arr[..., :n_channels])
    return stacks


def main() -> None:
    p = argparse.ArgumentParser(description="Marker sparsity via Positive Pixel Fraction (PPF)")
    p.add_argument("--gt-glob", required=True,
                   help="Glob for ground-truth tiles (e.g. '.../*_real_B.tif' or '.../testB/*.tif')")
    p.add_argument("--channels", required=True,
                   help="Comma-separated channel names in stack order (e.g. dapi,cd3,panck)")
    p.add_argument("--out-dir", default="results/marker_sparsity")
    p.add_argument("--global-otsu-percentile", type=float, default=100.0,
                   help="Use this percentile of pooled per-tile maxima cap before global Otsu (100=off)")
    args = p.parse_args()

    channels = [c.strip() for c in args.channels.split(",") if c.strip()]
    n_channels = len(channels)
    if n_channels == 0:
        raise SystemExit("No channels parsed from --channels")

    paths = sorted(Path(x) for x in glob.glob(str(Path(args.gt_glob).expanduser())))
    if not paths:
        raise SystemExit(f"No files matched --gt-glob {args.gt_glob}")

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {len(paths)} GT tiles, channels={channels}")
    stacks = load_gt_stacks(paths, n_channels)

    # Global per-marker Otsu threshold from pooled intensities (subsample for memory).
    global_thr: dict[str, float] = {}
    rng = np.random.default_rng(42)
    for ci, ch in enumerate(channels):
        pooled = []
        for s in stacks:
            flat = s[..., ci].ravel()
            if flat.size > 20000:
                flat = flat[rng.integers(0, flat.size, size=20000)]
            pooled.append(flat)
        pooled_arr = np.concatenate(pooled)
        global_thr[ch] = _safe_otsu(pooled_arr)

    # Per-tile PPF with both thresholding schemes.
    per_tile_rows = []
    for path, s in zip(paths, stacks):
        row = {"file_name": path.stem}
        for ci, ch in enumerate(channels):
            chan = s[..., ci]
            t_otsu = _safe_otsu(chan.ravel())
            row[f"{ch}_ppf_tile_otsu"] = _ppf(chan, t_otsu)
            row[f"{ch}_ppf_global_otsu"] = _ppf(chan, global_thr[ch])
            row[f"{ch}_mean_intensity"] = float(np.mean(chan))
        per_tile_rows.append(row)

    # Aggregate.
    summary: dict[str, object] = {
        "n_tiles": len(stacks),
        "channels": channels,
        "global_otsu_threshold": global_thr,
        "markers": {},
    }
    for ch in channels:
        block = {}
        for scheme in ("ppf_tile_otsu", "ppf_global_otsu"):
            vals = np.array([r[f"{ch}_{scheme}"] for r in per_tile_rows], dtype=np.float64)
            block[scheme] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0,
                "median": float(np.median(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            }
        summary["markers"][ch] = block

    # Write outputs.
    per_tile_csv = out_dir / "marker_sparsity_per_tile.csv"
    with per_tile_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_tile_rows[0].keys()))
        w.writeheader()
        w.writerows(per_tile_rows)
    (out_dir / "marker_sparsity_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    # Console report.
    print(f"\n=== Positive Pixel Fraction (PPF) over {len(stacks)} tiles ===")
    print(f"{'marker':<10} {'PPF(global Otsu)':>20} {'PPF(tile Otsu)':>18} {'globalThr':>10}")
    for ch in channels:
        g = summary["markers"][ch]["ppf_global_otsu"]
        t = summary["markers"][ch]["ppf_tile_otsu"]
        print(
            f"{ch:<10} {g['mean']*100:>8.2f}% ± {g['std']*100:>5.2f}%   "
            f"{t['mean']*100:>7.2f}% ± {t['std']*100:>5.2f}%   "
            f"{global_thr[ch]:>10.2f}"
        )
    ranked = sorted(channels, key=lambda c: summary["markers"][c]["ppf_global_otsu"]["mean"])
    print(f"\nSparsest → densest (by global-Otsu PPF): {' < '.join(ranked)}")
    print(f"\nWrote {per_tile_csv}")
    print(f"Wrote {out_dir / 'marker_sparsity_summary.json'}")


if __name__ == "__main__":
    main()
