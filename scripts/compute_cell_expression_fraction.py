#!/usr/bin/env python3
"""
Cell-level marker sparsity: fraction of nuclei that express each marker.

Pixel-area PPF (compute_marker_sparsity.py) is the right sparsity measure for
HEMIT (CD3 cleanly sparse), but it is misleading for cell culture: a compact
nuclear stain (Hoechst) covers little area while a broader cytoplasmic marker
(WT1) covers more, so WT1 looks "denser" by area even though only a subset of
cells express it. The biologically meaningful definition is:

    expression fraction = (# marker-positive nuclei) / (# nuclei)

Every nucleus is nuclear-stain-positive by construction (~100%); a sparse marker
is expressed by only a subset of nuclei. Nucleus segmentation reuses the same
Otsu pipeline as scripts/hemit_eval/downstream_biology.py.

A nucleus is marker-positive if its mean marker intensity exceeds a single
per-marker threshold computed on pooled per-nucleus means (global Otsu). This
gives one fixed, reproducible threshold per marker.

Examples
--------
CaMSC (nuclear=Hoechst ch0, marker=WT1 ch1), pooled over folds:
  python scripts/compute_cell_expression_fraction.py \
    --gt-glob "~/orcd/scratch/camsc/datasets/camsc_bf_kfold_aug/fold*/testB/*.tif" \
    --channels hoechst,wt1 \
    --out-dir results/cell_expression_camsc

HEMIT (nuclear=DAPI ch0, markers=CD3 ch1, panCK ch2):
  python scripts/compute_cell_expression_fraction.py \
    --gt-glob "results/hemit_fm_cross_attn_scratch_512/test_80/images/*_real_B.tif" \
    --channels dapi,cd3,panck \
    --out-dir results/cell_expression_hemit
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np
from skimage.filters import threshold_otsu
from skimage.io import imread
from skimage.measure import label, regionprops

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.hemit_eval.downstream_biology import segment_nuclei  # noqa: E402


def _to_float255(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img, dtype=np.float32)
    if arr.max() <= 1.0:
        arr = ((arr + 1.0) / 2.0) * 255.0
    return np.clip(arr, 0.0, 255.0)


def _safe_otsu(values: np.ndarray) -> float:
    v = values[np.isfinite(values)]
    if v.size == 0 or float(v.max()) <= float(v.min()):
        return float("nan")
    try:
        return float(threshold_otsu(v))
    except Exception:
        return float("nan")


def main() -> None:
    p = argparse.ArgumentParser(description="Cell-level marker expression fraction")
    p.add_argument("--gt-glob", required=True)
    p.add_argument("--channels", required=True,
                   help="Channel names in stack order; channel 0 is the nuclear stain")
    p.add_argument("--out-dir", default="results/cell_expression")
    p.add_argument("--min-area", type=int, default=36, help="Min nucleus area (px)")
    args = p.parse_args()

    channels = [c.strip() for c in args.channels.split(",") if c.strip()]
    if len(channels) < 2:
        raise SystemExit("Need >=2 channels (nuclear + >=1 marker)")
    nuclear, markers = channels[0], channels[1:]
    marker_idx = {m: i + 1 for i, m in enumerate(markers)}

    paths = sorted(Path(x) for x in glob.glob(str(Path(args.gt_glob).expanduser())))
    if not paths:
        raise SystemExit(f"No files matched --gt-glob {args.gt_glob}")

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {len(paths)} GT tiles; nuclear={nuclear}, markers={markers}")

    # Pass 1: segment nuclei per tile, collect per-nucleus mean intensities.
    tile_nuclei = []  # list of (path, n_nuclei, {marker: np.array of per-nucleus means})
    pooled_means = {m: [] for m in markers}
    for path in paths:
        arr = _to_float255(imread(path))
        if arr.ndim == 2 or arr.shape[-1] < len(channels):
            raise ValueError(f"{path.name}: need {len(channels)} channels")
        nuclei = segment_nuclei(arr[..., 0], min_area=args.min_area)
        labeled = label(nuclei)
        n_nuclei = int(labeled.max())
        per_marker = {}
        for m in markers:
            if n_nuclei == 0:
                per_marker[m] = np.array([], dtype=np.float64)
                continue
            props = regionprops(labeled, intensity_image=arr[..., marker_idx[m]])
            means = np.array([pr.mean_intensity for pr in props], dtype=np.float64)
            per_marker[m] = means
            pooled_means[m].append(means)
        tile_nuclei.append((path, n_nuclei, per_marker))

    # Global per-marker threshold on pooled per-nucleus means (single threshold/marker).
    global_thr = {}
    for m in markers:
        pooled = np.concatenate(pooled_means[m]) if pooled_means[m] else np.array([])
        global_thr[m] = _safe_otsu(pooled)

    # Pass 2: per-tile expression fraction + pooled nucleus-level fraction.
    per_tile_rows = []
    pooled_pos = {m: 0 for m in markers}
    pooled_total = 0
    for path, n_nuclei, per_marker in tile_nuclei:
        row = {"file_name": path.stem, "n_nuclei": n_nuclei}
        pooled_total += n_nuclei
        for m in markers:
            means = per_marker[m]
            thr = global_thr[m]
            if n_nuclei == 0 or not np.isfinite(thr):
                row[f"{m}_expr_fraction"] = float("nan")
                row[f"{m}_n_positive"] = 0
                continue
            n_pos = int(np.sum(means >= thr))
            row[f"{m}_n_positive"] = n_pos
            row[f"{m}_expr_fraction"] = n_pos / n_nuclei
            pooled_pos[m] += n_pos
        per_tile_rows.append(row)

    summary = {
        "n_tiles": len(paths),
        "nuclear_channel": nuclear,
        "markers": markers,
        "global_otsu_threshold_per_nucleus": global_thr,
        "total_nuclei": pooled_total,
        "per_marker": {},
    }
    for m in markers:
        fr = np.array([r[f"{m}_expr_fraction"] for r in per_tile_rows], dtype=np.float64)
        fr = fr[np.isfinite(fr)]
        summary["per_marker"][m] = {
            "tile_mean_expr_fraction": float(np.mean(fr)) if fr.size else float("nan"),
            "tile_std_expr_fraction": float(np.std(fr, ddof=1)) if fr.size > 1 else 0.0,
            "tile_median_expr_fraction": float(np.median(fr)) if fr.size else float("nan"),
            "pooled_expr_fraction": (pooled_pos[m] / pooled_total) if pooled_total else float("nan"),
            "n_positive_total": pooled_pos[m],
        }

    per_tile_csv = out_dir / "cell_expression_per_tile.csv"
    with per_tile_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_tile_rows[0].keys()))
        w.writeheader()
        w.writerows(per_tile_rows)
    (out_dir / "cell_expression_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    print(f"\n=== Cell-level expression fraction over {len(paths)} tiles ===")
    print(f"Nuclear reference: {nuclear} = 100% by construction "
          f"({pooled_total} nuclei total)")
    print(f"{'marker':<10} {'tile mean expr':>18} {'pooled expr':>14} {'nucleusThr':>11}")
    for m in markers:
        s = summary["per_marker"][m]
        print(
            f"{m:<10} {s['tile_mean_expr_fraction']*100:>8.2f}% ± "
            f"{s['tile_std_expr_fraction']*100:>5.2f}%   "
            f"{s['pooled_expr_fraction']*100:>10.2f}%   {global_thr[m]:>11.2f}"
        )
    ranked = sorted(markers, key=lambda m: summary["per_marker"][m]["pooled_expr_fraction"])
    order = " < ".join(ranked + [f"{nuclear}(100%)"])
    print(f"\nSparsest → densest (by pooled expr fraction): {order}")
    print(f"\nWrote {per_tile_csv}")
    print(f"Wrote {out_dir / 'cell_expression_summary.json'}")


if __name__ == "__main__":
    main()
