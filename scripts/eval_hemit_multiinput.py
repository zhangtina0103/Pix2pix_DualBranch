#!/usr/bin/env python3
"""Extended metrics for HEMIT multi-input runs (label order: CD3, panCK, pad)."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.hemit_eval.extended_metrics import (  # noqa: E402
    METRIC_SPECS,
    _LpipsScorer,
    _channel_metrics,
)

CHANNELS = ("cd3", "panck")
METRICS = tuple(METRIC_SPECS.keys())


def resolve_image_dir(srcdir: Path) -> Path:
    """test.py saves TIFFs under <test_epoch>/images/."""
    srcdir = srcdir.resolve()
    images = srcdir / "images"
    if images.is_dir():
        for ext in (".tif", ".tiff", ".png"):
            if any(images.glob(f"*_fake_B{ext}")):
                return images
    return srcdir


def _load_hwc(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path))
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[0] < min(arr.shape[1:3]):
        arr = np.moveaxis(arr[:3], 0, -1)
    if arr.ndim == 3 and arr.shape[-1] >= 3:
        return arr[..., :3].astype(np.float32)
    raise ValueError(f"Bad image shape {arr.shape} in {path}")


def _find_pairs(img_dir: Path) -> list[tuple[Path, Path, str]]:
    pairs = []
    for ext in (".tif", ".tiff", ".png"):
        for fake_path in sorted(img_dir.glob(f"*_fake_B{ext}")):
            stem = fake_path.name.replace(f"_fake_B{ext}", "")
            real_path = img_dir / f"{stem}_real_B{ext}"
            if real_path.is_file():
                pairs.append((real_path, fake_path, stem))
    return pairs


def _mean_std(vals: list[float]) -> tuple[float, float]:
    a = np.array(vals, dtype=np.float64)
    return float(np.nanmean(a)), float(np.nanstd(a))


def compute_rows(
    pairs: list[tuple[Path, Path, str]],
    *,
    use_lpips: bool,
) -> list[dict[str, float | str]]:
    lpips_scorer = _LpipsScorer() if use_lpips else None
    if use_lpips and lpips_scorer and not lpips_scorer.available:
        print("WARNING: LPIPS unavailable; lpips columns will be NaN", flush=True)

    rows: list[dict[str, float | str]] = []
    for real_path, fake_path, stem in tqdm(pairs, desc="score_cd3_panck"):
        real = _load_hwc(real_path)
        fake = _load_hwc(fake_path)
        row: dict[str, float | str] = {"file_name": stem}
        channel_metrics: dict[str, dict[str, float]] = {}
        for i, ch in enumerate(CHANNELS):
            r = real[..., i].astype(np.float64)
            f = fake[..., i].astype(np.float64)
            m = _channel_metrics(r, f)
            if lpips_scorer and lpips_scorer.available:
                m["lpips"] = lpips_scorer.score(r, f)
            else:
                m["lpips"] = float("nan")
            channel_metrics[ch] = m
            for metric, val in m.items():
                row[f"{ch}_{metric}"] = val
        for metric in METRICS:
            row[f"average_{metric}"] = float(
                np.nanmean([channel_metrics[ch][metric] for ch in CHANNELS])
            )
        rows.append(row)
    return rows


def _fieldnames() -> list[str]:
    cols = ["file_name"]
    for ch in CHANNELS:
        cols.extend(f"{ch}_{m}" for m in METRICS)
    cols.extend(f"average_{m}" for m in METRICS)
    return cols


def _write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = _fieldnames()
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _summary_dict(rows: list[dict[str, float | str]]) -> dict:
    summary: dict = {"channels": {}, "average": {}, "n_tiles": len(rows)}
    for ch in CHANNELS:
        summary["channels"][ch] = {}
        for m in METRICS:
            vals = [float(r[f"{ch}_{m}"]) for r in rows]
            mean, std = _mean_std(vals)
            summary["channels"][ch][m] = {"mean": mean, "std": std}
    for m in METRICS:
        vals = [float(r[f"average_{m}"]) for r in rows]
        mean, std = _mean_std(vals)
        summary["average"][m] = {"mean": mean, "std": std}
    return summary


def _print_summary(summary: dict) -> None:
    n = summary["n_tiles"]
    print(f"  n_tiles={n}")
    for ch in CHANNELS:
        print(f"  [{ch}]")
        for m in METRICS:
            s = summary["channels"][ch][m]
            print(f"    {m:8s}  {s['mean']:.4f} ± {s['std']:.4f}")
    print("  [average cd3+panck]")
    for m in METRICS:
        s = summary["average"][m]
        print(f"    {m:8s}  {s['mean']:.4f} ± {s['std']:.4f}")


def main() -> None:
    p = argparse.ArgumentParser(description="Eval HEMIT multi-input CD3/panCK predictions")
    p.add_argument("--srcdir", type=str, required=True, help="results/.../test_NN/")
    p.add_argument("--out-csv", type=str, default="", help="Per-tile CSV (default: srcdir/score_cd3_panck.csv)")
    p.add_argument("--no-lpips", action="store_true", help="Skip LPIPS (faster on CPU)")
    args = p.parse_args()

    srcdir = Path(args.srcdir).expanduser()
    img_dir = resolve_image_dir(srcdir)
    out_csv = Path(args.out_csv).expanduser() if args.out_csv else srcdir / "score_cd3_panck.csv"
    out_extended = srcdir / "extended_metrics_per_tile.csv"
    out_summary_json = srcdir / "extended_metrics_summary.json"

    pairs = _find_pairs(img_dir)
    if not pairs:
        raise SystemExit(f"No *_fake_B.{{tif,png}} pairs under {img_dir}")

    print(f"Metrics on: {img_dir} ({len(pairs)} tiles)")

    rows = compute_rows(pairs, use_lpips=not args.no_lpips)
    _write_csv(out_csv, rows)
    _write_csv(out_extended, rows)
    summary = _summary_dict(rows)
    out_summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {out_csv}")
    print(f"Wrote {out_extended}")
    print(f"Wrote {out_summary_json}")
    _print_summary(summary)


if __name__ == "__main__":
    main()
