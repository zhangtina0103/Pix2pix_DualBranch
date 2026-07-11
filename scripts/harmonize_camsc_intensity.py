#!/usr/bin/env python3
"""
Percentile intensity harmonization for CaMSC flat TIF pool.

Learns per-marker p-low / p-high from a reference batch (default: field index 1–10,
original acquisition), then linearly rescales ALL fields to [0, 255].

Use when combining old + new Dropbox captures with different exposure/gain.

  python scripts/harmonize_camsc_intensity.py \\
    --src ~/orcd/scratch/camsc/camsc_all \\
    --dst ~/orcd/scratch/camsc/camsc_all_harm \\
    --ref-max-index 10

Then re-prep k-fold from --dst and re-finetune.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

_MARKER_RE = re.compile(
    r"^CaMSC\s+(\d+)%\s+(\d+x)\s+(BF|Hoechst|WT1)_(\d+)\.tif$",
    re.IGNORECASE,
)
_MARKERS = ("BF", "Hoechst", "WT1")


def _load_gray(path: Path, step: int = 1) -> np.ndarray:
    arr = tifffile.imread(path)
    if arr.ndim == 3:
        if arr.shape[-1] in (3, 4):
            arr = np.max(arr[..., :3], axis=-1)
        elif arr.shape[0] in (3, 4):
            arr = np.max(arr[:3], axis=0)
        else:
            arr = arr[..., 0]
    arr = np.asarray(arr, dtype=np.float32)
    if arr.max() <= 1.0:
        arr *= 255.0
    if step > 1:
        arr = arr[::step, ::step]
    return arr


def _values_for_stats(arr: np.ndarray, marker: str, wt1_nonzero: bool) -> np.ndarray:
    flat = arr.ravel()
    if wt1_nonzero and marker.upper() == "WT1":
        flat = flat[flat > 0]
    return flat if flat.size else arr.ravel()


def _parse_marker_path(path: Path) -> tuple[str, int] | None:
    m = _MARKER_RE.match(path.name)
    if not m:
        return None
    return m.group(3).capitalize(), int(m.group(4))


def _normalize_marker(name: str) -> str:
    n = name.capitalize()
    if n.lower() == "bf":
        return "BF"
    if n.lower() == "hoechst":
        return "Hoechst"
    if n.lower() == "wt1":
        return "WT1"
    raise ValueError(name)


def collect_paths(src: Path) -> list[Path]:
    paths = [p for p in sorted(src.glob("*.tif")) if _parse_marker_path(p)]
    if not paths:
        raise SystemExit(f"No CaMSC marker TIFs under {src}")
    return paths


def fit_reference(
    paths: list[Path],
    ref_max_index: int,
    p_low: float,
    p_high: float,
    wt1_nonzero: bool,
    sample_step: int,
) -> dict[str, dict[str, float]]:
    pools: dict[str, list[np.ndarray]] = {m: [] for m in _MARKERS}
    for p in paths:
        parsed = _parse_marker_path(p)
        assert parsed is not None
        marker, idx = parsed
        marker = _normalize_marker(marker)
        if idx > ref_max_index:
            continue
        arr = _load_gray(p, step=sample_step)
        vals = _values_for_stats(arr, marker, wt1_nonzero)
        pools[marker].append(vals)

    ref: dict[str, dict[str, float]] = {}
    for marker in _MARKERS:
        chunks = pools[marker]
        if not chunks:
            raise SystemExit(f"No reference images for {marker} with index <= {ref_max_index}")
        vals = np.concatenate(chunks)
        lo = float(np.percentile(vals, p_low))
        hi = float(np.percentile(vals, p_high))
        if hi <= lo + 1e-6:
            hi = lo + 1.0
        ref[marker] = {"p_low": lo, "p_high": hi, "n_ref_files": len(chunks)}
        print(f"  {marker}: p{p_low:g}={lo:.2f} p{p_high:g}={hi:.2f}  ({len(chunks)} ref files)")
    return ref


def apply_stretch(arr: np.float32, lo: float, hi: float) -> np.ndarray:
    out = (arr - lo) / (hi - lo) * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def harmonize(
    src: Path,
    dst: Path,
    ref: dict[str, dict[str, float]],
    dry_run: bool,
) -> int:
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in collect_paths(src):
        marker, _idx = _parse_marker_path(p)
        assert marker is not None
        marker = _normalize_marker(marker)
        lo = ref[marker]["p_low"]
        hi = ref[marker]["p_high"]

        arr = _load_gray(p, step=1)
        out = apply_stretch(arr, lo, hi)
        out_path = dst / p.name
        if dry_run:
            print(f"[dry-run] {p.name} -> mean {arr.mean():.1f} => {out.mean():.1f}")
        else:
            tifffile.imwrite(out_path, out)
        n += 1
        if n % 25 == 0:
            print(f"  wrote {n}...", flush=True)
    return n


def preview_indices(src: Path, dst: Path, ref: dict, indices: list[int]) -> None:
    print("\nPreview (one file per marker / index):")
    for idx in indices:
        batch = "old" if idx <= 10 else "new"
        print(f"  index {idx} ({batch})")
        for marker in _MARKERS:
            matches = list(src.glob(f"*{marker}_{idx}.tif"))
            if not matches:
                continue
            p = matches[0]
            m = _normalize_marker(marker)
            arr = _load_gray(p)
            out = apply_stretch(arr, ref[m]["p_low"], ref[m]["p_high"])
            print(
                f"    {marker}: mean {arr.mean():.1f} -> {out.mean():.1f}  "
                f"p99 {np.percentile(arr, 99):.1f} -> {np.percentile(out, 99):.1f}"
            )


def main() -> None:
    p = argparse.ArgumentParser(description="Harmonize CaMSC flat TIF intensities")
    p.add_argument("--src", type=str, required=True)
    p.add_argument("--dst", type=str, required=True)
    p.add_argument("--ref-max-index", type=int, default=10,
                   help="Use fields with index <= N as reference (default: old batch 1–10)")
    p.add_argument("--p-low", type=float, default=1.0)
    p.add_argument("--p-high", type=float, default=99.0)
    p.add_argument("--wt1-nonzero", action="store_true", default=True,
                   help="WT1 reference percentiles on pixels > 0 only")
    p.add_argument("--sample-step", type=int, default=4,
                   help="Downsample step when fitting reference (speed)")
    p.add_argument("--preview-indices", type=str, default="8,13",
                   help="Comma indices to print before/after means")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    src = Path(args.src).expanduser().resolve()
    dst = Path(args.dst).expanduser().resolve()
    if not src.is_dir():
        raise SystemExit(f"Missing src: {src}")

    paths = collect_paths(src)
    print(f"Found {len(paths)} marker TIFs in {src}")
    print(f"Fitting reference (index <= {args.ref_max_index})...")
    ref = fit_reference(
        paths, args.ref_max_index, args.p_low, args.p_high,
        args.wt1_nonzero, args.sample_step,
    )

    preview = [int(x) for x in args.preview_indices.split(",") if x.strip()]
    preview_indices(src, dst, ref, preview)

    print(f"\nWriting {'[dry-run] ' if args.dry_run else ''}to {dst} ...")
    n = harmonize(src, dst, ref, args.dry_run)
    meta = {
        "src": str(src),
        "dst": str(dst),
        "ref_max_index": args.ref_max_index,
        "p_low": args.p_low,
        "p_high": args.p_high,
        "wt1_nonzero": args.wt1_nonzero,
        "reference": ref,
        "n_written": n,
    }
    if not args.dry_run:
        (dst / "harmonize_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Done. {n} files.")
    if not args.dry_run:
        print(f"Meta: {dst / 'harmonize_meta.json'}")
        print("\nNext:")
        print(f"  export CAMSC_SRC={dst}")
        print("  rm -rf ~/orcd/scratch/camsc/datasets/camsc_bf_kfold_125_harm")
        print("  export CAMSC_KFOLD_ROOT=~/orcd/scratch/camsc/datasets/camsc_bf_kfold_125_harm")
        print("  python scripts/prepare_camsc_bf.py --src $CAMSC_SRC --auto-discover \\")
        print("    --k-folds 5 --dst $CAMSC_KFOLD_ROOT --tile-size 0 --seed 42")


if __name__ == "__main__":
    main()
