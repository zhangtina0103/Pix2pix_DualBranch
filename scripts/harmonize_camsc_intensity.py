#!/usr/bin/env python3
"""
Percentile intensity harmonization for CaMSC flat TIF pool.

Modes
-----
split (default):
  - index <= ref_max_index (old batch): copy unchanged
  - index > ref_max_index (new batch): stretch using NEW-batch percentiles only
  Old GT untouched; helps but dim/bright alternating new fields may stay mismatched.

split-per-image (recommended when new batch alternates dim/bright fields):
  - old batch: passthrough
  - new batch: each TIF stretched with its own p-low/p-high (full contrast per file)

per-batch:
  Each batch uses its own per-marker p-low/p-high (old vs new separately).

global (not recommended here):
  One reference from old batch applied to all files — fails when new batch is
  bimodal (dim BF ~50 vs bright BF ~200 interleaved).

  python scripts/harmonize_camsc_intensity.py \\
    --src ~/orcd/scratch/camsc/camsc_all \\
    --dst ~/orcd/scratch/camsc/camsc_all_harm \\
    --mode split --ref-max-index 10
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np
import tifffile

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


def _batch_name(idx: int, ref_max_index: int) -> str:
    return "old" if idx <= ref_max_index else "new"


def fit_reference_for_batch(
    paths: list[Path],
    batch: str,
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
        if _batch_name(idx, ref_max_index) != batch:
            continue
        arr = _load_gray(p, step=sample_step)
        vals = _values_for_stats(arr, marker, wt1_nonzero)
        pools[marker].append(vals)

    ref: dict[str, dict[str, float]] = {}
    for marker in _MARKERS:
        chunks = pools[marker]
        if not chunks:
            raise SystemExit(f"No files for batch={batch!r} marker={marker}")
        vals = np.concatenate(chunks)
        lo = float(np.percentile(vals, p_low))
        hi = float(np.percentile(vals, p_high))
        if hi <= lo + 1e-6:
            hi = lo + 1.0
        ref[marker] = {"p_low": lo, "p_high": hi, "n_ref_files": len(chunks)}
        print(
            f"  [{batch}] {marker}: p{p_low:g}={lo:.2f} p{p_high:g}={hi:.2f} "
            f"({len(chunks)} files)"
        )
    return ref


def apply_stretch(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    out = (arr - lo) / (hi - lo) * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def fit_image_percentiles(
    arr: np.ndarray,
    marker: str,
    p_low: float,
    p_high: float,
    wt1_nonzero: bool,
) -> tuple[float, float]:
    vals = _values_for_stats(arr, marker, wt1_nonzero)
    lo = float(np.percentile(vals, p_low))
    hi = float(np.percentile(vals, p_high))
    if hi <= lo + 1e-6:
        hi = lo + 1.0
    return lo, hi


def harmonize(
    src: Path,
    dst: Path,
    refs: dict[str, dict[str, dict[str, float]]],
    ref_max_index: int,
    mode: str,
    dry_run: bool,
    p_low: float,
    p_high: float,
    wt1_nonzero: bool,
) -> int:
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in collect_paths(src):
        marker, idx = _parse_marker_path(p)
        assert marker is not None
        marker = _normalize_marker(marker)
        batch = _batch_name(idx, ref_max_index)

        if mode in ("split", "split-per-image") and batch == "old":
            out_path = dst / p.name
            if dry_run:
                arr = _load_gray(p)
                print(f"[dry-run] {p.name} -> passthrough mean {arr.mean():.1f}")
            else:
                shutil.copy2(p, out_path)
            n += 1
            continue

        arr = _load_gray(p)
        if mode == "split-per-image":
            lo, hi = fit_image_percentiles(arr, marker, p_low, p_high, wt1_nonzero)
            tag = "per-image"
        else:
            ref = refs[batch if mode in ("split", "per-batch") else "global"]
            lo = ref[marker]["p_low"]
            hi = ref[marker]["p_high"]
            tag = batch
        out = apply_stretch(arr, lo, hi)
        out_path = dst / p.name
        if dry_run:
            print(f"[dry-run] {p.name} ({tag}) mean {arr.mean():.1f} => {out.mean():.1f}")
        else:
            tifffile.imwrite(out_path, out)
        n += 1
        if n % 50 == 0:
            print(f"  processed {n}...", flush=True)
    return n


def preview_indices(
    src: Path,
    refs: dict[str, dict[str, dict[str, float]]],
    ref_max_index: int,
    mode: str,
    indices: list[int],
    p_low: float,
    p_high: float,
    wt1_nonzero: bool,
) -> None:
    print("\nPreview (first glob match per marker):")
    for idx in indices:
        batch = _batch_name(idx, ref_max_index)
        print(f"  index {idx} ({batch})")
        for marker in _MARKERS:
            matches = list(src.glob(f"*{marker}_{idx}.tif"))
            if not matches:
                continue
            p = matches[0]
            m = _normalize_marker(marker)
            arr = _load_gray(p)
            if mode in ("split", "split-per-image") and batch == "old":
                out = np.clip(arr, 0, 255).astype(np.uint8)
                tag = "passthrough"
            elif mode == "split-per-image":
                lo, hi = fit_image_percentiles(arr, m, p_low, p_high, wt1_nonzero)
                out = apply_stretch(arr, lo, hi)
                tag = "per-image"
            else:
                ref = refs[batch if mode in ("split", "per-batch") else "global"]
                out = apply_stretch(arr, ref[m]["p_low"], ref[m]["p_high"])
                tag = batch
            print(
                f"    {marker} ({tag}): mean {arr.mean():.1f} -> {out.mean():.1f}  "
                f"p99 {np.percentile(arr, 99):.1f} -> {np.percentile(out, 99):.1f}"
            )


def main() -> None:
    p = argparse.ArgumentParser(description="Harmonize CaMSC flat TIF intensities")
    p.add_argument("--src", type=str, required=True)
    p.add_argument("--dst", type=str, required=True)
    p.add_argument(
        "--mode",
        choices=("split", "split-per-image", "per-batch", "global"),
        default="split-per-image",
        help="split-per-image=old passthrough + each new TIF own stretch (best for dim/bright mix)",
    )
    p.add_argument("--ref-max-index", type=int, default=10)
    p.add_argument("--p-low", type=float, default=1.0)
    p.add_argument("--p-high", type=float, default=99.0)
    p.add_argument("--wt1-nonzero", action="store_true", default=True)
    p.add_argument("--sample-step", type=int, default=4)
    p.add_argument("--preview-indices", type=str, default="8,13,14")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    src = Path(args.src).expanduser().resolve()
    dst = Path(args.dst).expanduser().resolve()
    if not src.is_dir():
        raise SystemExit(f"Missing src: {src}")

    paths = collect_paths(src)
    print(f"Found {len(paths)} marker TIFs in {src}")
    print(f"Mode: {args.mode}  (split index <= {args.ref_max_index} = old)\n")

    refs: dict[str, dict[str, dict[str, float]]] = {}
    if args.mode == "split-per-image":
        print("New batch: per-image stretch (no pooled reference).")
    elif args.mode in ("split", "per-batch"):
        for batch in ("old", "new"):
            if args.mode == "split" and batch == "old":
                continue
            print(f"Fitting reference for batch={batch!r}...")
            refs[batch] = fit_reference_for_batch(
                paths, batch, args.ref_max_index,
                args.p_low, args.p_high, args.wt1_nonzero, args.sample_step,
            )
        if args.mode == "per-batch":
            print("Fitting reference for batch='old'...")
            refs["old"] = fit_reference_for_batch(
                paths, "old", args.ref_max_index,
                args.p_low, args.p_high, args.wt1_nonzero, args.sample_step,
            )
    else:
        print("Fitting global reference from old batch...")
        refs["global"] = fit_reference_for_batch(
            paths, "old", args.ref_max_index,
            args.p_low, args.p_high, args.wt1_nonzero, args.sample_step,
        )

    preview = [int(x) for x in args.preview_indices.split(",") if x.strip()]
    preview_indices(
        src, refs, args.ref_max_index, args.mode, preview,
        args.p_low, args.p_high, args.wt1_nonzero,
    )

    print(f"\nWriting {'[dry-run] ' if args.dry_run else ''}to {dst} ...")
    n = harmonize(
        src, dst, refs, args.ref_max_index, args.mode, args.dry_run,
        args.p_low, args.p_high, args.wt1_nonzero,
    )
    meta = {
        "src": str(src),
        "dst": str(dst),
        "mode": args.mode,
        "ref_max_index": args.ref_max_index,
        "p_low": args.p_low,
        "p_high": args.p_high,
        "wt1_nonzero": args.wt1_nonzero,
        "reference": refs,
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
