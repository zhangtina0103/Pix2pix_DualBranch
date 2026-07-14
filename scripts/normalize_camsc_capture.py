#!/usr/bin/env python3
"""
Normalize nested CaMSC Capture*.tif batches into flat legacy names for prepare_camsc_bf.py.

New layout (20260709 Dropbox zip):
  caMSC_{pct}%/{field}/Capture#####.tif  (x3)
  — and sometimes labeled copies: BF.tif, Wt1.tif, hoecst.tif (typo for Hoechst)

IMPORTANT: Capture##### order is NOT a reliable channel key. Across fields, BF can be
first or last, and the two fluorescence captures swap. Prefer labeled files when
present; otherwise classify by RGB signature learned from the labeled key folder:

  BF.tif     → all RGB high (brightfield / "black and white")
  Wt1.tif    → R+G dominant, B ≈ 0
  hoecst.tif → G+B dominant, R ≈ 0  (Hoechst)

Writes flat files:
  CaMSC {pct}% 10x BF_{n}.tif
  CaMSC {pct}% 10x Hoechst_{n}.tif
  CaMSC {pct}% 10x WT1_{n}.tif

Use --index-offset 10 so new fields start at 11 (old batch uses 1–10).

Usage:
  python scripts/normalize_camsc_capture.py \\
    --src ~/orcd/scratch/camsc/20260709 \\
    --dst ~/orcd/scratch/camsc/camsc_all \\
    --merge-old ~/orcd/scratch/camsc/20260504 \\
    --index-offset 10
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

_PCT_DIR_RE = re.compile(r"^caMSC_(\d+)%$", re.IGNORECASE)
_CAPTURE_RE = re.compile(r"^Capture(\d+)\.tif$", re.IGNORECASE)
_LEGACY_RE = re.compile(
    r"^CaMSC\s+(\d+)%\s+(\d+x)\s+(BF|Hoechst|WT1)(?:_+(\d+))?\.tif$",
    re.IGNORECASE,
)

# Labeled filenames seen in Dropbox (hoecst is a typo for Hoechst).
_LABEL_CANDIDATES = {
    "BF": ("BF.tif", "bf.tif", "Brightfield.tif", "brightfield.tif"),
    "Hoechst": ("hoecst.tif", "hoechst.tif", "Hoechst.tif", "Hoecst.tif"),
    "WT1": ("Wt1.tif", "WT1.tif", "wt1.tif"),
}


def discover_field_dirs(src: Path, mag: str = "10x") -> list[dict]:
    rows: list[dict] = []
    for pct_dir in sorted(src.iterdir()):
        m = _PCT_DIR_RE.match(pct_dir.name)
        if not m or not pct_dir.is_dir():
            continue
        pct = m.group(1)
        for field_dir in sorted(
            pct_dir.iterdir(),
            key=lambda p: int(p.name) if p.name.isdigit() else p.name,
        ):
            if not field_dir.is_dir():
                continue
            rows.append({
                "pct": pct,
                "mag": mag,
                "field": field_dir.name,
                "dir": field_dir,
            })
    return rows


def copy_legacy_batch(old_src: Path, dst: Path) -> int:
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for path in sorted(old_src.glob("*.tif")):
        if not _LEGACY_RE.match(path.name):
            continue
        out = dst / path.name
        if out.exists():
            print(f"  [skip exists] {path.name}")
            continue
        shutil.copy2(path, out)
        n += 1
    return n


def _rgb_means(path: Path) -> tuple[float, float, float, float]:
    arr = np.asarray(Image.open(path))
    if arr.ndim != 3 or arr.shape[-1] < 3:
        raise ValueError(f"Expected RGB TIF at {path}, got shape {getattr(arr, 'shape', None)}")
    r, g, b = [float(arr[..., i].mean()) for i in range(3)]
    return r, g, b, float(arr.mean())


def classify_rgb(path: Path) -> str:
    """Map a Capture RGB TIF to BF / Hoechst / WT1 using the labeled Dropbox key."""
    r, g, b, mean = _rgb_means(path)
    # BF: brightfield — all channels high ("black and white" gray morphology)
    # Looser than mean>120 to catch dim BF (means ~113–119).
    if mean > 100.0 and min(r, g, b) > 70.0:
        return "BF"
    # WT1 key: R+G, B ≈ 0
    if b < 5.0 and r > 10.0:
        return "WT1"
    # Hoechst key (hoecst.tif): G+B, R ≈ 0 (B can be small on some fields)
    if r < 5.0 and g > 20.0:
        return "Hoechst"
    raise ValueError(
        f"Cannot classify {path.name}: R={r:.1f} G={g:.1f} B={b:.1f} mean={mean:.1f}"
    )


def load_channel_map(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    """Load CSV with columns pct_dir,field,BF,Hoechst,WT1 (Capture filenames)."""
    out: dict[tuple[str, str], dict[str, str]] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            key = (row["pct_dir"], str(row["field"]))
            out[key] = {
                "BF": row["BF"],
                "Hoechst": row["Hoechst"],
                "WT1": row["WT1"],
            }
    return out


def resolve_triplet(
    field_dir: Path,
    channel_map: dict[tuple[str, str], dict[str, str]] | None = None,
    pct_dir: str | None = None,
    field: str | None = None,
) -> dict[str, Path]:
    """Return {'BF': path, 'Hoechst': path, 'WT1': path} for one field folder."""
    out: dict[str, Path] = {}

    # 1) Prefer explicit labels when Dropbox has them
    for marker, names in _LABEL_CANDIDATES.items():
        for name in names:
            cand = field_dir / name
            if cand.is_file():
                out[marker] = cand
                break

    if len(out) == 3:
        return out

    # 2) Prefer precomputed manual map (Capture filenames per field)
    if channel_map is not None and pct_dir is not None and field is not None:
        mapped = channel_map.get((pct_dir, str(field)))
        if mapped is not None:
            for marker in ("BF", "Hoechst", "WT1"):
                cand = field_dir / mapped[marker]
                if not cand.is_file():
                    raise ValueError(f"{field_dir}: map lists missing file {mapped[marker]}")
                out[marker] = cand
            return out

    # 3) Fall back: classify Capture*.tif by RGB signature
    captures = []
    for tif in field_dir.glob("*.tif"):
        if _CAPTURE_RE.match(tif.name):
            captures.append(tif)
    if len(captures) != 3:
        missing = {"BF", "Hoechst", "WT1"} - set(out)
        raise ValueError(
            f"{field_dir}: need 3 Capture*.tif or full labels; "
            f"have labels={sorted(out)} captures={len(captures)} missing={sorted(missing)}"
        )

    classified: dict[str, Path] = {}
    unknown: list[Path] = []
    for path in captures:
        try:
            marker = classify_rgb(path)
        except ValueError:
            unknown.append(path)
            continue
        if marker in classified:
            raise ValueError(
                f"{field_dir}: two files classified as {marker}: "
                f"{classified[marker].name} and {path.name}"
            )
        classified[marker] = path

    # If exactly one unknown left, assign leftover marker (handles dim BF edge cases)
    if len(unknown) == 1:
        leftover = ({"BF", "Hoechst", "WT1"} - set(classified)).pop()
        classified[leftover] = unknown[0]
    elif unknown:
        raise ValueError(f"{field_dir}: unclassified captures {[p.name for p in unknown]}")

    for marker in ("BF", "Hoechst", "WT1"):
        if marker not in out:
            if marker not in classified:
                raise ValueError(f"{field_dir}: missing {marker} after RGB classify")
            out[marker] = classified[marker]
    return out


def write_normalized(
    rows: list[dict],
    dst: Path,
    index_offset: int,
    channel_map: dict[tuple[str, str], dict[str, str]] | None = None,
) -> tuple[int, list[str]]:
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    reports: list[str] = []
    for row in rows:
        pct = row["pct"]
        mag = row["mag"]
        field_dir: Path = row["dir"]
        pct_dir = field_dir.parent.name
        try:
            field_idx = int(row["field"])
        except ValueError:
            print(f"  [skip] non-numeric field dir {row['field']}", file=sys.stderr)
            continue

        try:
            triplet = resolve_triplet(
                field_dir, channel_map=channel_map, pct_dir=pct_dir, field=row["field"],
            )
        except ValueError as e:
            print(f"  [skip] {e}", file=sys.stderr)
            continue

        out_idx = index_offset + field_idx
        src_note = ", ".join(f"{k}={triplet[k].name}" for k in ("BF", "Hoechst", "WT1"))
        reports.append(f"{pct}%/{row['field']}: {src_note}")

        for marker in ("BF", "Hoechst", "WT1"):
            src_path = triplet[marker]
            out_name = f"CaMSC {pct}% {mag} {marker}_{out_idx}.tif"
            out_path = dst / out_name
            if out_path.exists():
                print(f"  [skip exists] {out_name}")
                continue
            shutil.copy2(src_path, out_path)
            n += 1
    return n, reports


def main() -> None:
    p = argparse.ArgumentParser(
        description="Normalize CaMSC Capture layout to legacy flat names (label-aware)"
    )
    p.add_argument("--src", type=str, required=True,
                   help="Extracted 20260709 root (contains caMSC_*%/ folders)")
    p.add_argument("--dst", type=str, required=True, help="Flat output dir (e.g. camsc_all/)")
    p.add_argument("--merge-old", type=str, default="",
                   help="Optional legacy flat dir (20260504) to copy first")
    p.add_argument("--mag", type=str, default="10x")
    p.add_argument(
        "--index-offset",
        type=int,
        default=10,
        help="Added to field subfolder id (default 10 → fields 1..15 → indices 11..25)",
    )
    p.add_argument(
        "--report",
        type=str,
        default="",
        help="Optional path to write per-field BF/Hoechst/WT1 source mapping",
    )
    p.add_argument(
        "--channel-map",
        type=str,
        default="",
        help="CSV map of Capture→channel (default: scripts/camsc_20260709_channel_map.csv if present)",
    )
    # Kept for CLI compatibility; Capture sort order is no longer used.
    p.add_argument(
        "--channel-order",
        type=str,
        default="auto",
        help="Ignored (map / label / RGB). Kept for compatibility.",
    )
    args = p.parse_args()

    src = Path(args.src).expanduser().resolve()
    dst = Path(args.dst).expanduser().resolve()
    if not src.is_dir():
        raise SystemExit(f"Source not found: {src}")

    rows = discover_field_dirs(src, mag=args.mag)
    if not rows:
        raise SystemExit(f"No field folders found under {src}")

    channel_map = None
    map_path = Path(args.channel_map).expanduser() if args.channel_map else (
        Path(__file__).resolve().parent / "camsc_20260709_channel_map.csv"
    )
    if map_path.is_file():
        channel_map = load_channel_map(map_path)
        print(f"Using channel map ({len(channel_map)} fields): {map_path}")
    else:
        print("No --channel-map CSV; using labels / RGB key classify")

    print(f"Found {len(rows)} field folders under {src}")
    print("Priority: BF.tif/Wt1.tif/hoecst.tif → channel-map CSV → RGB classify")
    if args.merge_old:
        old_src = Path(args.merge_old).expanduser().resolve()
        if not old_src.is_dir():
            raise SystemExit(f"--merge-old not found: {old_src}")
        n_old = copy_legacy_batch(old_src, dst)
        print(f"Copied {n_old} legacy TIFs from {old_src}")

    n_new, reports = write_normalized(rows, dst, args.index_offset, channel_map=channel_map)
    total = len(list(dst.glob("*.tif")))
    print(f"Wrote {n_new} normalized TIFs → {dst}")
    print(f"Combined pool: {total} TIFs ({total // 3} fields if all triplets complete)")

    if args.report:
        report_path = Path(args.report).expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("\n".join(reports) + "\n")
        print(f"Mapping report → {report_path}")
    else:
        print("Sample mappings:")
        for line in reports[:8]:
            print(f"  {line}")
        if len(reports) > 8:
            print(f"  ... ({len(reports)} fields total)")

    print("Next:")
    print(f"  export CAMSC_SRC={dst}")
    print("  # then harmonize (optional) → prepare_camsc_bf.py → retrain")


if __name__ == "__main__":
    main()
