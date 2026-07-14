#!/usr/bin/env python3
"""
Verify camsc_all channel mapping against the Dropbox key:

  BF     = brightfield (B&W / all RGB high)
  WT1    = R+G dominant, B ≈ 0   (Wt1.tif)
  Hoechst= G+B dominant, R ≈ 0   (hoecst.tif)

Also cross-checks:
  - scripts/camsc_20260709_channel_map.csv (expected Capture sources)
  - optional channel_map_applied.txt
  - optional labeled BF.tif / Wt1.tif / hoecst.tif under 20260709

Usage (on ORCD):
  python scripts/verify_camsc_channel_map.py \\
    --camsc-all ~/orcd/scratch/camsc/camsc_all \\
    --src-new ~/orcd/scratch/camsc/20260709 \\
    --channel-map scripts/camsc_20260709_channel_map.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

_LEGACY_RE = re.compile(
    r"^CaMSC\s+(\d+)%\s+(\d+x)\s+(BF|Hoechst|WT1)_(\d+)\.tif$",
    re.IGNORECASE,
)


def rgb_means(path: Path) -> tuple[float, float, float, float]:
    arr = np.asarray(Image.open(path))
    if arr.ndim == 2:
        m = float(arr.mean())
        return m, m, m, m
    if arr.ndim != 3 or arr.shape[-1] < 3:
        raise ValueError(f"Unexpected shape {arr.shape} at {path}")
    r, g, b = [float(arr[..., i].mean()) for i in range(3)]
    return r, g, b, float(arr.mean())


def classify_key(r: float, g: float, b: float, mean: float) -> str:
    """Same key as Dropbox labels / normalize script."""
    if mean > 100.0 and min(r, g, b) > 70.0:
        return "BF"
    if b < 5.0 and r > 10.0:
        return "WT1"
    if r < 5.0 and g > 20.0:
        return "Hoechst"
    return "UNKNOWN"


def load_map(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    out: dict[tuple[str, str], dict[str, str]] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            out[(row["pct_dir"], str(row["field"]))] = {
                "BF": row["BF"],
                "Hoechst": row["Hoechst"],
                "WT1": row["WT1"],
            }
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Verify CaMSC channel mapping against Dropbox key")
    p.add_argument("--camsc-all", type=str, required=True)
    p.add_argument("--src-new", type=str, default="", help="20260709 root (optional but recommended)")
    p.add_argument("--channel-map", type=str, default="scripts/camsc_20260709_channel_map.csv")
    p.add_argument("--ref-max-index", type=int, default=10, help="index <= this = old batch")
    args = p.parse_args()

    dst = Path(args.camsc_all).expanduser().resolve()
    src_new = Path(args.src_new).expanduser().resolve() if args.src_new else None
    map_path = Path(args.channel_map).expanduser()
    if not map_path.is_file():
        map_path = Path(__file__).resolve().parent / "camsc_20260709_channel_map.csv"
    if not dst.is_dir():
        raise SystemExit(f"Missing --camsc-all: {dst}")
    if not map_path.is_file():
        raise SystemExit(f"Missing channel map: {map_path}")

    channel_map = load_map(map_path)
    print(f"camsc_all:    {dst}")
    print(f"channel map:  {map_path} ({len(channel_map)} fields)")
    print(f"src-new:      {src_new if src_new else '(not provided)'}")
    print()

    # Index flat pool
    by_key: dict[tuple[str, str, int], Path] = {}
    for path in sorted(dst.glob("*.tif")):
        m = _LEGACY_RE.match(path.name)
        if not m:
            continue
        pct, _mag, marker, idx = m.group(1), m.group(2), m.group(3), int(m.group(4))
        marker = "BF" if marker.upper() == "BF" else marker.capitalize()
        if marker.lower() == "hoechst":
            marker = "Hoechst"
        if marker.upper() == "WT1":
            marker = "WT1"
        by_key[(pct, marker, idx)] = path

    # Group into triplets
    fields: dict[tuple[str, int], set[str]] = defaultdict(set)
    for pct, marker, idx in by_key:
        fields[(pct, idx)].add(marker)

    n_complete = sum(1 for markers in fields.values() if markers >= {"BF", "Hoechst", "WT1"})
    n_old = sum(1 for (_p, i) in fields if i <= args.ref_max_index)
    n_new = sum(1 for (_p, i) in fields if i > args.ref_max_index)
    print(f"Flat pool: {len(by_key)} marker TIFs | {n_complete} complete triplets "
          f"(old idx≤{args.ref_max_index}: {n_old}, new: {n_new})")

    incomplete = [(pct, idx, sorted(markers)) for (pct, idx), markers in sorted(fields.items())
                  if markers < {"BF", "Hoechst", "WT1"}]
    if incomplete:
        print(f"INCOMPLETE triplets: {len(incomplete)}")
        for pct, idx, markers in incomplete[:10]:
            print(f"  {pct}% idx {idx}: have {markers}")
    else:
        print("All discovered fields have BF+Hoechst+WT1 ✓")

    # --- KEY CHECK: classify every new-batch flat file ---
    print("\n=== RGB key check on camsc_all (new indices) ===")
    mismatches = []
    unknown = []
    ok = 0
    by_marker_ok = Counter()
    samples = []

    for (pct, idx), markers in sorted(fields.items()):
        if idx <= args.ref_max_index:
            continue
        if markers < {"BF", "Hoechst", "WT1"}:
            continue
        for marker in ("BF", "Hoechst", "WT1"):
            path = by_key[(pct, marker, idx)]
            r, g, b, mean = rgb_means(path)
            pred = classify_key(r, g, b, mean)
            row = {
                "pct": pct, "idx": idx, "label": marker, "pred": pred,
                "r": r, "g": g, "b": b, "mean": mean, "file": path.name,
            }
            if pred == "UNKNOWN":
                unknown.append(row)
            elif pred != marker:
                mismatches.append(row)
            else:
                ok += 1
                by_marker_ok[marker] += 1
            if len(samples) < 6 and marker == "BF":
                samples.append(row)

    n_checked = ok + len(mismatches) + len(unknown)
    print(f"Checked {n_checked} new-batch labeled files")
    print(f"  match key:     {ok}")
    print(f"  mismatch:      {len(mismatches)}")
    print(f"  unknown/edge:  {len(unknown)}")
    print(f"  by marker OK:  {dict(by_marker_ok)}")

    if samples:
        print("\nSample BF means (should be bright ~110–220):")
        for s in samples:
            print(f"  {s['file']}: mean={s['mean']:.1f} R/G/B={s['r']:.0f}/{s['g']:.0f}/{s['b']:.0f}")

    if mismatches:
        print("\nMISMATCHES (label in filename ≠ RGB key):")
        for row in mismatches[:30]:
            print(
                f"  {row['file']}: labeled={row['label']} key={row['pred']} "
                f"R/G/B={row['r']:.1f}/{row['g']:.1f}/{row['b']:.1f} mean={row['mean']:.1f}"
            )
        if len(mismatches) > 30:
            print(f"  ... +{len(mismatches) - 30} more")

    if unknown:
        print("\nUNKNOWN / edge (still inspect):")
        for row in unknown[:20]:
            print(
                f"  {row['file']}: labeled={row['label']} "
                f"R/G/B={row['r']:.1f}/{row['g']:.1f}/{row['b']:.1f} mean={row['mean']:.1f}"
            )

    # --- Cross-check channel map vs source Capture files ---
    if src_new and src_new.is_dir():
        print("\n=== Cross-check channel-map CSV vs source Capture RGB ===")
        map_bad = []
        map_ok = 0
        for (pct_dir, field), caps in sorted(
            channel_map.items(),
            key=lambda item: (item[0][0], int(item[0][1])),
        ):
            field_dir = src_new / pct_dir / field
            if not field_dir.is_dir():
                map_bad.append((pct_dir, field, "missing field dir"))
                continue
            for marker in ("BF", "Hoechst", "WT1"):
                cap_path = field_dir / caps[marker]
                if not cap_path.is_file():
                    map_bad.append((pct_dir, field, f"missing {caps[marker]}"))
                    continue
                r, g, b, mean = rgb_means(cap_path)
                pred = classify_key(r, g, b, mean)
                if pred != marker and pred != "UNKNOWN":
                    map_bad.append(
                        (pct_dir, field, f"{marker} map={caps[marker]} key={pred} "
                         f"R/G/B={r:.0f}/{g:.0f}/{b:.0f}")
                    )
                elif pred == "UNKNOWN":
                    # leftover dim-BF cases: if other two match, accept as BF
                    if marker == "BF" and mean > 100 and min(r, g, b) > 60:
                        map_ok += 1
                    else:
                        map_bad.append(
                            (pct_dir, field, f"{marker} map={caps[marker]} UNKNOWN "
                             f"R/G/B={r:.0f}/{g:.0f}/{b:.0f} mean={mean:.0f}")
                        )
                else:
                    map_ok += 1
        print(f"Source Capture checks OK: {map_ok}")
        if map_bad:
            print(f"Source issues: {len(map_bad)}")
            for item in map_bad[:25]:
                print(f"  {item}")
        else:
            print("All channel-map Captures match key ✓")

        # Labeled key folder sanity
        key_dir = src_new / "caMSC_90%" / "1"
        if key_dir.is_dir() and (key_dir / "BF.tif").is_file():
            print("\n=== Dropbox labeled key folder caMSC_90%/1 ===")
            for label, fname in (("BF", "BF.tif"), ("Hoechst", "hoecst.tif"), ("WT1", "Wt1.tif")):
                path = key_dir / fname
                if not path.is_file():
                    print(f"  missing {fname}")
                    continue
                r, g, b, mean = rgb_means(path)
                pred = classify_key(r, g, b, mean)
                status = "OK" if pred == label else f"KEY≠{pred}"
                print(f"  {fname}: labeled={label} key={pred} [{status}] "
                      f"R/G/B={r:.0f}/{g:.0f}/{b:.0f}")

            # flat index 11 for 90% should match labels
            flat_idx = 10 + 1
            for marker, fname in (("BF", "BF.tif"), ("Hoechst", "hoecst.tif"), ("WT1", "Wt1.tif")):
                flat = by_key.get(("90", marker, flat_idx))
                lab = key_dir / fname
                if flat and lab.is_file():
                    same = flat.read_bytes() == lab.read_bytes()
                    print(f"  flat 90% idx{flat_idx} {marker} == {fname}? {same}")

    # Verdict
    print("\n=== VERDICT ===")
    if mismatches:
        print(f"FAIL: {len(mismatches)} flat files disagree with Dropbox RGB key")
        sys.exit(1)
    if unknown:
        print(f"WARN: {len(unknown)} edge cases (usually dim BF). Inspect list above.")
        print("If those are dim BF labeled BF, you can proceed.")
    else:
        print("PASS: all new-batch flat labels match Dropbox key ✓")
    print("Next: harmonize → prepare kfold → SKIP_IF_DONE=0 retrain")


if __name__ == "__main__":
    main()
