#!/usr/bin/env python3
"""Merge multiple robustness manifests (HEMIT + diffusion + future models)."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def load_manifest(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            model = (row.get("model") or "").strip()
            srcdir = (row.get("srcdir") or "").strip()
            if model and srcdir:
                rows.append({"model": model, "srcdir": srcdir})
    if not rows:
        raise ValueError(f"Empty manifest: {path}")
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description="Concatenate robustness manifest CSVs.")
    p.add_argument("--manifest", action="append", required=True,
                   help="Input manifest (repeat for multiple)")
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--allow-duplicate-models", action="store_true",
                   help="Keep duplicate model labels (default: error)")
    args = p.parse_args()

    merged: list[dict[str, str]] = []
    seen_models: set[str] = set()

    for raw in args.manifest:
        path = Path(raw).expanduser().resolve()
        if not path.is_file():
            raise SystemExit(f"Missing manifest: {path}")
        for row in load_manifest(path):
            if row["model"] in seen_models and not args.allow_duplicate_models:
                raise SystemExit(
                    f"Duplicate model '{row['model']}' from {path}. "
                    "Rename in source manifest or pass --allow-duplicate-models."
                )
            seen_models.add(row["model"])
            merged.append(row)

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "srcdir"])
        writer.writeheader()
        writer.writerows(merged)

    print(f"Wrote {out_path} ({len(merged)} models)")
    for row in merged:
        print(f"  {row['model']}: {row['srcdir']}")


if __name__ == "__main__":
    main()
