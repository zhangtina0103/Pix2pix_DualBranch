#!/usr/bin/env python3
"""Count CD3+ test tiles under different pseudo-label thresholds (power analysis)."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.hemit_eval_path import setup_hemit_eval_path

setup_hemit_eval_path()

from hemit_eval.image_io import list_fake_files, load_pair, resolve_image_dir
from hemit_eval.yolo_cd3 import cd3_positive_boxes


def main() -> None:
    p = argparse.ArgumentParser(description="CD3+ tile coverage on test real_B stacks.")
    p.add_argument("--srcdir", type=str, required=True)
    p.add_argument("--outdir", type=str, default="eval/hemit/cd3_coverage")
    p.add_argument(
        "--percentiles", type=str, default="50,55,60,65,70",
        help="marker_percentile values for cd3_positive_boxes",
    )
    args = p.parse_args()

    image_dir = resolve_image_dir(args.srcdir)
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    percentiles = [float(x.strip()) for x in args.percentiles.split(",") if x.strip()]
    rows: list[dict] = []
    per_tile: dict[float, list[dict]] = {pct: [] for pct in percentiles}

    for fake_path in list_fake_files(image_dir):
        real, _, base = load_pair(fake_path)
        for pct in percentiles:
            n_box = len(cd3_positive_boxes(real, marker_percentile=pct))
            per_tile[pct].append({"file_name": base, "n_cd3_boxes": n_box, "percentile": pct})
        rows.append({"file_name": base})

    summary_path = outdir / "cd3_positive_coverage.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["marker_percentile", "n_cd3_positive_tiles", "n_total", "frac_positive", "total_boxes"])
        for pct in percentiles:
            tiles = per_tile[pct]
            pos = [t for t in tiles if t["n_cd3_boxes"] >= 1]
            total_boxes = sum(t["n_cd3_boxes"] for t in tiles)
            n_total = len(tiles)
            w.writerow([pct, len(pos), n_total, len(pos) / max(n_total, 1), total_boxes])
            print(f"pct={pct:4.0f}: {len(pos):4d}/{n_total} tiles CD3+  ({total_boxes} boxes)")

    detail_path = outdir / "cd3_positive_per_tile_p60.csv"
    with detail_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file_name", "n_cd3_boxes", "percentile"])
        w.writeheader()
        w.writerows(per_tile.get(60.0, per_tile[percentiles[0]]))

    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
