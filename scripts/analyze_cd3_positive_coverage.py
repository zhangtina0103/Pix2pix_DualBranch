#!/usr/bin/env python3
"""CD3 tile coverage: nucleus-positive counts + mean-intensity enrichment."""

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

from hemit_eval.cd3_subset import (
    cd3_enrichment_threshold,
    is_cd3_enriched_tile,
    tile_mean_cd3_intensity,
)
from hemit_eval.image_io import list_fake_files, load_pair, resolve_image_dir
from hemit_eval.yolo_cd3 import cd3_positive_boxes


def main() -> None:
    p = argparse.ArgumentParser(description="CD3 tile coverage on test real_B stacks.")
    p.add_argument("--srcdir", type=str, required=True)
    p.add_argument("--outdir", type=str, default="eval/hemit/cd3_coverage")
    p.add_argument(
        "--percentiles", type=str, default="50,55,60,65,70",
        help="marker_percentile values for cd3_positive_boxes",
    )
    p.add_argument(
        "--top-fracs", type=str, default="0.05,0.10,0.15,0.20",
        help="Top fractions for CD3 mean-intensity enrichment",
    )
    args = p.parse_args()

    image_dir = resolve_image_dir(args.srcdir)
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    percentiles = [float(x.strip()) for x in args.percentiles.split(",") if x.strip()]
    top_fracs = [float(x.strip()) for x in args.top_fracs.split(",") if x.strip()]
    per_tile_pos: dict[float, list[dict]] = {pct: [] for pct in percentiles}
    tile_rows: list[dict] = []

    for fake_path in list_fake_files(image_dir):
        real, _, base = load_pair(fake_path)
        mean_cd3 = tile_mean_cd3_intensity(real)
        row = {"file_name": base, "cd3_mean_real": mean_cd3}
        for pct in percentiles:
            n_box = len(cd3_positive_boxes(real, marker_percentile=pct))
            per_tile_pos[pct].append({"file_name": base, "n_cd3_boxes": n_box, "percentile": pct})
            if pct == percentiles[0]:
                row["n_cd3_boxes"] = n_box
        tile_rows.append(row)

    scores = [r["cd3_mean_real"] for r in tile_rows]
    n_total = len(tile_rows)

    summary_path = outdir / "cd3_positive_coverage.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["marker_percentile", "n_cd3_positive_tiles", "n_total", "frac_positive", "total_boxes"])
        for pct in percentiles:
            tiles = per_tile_pos[pct]
            pos = [t for t in tiles if t["n_cd3_boxes"] >= 1]
            total_boxes = sum(t["n_cd3_boxes"] for t in tiles)
            w.writerow([pct, len(pos), n_total, len(pos) / max(n_total, 1), total_boxes])
            print(f"pct={pct:4.0f}: {len(pos):4d}/{n_total} tiles CD3+  ({total_boxes} boxes)")

    enrich_path = outdir / "cd3_enrichment_coverage.csv"
    enrich_flags: dict[float, list[bool]] = {}
    with enrich_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "top_frac", "n_enriched_tiles", "n_total", "frac_enriched",
            "mean_cd3_threshold", "n_overlap_cd3_positive",
        ])
        cd3pos_names = {r["file_name"] for r in per_tile_pos.get(60.0, per_tile_pos[percentiles[0]]) if r["n_cd3_boxes"] >= 1}
        for frac in top_fracs:
            thr = cd3_enrichment_threshold(scores, frac)
            flags = [is_cd3_enriched_tile(s, thr) for s in scores]
            enrich_flags[frac] = flags
            enriched_names = {r["file_name"] for r, ok in zip(tile_rows, flags) if ok}
            overlap = len(enriched_names & cd3pos_names)
            n_enr = sum(flags)
            w.writerow([frac, n_enr, n_total, n_enr / max(n_total, 1), thr, overlap])
            print(
                f"top={frac:4.2f}: {n_enr:4d}/{n_total} enriched tiles  "
                f"(thr={thr:.3f}, overlap w/ CD3+={overlap})"
            )

    default_frac = 0.10 if 0.10 in top_fracs else top_fracs[0]
    default_thr = cd3_enrichment_threshold(scores, default_frac)
    detail_path = outdir / "cd3_tile_scores.csv"
    with detail_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "file_name", "cd3_mean_real", "n_cd3_boxes",
            f"enriched_top_{default_frac:.2f}",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        default_flags = enrich_flags.get(default_frac, [])
        for row, enriched in zip(tile_rows, default_flags):
            w.writerow({
                "file_name": row["file_name"],
                "cd3_mean_real": f"{row['cd3_mean_real']:.6f}",
                "n_cd3_boxes": row.get("n_cd3_boxes", 0),
                f"enriched_top_{default_frac:.2f}": int(enriched),
            })

    print(f"Wrote {summary_path}")
    print(f"Wrote {enrich_path}")
    print(f"Wrote {detail_path} (default enrichment top_frac={default_frac}, thr={default_thr:.3f})")


if __name__ == "__main__":
    main()
