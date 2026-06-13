#!/usr/bin/env python3
"""Print HNSCC model leaderboard from score.csv files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

MARKERS = ["cd3", "cd8", "foxp3", "panck"]


def read_score_csv(path: Path) -> dict[str, float]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    out: dict[str, float] = {}
    for m in MARKERS:
        vals = [float(r[f"{m}_pearson"]) for r in rows if r.get(f"{m}_pearson")]
        if vals:
            out[f"{m}_pearson"] = sum(vals) / len(vals)
    for key in ("average_pearson", "average_ssim"):
        vals = [float(r[key]) for r in rows if r.get(key)]
        if vals:
            out[key] = sum(vals) / len(vals)
    out["n"] = float(len(rows))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results-root", type=Path, default=Path("results"))
    p.add_argument("--epoch", type=int, default=80)
    p.add_argument(
        "--models",
        nargs="*",
        default=[
            "hnscc_pix2pix_resnet9_512",
            "hnscc_cut_joint_512",
            "hnscc_asp_joint_512",
            "hnscc_cyclegan_joint_512",
            "hnscc_vanilla_fm_joint_perc_scratch_512",
            "hnscc_fm_cross_attn_scratch_512",
        ],
    )
    args = p.parse_args()

    rows = []
    for name in args.models:
        csv_path = args.results_root / name / f"test_{args.epoch}" / "images" / "score.csv"
        if not csv_path.is_file():
            csv_path = args.results_root / name / f"test_{args.epoch}" / "score.csv"
        if not csv_path.is_file():
            print(f"[skip] {name}: no score.csv")
            continue
        stats = read_score_csv(csv_path)
        rows.append((name, stats))

    if not rows:
        print("No score.csv files found.")
        return

    rows.sort(key=lambda x: x[1].get("average_pearson", -1.0), reverse=True)
    print(f"HNSCC leaderboard @ epoch {args.epoch} (avg Pearson, higher=better)\n")
    print(f"{'rank':<5} {'model':<45} {'avg_pearson':>12} {'cd3':>8} {'cd8':>8} {'foxp3':>8} {'panck':>8}")
    for i, (name, s) in enumerate(rows, 1):
        print(
            f"{i:<5} {name:<45} "
            f"{s.get('average_pearson', float('nan')):12.4f} "
            f"{s.get('cd3_pearson', float('nan')):8.4f} "
            f"{s.get('cd8_pearson', float('nan')):8.4f} "
            f"{s.get('foxp3_pearson', float('nan')):8.4f} "
            f"{s.get('panck_pearson', float('nan')):8.4f}"
        )

    winner = rows[0][0]
    print(f"\nLeader: {winner}")
    if "cross_attn" in winner:
        print("Cross-attn FM is #1.")
    else:
        print("Cross-attn FM is NOT #1 yet — check training or tune FM_CHANNEL_WEIGHTS / epochs.")


if __name__ == "__main__":
    main()
