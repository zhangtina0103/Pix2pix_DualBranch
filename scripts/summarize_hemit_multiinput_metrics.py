#!/usr/bin/env python3
"""Print comparison table from multi-input extended_metrics_summary.json files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

METRICS = ("pearson", "spearman", "ssim", "psnr", "mae", "rmse", "lpips")


def load_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    p = argparse.ArgumentParser(description="Compare HEMIT multi-input metric summaries")
    p.add_argument(
        "--summaries",
        nargs="+",
        required=True,
        metavar="LABEL=PATH",
        help="e.g. H&E=results/.../extended_metrics_summary.json",
    )
    p.add_argument("--channel", choices=("cd3", "panck", "average"), default="average")
    args = p.parse_args()

    entries: list[tuple[str, dict]] = []
    for spec in args.summaries:
        if "=" not in spec:
            raise SystemExit(f"Expected LABEL=PATH, got {spec!r}")
        label, path = spec.split("=", 1)
        entries.append((label, load_summary(Path(path).expanduser())))

    ch = args.channel
    scope = "average" if ch == "average" else "channels"
    key = ch if ch == "average" else ch

    header = f"{'Input':14s}" + "".join(f"{m:>12s}" for m in METRICS)
    print(header)
    print("-" * len(header))
    for label, summary in entries:
        block = summary[scope][key] if scope == "channels" else summary["average"]
        parts = []
        for m in METRICS:
            if m not in block:
                parts.append(f"{'N/A':>12s}")
                continue
            parts.append(f"{block[m]['mean']:12.4f}")
        print(f"{label:14s}" + "".join(parts))


if __name__ == "__main__":
    main()
