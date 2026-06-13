#!/usr/bin/env python3
"""
YOLO CD3 downstream: train on real stains, evaluate frozen detector on generated test TIFFs.

Reference boxes = detector(real_B CD3). Predicted boxes = detector(fake_B CD3).
Metrics: count MAE, precision/recall/F1 (IoU-matched), bootstrap CI.

Examples:
  python scripts/run_hemit_yolo_downstream.py \\
    --weights weights/yolo_cd3_hemit.pt \\
    --manifest eval/hemit/manifest.csv \\
    --outdir eval/hemit/yolo_downstream
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.hemit_eval_path import setup_hemit_eval_path

setup_hemit_eval_path()

from hemit_eval.compare_models import load_manifest
from hemit_eval.yolo_cd3 import (
    compute_yolo_downstream,
    write_yolo_downstream_results,
    write_yolo_leaderboard,
)


def main() -> None:
    p = argparse.ArgumentParser(description="HEMIT YOLO CD3 downstream evaluation.")
    p.add_argument("--weights", type=str, required=True, help="Frozen YOLO weights (trained on real train)")
    p.add_argument("--srcdir", type=str, default=None)
    p.add_argument("--manifest", type=str, default=None)
    p.add_argument("--outdir", type=str, required=True)
    p.add_argument("--model-name", type=str, default="model")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou-threshold", type=float, default=0.5)
    p.add_argument("--imgsz", type=int, default=512)
    p.add_argument("--bootstrap-resamples", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("pip install ultralytics>=8.0") from exc

    weights = Path(args.weights).expanduser().resolve()
    if not weights.is_file():
        raise SystemExit(f"Missing weights: {weights}")

    yolo = YOLO(str(weights))
    outdir = Path(args.outdir).expanduser().resolve()

    if args.manifest:
        manifest = load_manifest(args.manifest)
        summaries = []
        for entry in manifest:
            model_out = outdir / entry["model"]
            per_tile, summary = compute_yolo_downstream(
                entry["srcdir"],
                yolo,
                model_name=entry["model"],
                conf=args.conf,
                iou_threshold=args.iou_threshold,
                imgsz=args.imgsz,
                bootstrap_resamples=args.bootstrap_resamples,
                seed=args.seed,
            )
            write_yolo_downstream_results(model_out, per_tile, summary)
            summaries.append(summary)
        leaderboard = write_yolo_leaderboard(summaries, outdir)
        print(json.dumps({
            "mode": "comparison",
            "outdir": str(outdir),
            "weights": str(weights),
            "models": [s["model"] for s in summaries],
            "leaderboard_csv": str(leaderboard),
        }, indent=2))
        return

    if not args.srcdir:
        p.error("Provide --srcdir or --manifest")

    per_tile, summary = compute_yolo_downstream(
        args.srcdir,
        yolo,
        model_name=args.model_name,
        conf=args.conf,
        iou_threshold=args.iou_threshold,
        imgsz=args.imgsz,
        bootstrap_resamples=args.bootstrap_resamples,
        seed=args.seed,
    )
    paths = write_yolo_downstream_results(outdir, per_tile, summary)
    print(json.dumps({
        "mode": "single",
        "outdir": str(outdir),
        "count_abs_error_mean": summary["metrics"]["count_abs_error"]["mean"],
        "f1_mean": summary["metrics"]["f1"]["mean"],
        **{k: str(v) for k, v in paths.items()},
    }, indent=2))


if __name__ == "__main__":
    main()
