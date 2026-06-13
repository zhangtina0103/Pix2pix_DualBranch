#!/usr/bin/env python3
"""
YOLO CD3 downstream: train on real stains, evaluate frozen detector on generated test TIFFs.

Reference boxes = detector(real_B) or pseudo CD3+ boxes on real (--ref-mode).
Predicted boxes = detector(fake_B CD3).

Examples:
  # Recommended after weak test transfer: pseudo ref + CD3+ tiles only
  python scripts/run_hemit_yolo_downstream.py \\
    --weights weights/yolo_cd3_hemit.pt \\
    --manifest eval/hemit/manifest.csv \\
    --outdir eval/hemit/yolo_downstream_v2 \\
    --ref-mode pseudo --cd3-positive-only --conf 0.1

  # Find a usable conf threshold first (one model):
  python scripts/run_hemit_yolo_conf_sweep.py \\
    --weights weights/yolo_cd3_hemit.pt \\
    --srcdir results/hemit_fm_cross_attn_scratch_512/test_80/images \\
    --outdir eval/hemit/yolo_conf_sweep \\
    --ref-mode pseudo --cd3-positive-only
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
    p.add_argument("--conf", type=float, default=0.1)
    p.add_argument("--iou-threshold", type=float, default=0.25,
                   help="IoU for box matching (0.25 for sparse small cells; 0.5 is strict)")
    p.add_argument("--imgsz", type=int, default=512)
    p.add_argument("--ref-mode", type=str, default="pseudo", choices=("yolo", "pseudo"),
                   help="yolo=detector on real_B; pseudo=CD3+ pseudo-boxes on real (stable on sparse test)")
    p.add_argument("--cd3-positive-only", action="store_true",
                   help="Eval only tiles with >=1 pseudo CD3+ cell on real_B")
    p.add_argument("--cd3-marker-percentile", type=float, default=60,
                   help="Lower (e.g. 50) to include more CD3+ tiles — run analyze_cd3_positive_coverage.py first")
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
                ref_mode=args.ref_mode,
                cd3_positive_only=args.cd3_positive_only,
                cd3_marker_percentile=args.cd3_marker_percentile,
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
        ref_mode=args.ref_mode,
        cd3_positive_only=args.cd3_positive_only,
        cd3_marker_percentile=args.cd3_marker_percentile,
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
