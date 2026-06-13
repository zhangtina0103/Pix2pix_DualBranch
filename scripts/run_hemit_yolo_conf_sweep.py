#!/usr/bin/env python3
"""Sweep YOLO conf on one model to find a usable detection threshold."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.hemit_eval_path import setup_hemit_eval_path

setup_hemit_eval_path()

from hemit_eval.yolo_cd3 import compute_yolo_downstream


def main() -> None:
    p = argparse.ArgumentParser(description="Sweep YOLO conf for downstream eval.")
    p.add_argument("--weights", type=str, required=True)
    p.add_argument("--srcdir", type=str, required=True)
    p.add_argument("--outdir", type=str, required=True)
    p.add_argument("--model-name", type=str, default="probe")
    p.add_argument("--conf-values", type=str, default="0.01,0.05,0.1,0.15,0.25")
    p.add_argument("--ref-mode", type=str, default="yolo", choices=("yolo", "pseudo"))
    p.add_argument("--cd3-positive-only", action="store_true")
    p.add_argument("--imgsz", type=int, default=512)
    args = p.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("pip install ultralytics>=8.0") from exc

    yolo = YOLO(str(Path(args.weights).expanduser().resolve()))
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for conf_s in args.conf_values.split(","):
        conf = float(conf_s.strip())
        _, summary = compute_yolo_downstream(
            args.srcdir,
            yolo,
            model_name=args.model_name,
            conf=conf,
            imgsz=args.imgsz,
            ref_mode=args.ref_mode,
            cd3_positive_only=args.cd3_positive_only,
            bootstrap_resamples=2000,
        )
        m = summary["metrics"]
        rows.append({
            "conf": conf,
            "n_tiles": summary["n_tiles"],
            "n_skipped": summary.get("n_skipped_cd3_negative", 0),
            "mean_n_ref": m["mean_n_ref"]["mean"],
            "mean_n_pred": m["mean_n_pred"]["mean"],
            "count_abs_error": m["count_abs_error"]["mean"],
            "f1": m["f1"]["mean"],
            "degenerate": summary.get("degenerate", False),
            "ref_mode": args.ref_mode,
        })

    csv_path = outdir / "conf_sweep.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    best = [r for r in rows if not r["degenerate"] and r["mean_n_ref"] > 0]
    best.sort(key=lambda r: (-r["f1"], r["count_abs_error"]))
    pick = best[0] if best else None
    print(json.dumps({"outdir": str(outdir), "csv": str(csv_path), "recommended": pick, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
