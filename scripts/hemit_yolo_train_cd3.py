#!/usr/bin/env python3
"""
Train YOLO CD3 detector on real HEMIT stains only.

Usage:
  python scripts/hemit_yolo_prepare_cd3_dataset.py --hemit-root /path/to/hemit
  python scripts/hemit_yolo_train_cd3.py --data datasets/hemit_cd3_yolo/hemit_cd3.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description="Train YOLOv8 CD3 detector on real HEMIT stains.")
    p.add_argument("--data", type=str, required=True, help="Dataset yaml from prepare script")
    p.add_argument("--model", type=str, default="yolov8n.pt", help="Ultralytics base checkpoint")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=512)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--project", type=str, default="runs/hemit_yolo")
    p.add_argument("--name", type=str, default="cd3_real")
    p.add_argument("--weights-out", type=str, default="weights/yolo_cd3_hemit.pt",
                   help="Copy best.pt here after training")
    args = p.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "ultralytics not installed. Run: pip install ultralytics>=8.0"
        ) from exc

    data_path = Path(args.data).expanduser().resolve()
    if not data_path.is_file():
        raise SystemExit(f"Missing dataset yaml: {data_path}")

    yolo = YOLO(args.model)
    results = yolo.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        exist_ok=True,
        patience=20,
        save=True,
        verbose=True,
    )

    best = Path(args.project) / args.name / "weights" / "best.pt"
    if not best.is_file():
        best = Path(results.save_dir) / "weights" / "best.pt"  # type: ignore[attr-defined]
    if not best.is_file():
        raise SystemExit(f"Training finished but best.pt not found under {args.project}/{args.name}")

    weights_out = Path(args.weights_out).expanduser().resolve()
    weights_out.parent.mkdir(parents=True, exist_ok=True)
    weights_out.write_bytes(best.read_bytes())
    print(f"Copied best weights → {weights_out}")


if __name__ == "__main__":
    main()
