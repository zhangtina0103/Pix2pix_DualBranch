"""CD3 YOLO downstream: pseudo-labels from real stains + detection metrics on generated."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import closing, disk, remove_small_objects

from hemit_eval.image_io import list_fake_files, load_pair, resolve_image_dir
from hemit_eval.statistics import summarize_values

CD3_CLASS_ID = 0
CD3_CHANNEL_INDEX = 1


@dataclass(frozen=True)
class Box:
    """Axis-aligned box in pixel coordinates (x1, y1, x2, y2) inclusive-exclusive like xyxy."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    def to_yolo_line(self, img_w: int, img_h: int, class_id: int = CD3_CLASS_ID) -> str:
        xc = (self.x1 + self.x2) / 2.0 / img_w
        yc = (self.y1 + self.y2) / 2.0 / img_h
        w = self.width / img_w
        h = self.height / img_h
        return f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"

    @classmethod
    def from_yolo_line(cls, line: str, img_w: int, img_h: int) -> Box:
        parts = line.strip().split()
        if len(parts) != 5:
            raise ValueError(f"Expected 5 YOLO fields, got {line!r}")
        _, xc, yc, w, h = parts
        xc_f, yc_f, w_f, h_f = float(xc), float(yc), float(w), float(h)
        bw, bh = w_f * img_w, h_f * img_h
        x1 = xc_f * img_w - bw / 2.0
        y1 = yc_f * img_h - bh / 2.0
        return cls(x1, y1, x1 + bw, y1 + bh)

    @classmethod
    def from_region_bbox(
        cls,
        min_row: int,
        min_col: int,
        max_row: int,
        max_col: int,
        *,
        pad: int = 8,
        min_side: int = 24,
        img_w: int | None = None,
        img_h: int | None = None,
    ) -> Box:
        x1 = max(0, min_col - pad)
        y1 = max(0, min_row - pad)
        x2 = max_col + pad
        y2 = max_row + pad
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = max(x2 - x1, float(min_side))
        h = max(y2 - y1, float(min_side))
        x1, y1 = cx - w / 2.0, cy - h / 2.0
        x2, y2 = cx + w / 2.0, cy + h / 2.0
        if img_w is not None:
            x1 = max(0.0, min(float(img_w), x1))
            x2 = max(0.0, min(float(img_w), x2))
        if img_h is not None:
            y1 = max(0.0, min(float(img_h), y1))
            y2 = max(0.0, min(float(img_h), y2))
        return cls(x1, y1, x2, y2)


def segment_nuclei(dapi: np.ndarray, *, min_area: int = 36) -> np.ndarray:
    img = np.clip(dapi, 0, 255).astype(np.float32)
    if img.max() <= 0:
        return np.zeros_like(img, dtype=bool)
    mask = img >= threshold_otsu(img)
    mask = closing(mask, disk(2))
    return remove_small_objects(mask, min_size=min_area)


def cd3_positive_boxes(
    stack: np.ndarray,
    *,
    min_area: int = 36,
    pad: int = 8,
    min_side: int = 24,
) -> list[Box]:
    """Pseudo-label CD3+ nuclei on a real multiplex stack (H,W,3)."""
    h, w = stack.shape[:2]
    nuclei = segment_nuclei(stack[..., 0], min_area=min_area)
    labeled = label(nuclei)
    if labeled.max() == 0:
        return []
    marker = stack[..., CD3_CHANNEL_INDEX]
    props = list(regionprops(labeled, intensity_image=marker))
    if not props:
        return []
    means = [p.mean_intensity for p in props]
    thr = max(
        threshold_otsu(marker[marker > 0]) if np.any(marker > 0) else 0.0,
        float(np.percentile(means, 60)),
    )
    boxes: list[Box] = []
    for prop in props:
        if prop.mean_intensity < thr:
            continue
        min_row, min_col, max_row, max_col = prop.bbox
        boxes.append(
            Box.from_region_bbox(
                min_row, min_col, max_row, max_col,
                pad=pad, min_side=min_side, img_w=w, img_h=h,
            )
        )
    return boxes


def stack_to_cd3_uint8(stack: np.ndarray) -> np.ndarray:
    """Single-channel CD3 duplicated to HxWx3 uint8 for YOLO inference."""
    cd3 = np.clip(stack[..., CD3_CHANNEL_INDEX], 0, 255).astype(np.uint8)
    return np.stack([cd3, cd3, cd3], axis=-1)


def box_iou(a: Box, b: Box) -> float:
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = a.area + b.area - inter
    if union <= 0:
        return 0.0
    return inter / union


def match_boxes(
    reference: Sequence[Box],
    predicted: Sequence[Box],
    *,
    iou_threshold: float = 0.5,
) -> tuple[int, int, int]:
    """Greedy IoU matching: return (tp, fp, fn)."""
    if not reference and not predicted:
        return 0, 0, 0
    if not reference:
        return 0, len(predicted), 0
    if not predicted:
        return 0, 0, len(reference)

    pairs: list[tuple[float, int, int]] = []
    for ri, ref in enumerate(reference):
        for pi, pred in enumerate(predicted):
            iou = box_iou(ref, pred)
            if iou >= iou_threshold:
                pairs.append((iou, ri, pi))
    pairs.sort(reverse=True)

    matched_ref: set[int] = set()
    matched_pred: set[int] = set()
    tp = 0
    for _, ri, pi in pairs:
        if ri in matched_ref or pi in matched_pred:
            continue
        matched_ref.add(ri)
        matched_pred.add(pi)
        tp += 1
    fp = len(predicted) - len(matched_pred)
    fn = len(reference) - len(matched_ref)
    return tp, fp, fn


def detection_prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if fn == 0 else 0.0)
    recall = tp / (tp + fn) if (tp + fn) > 0 else (1.0 if fp == 0 else 0.0)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return float(precision), float(recall), float(f1)


def _boxes_from_yolo_result(result: Any, conf_threshold: float) -> list[Box]:
    boxes: list[Box] = []
    if result.boxes is None or len(result.boxes) == 0:
        return boxes
    xyxy = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    for (x1, y1, x2, y2), conf in zip(xyxy, confs):
        if float(conf) < conf_threshold:
            continue
        boxes.append(Box(float(x1), float(y1), float(x2), float(y2)))
    return boxes


def run_yolo_on_cd3(model: Any, stack: np.ndarray, *, conf: float = 0.25, imgsz: int | None = None) -> list[Box]:
    img = stack_to_cd3_uint8(stack)
    kwargs: dict[str, Any] = {"verbose": False, "conf": conf}
    if imgsz is not None:
        kwargs["imgsz"] = imgsz
    results = model.predict(img, **kwargs)
    if not results:
        return []
    return _boxes_from_yolo_result(results[0], conf)


def compute_tile_detection(
    real: np.ndarray,
    fake: np.ndarray,
    model: Any,
    *,
    conf: float = 0.25,
    iou_threshold: float = 0.5,
    imgsz: int | None = None,
) -> dict[str, Any]:
    ref_boxes = run_yolo_on_cd3(model, real, conf=conf, imgsz=imgsz)
    pred_boxes = run_yolo_on_cd3(model, fake, conf=conf, imgsz=imgsz)
    tp, fp, fn = match_boxes(ref_boxes, pred_boxes, iou_threshold=iou_threshold)
    precision, recall, f1 = detection_prf(tp, fp, fn)
    n_ref, n_pred = len(ref_boxes), len(pred_boxes)
    return {
        "n_ref": n_ref,
        "n_pred": n_pred,
        "count_abs_error": abs(n_ref - n_pred),
        "count_signed_error": n_pred - n_ref,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def compute_yolo_downstream(
    srcdir: str | Path,
    model: Any,
    *,
    model_name: str = "model",
    conf: float = 0.25,
    iou_threshold: float = 0.5,
    imgsz: int | None = None,
    bootstrap_resamples: int = 10000,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    image_dir = resolve_image_dir(srcdir)
    per_tile: list[dict[str, Any]] = []
    for fake_path in list_fake_files(image_dir):
        real, fake, base = load_pair(fake_path)
        row = {
            "file_name": base,
            "model": model_name,
            **compute_tile_detection(
                real, fake, model, conf=conf, iou_threshold=iou_threshold, imgsz=imgsz,
            ),
        }
        per_tile.append(row)

    metrics = ("count_abs_error", "precision", "recall", "f1")
    higher_better = {"count_abs_error": False, "precision": True, "recall": True, "f1": True}
    summary: dict[str, Any] = {
        "model": model_name,
        "n_tiles": len(per_tile),
        "image_dir": str(image_dir),
        "conf": conf,
        "iou_threshold": iou_threshold,
        "metrics": {},
    }
    for name in metrics:
        vals = np.array([r[name] for r in per_tile], dtype=np.float64)
        summary["metrics"][name] = summarize_values(
            vals,
            n_resamples=bootstrap_resamples,
            random_state=seed,
            higher_is_better=higher_better[name],
        )
    summary["metrics"]["mean_n_ref"] = summarize_values(
        np.array([r["n_ref"] for r in per_tile], dtype=np.float64),
        n_resamples=bootstrap_resamples,
        random_state=seed,
    )
    summary["metrics"]["mean_n_pred"] = summarize_values(
        np.array([r["n_pred"] for r in per_tile], dtype=np.float64),
        n_resamples=bootstrap_resamples,
        random_state=seed,
    )
    mean_ref = summary["metrics"]["mean_n_ref"]["mean"]
    summary["degenerate"] = bool(mean_ref == 0.0)
    if summary["degenerate"]:
        summary["warning"] = (
            "Detector found 0 cells on real_B across all tiles — metrics are not meaningful."
        )
    return per_tile, summary


def write_yolo_downstream_results(
    outdir: str | Path,
    per_tile: list[dict[str, Any]],
    summary: dict[str, Any],
) -> dict[str, Path]:
    outdir = Path(outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    per_tile_path = outdir / "yolo_cd3_per_tile.csv"
    if per_tile:
        with per_tile_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(per_tile[0].keys()))
            w.writeheader()
            w.writerows(per_tile)
    summary_path = outdir / "yolo_cd3_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    flat_path = outdir / "yolo_cd3_summary.csv"
    with flat_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "metric", "mean", "ci_low", "ci_high", "higher_is_better"])
        for metric, stats in summary.get("metrics", {}).items():
            w.writerow([
                summary["model"],
                metric,
                stats.get("mean"),
                stats.get("ci_low"),
                stats.get("ci_high"),
                stats.get("higher_is_better", True),
            ])
    return {"per_tile": per_tile_path, "summary_json": summary_path, "summary_csv": flat_path}


def write_yolo_leaderboard(summaries: list[dict[str, Any]], outdir: str | Path) -> Path:
    outdir = Path(outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "yolo_cd3_leaderboard.csv"
    metric_names = ("count_abs_error", "f1", "precision", "recall")
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "n_tiles", "metric", "mean", "ci_low", "ci_high", "rank"])
        for metric in metric_names:
            rows = []
            for s in summaries:
                m = s["metrics"].get(metric, {})
                mean = m.get("mean", float("nan"))
                rows.append((s["model"], s["n_tiles"], mean, m.get("ci_low"), m.get("ci_high")))
            higher = metric != "count_abs_error"
            rows.sort(key=lambda r: (float("inf") if not np.isfinite(r[2]) else -r[2]) if higher else r[2])
            for rank, row in enumerate(rows, start=1):
                w.writerow([row[0], row[1], metric, row[2], row[3], row[4], rank])
    return path
