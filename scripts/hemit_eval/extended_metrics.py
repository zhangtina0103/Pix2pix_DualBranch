"""Extended per-tile image metrics for HEMIT."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity as ssim

from hemit_eval.image_io import HEMIT_CHANNELS, list_fake_files, load_pair, resolve_image_dir
from hemit_eval.statistics import summarize_values

METRIC_SPECS: dict[str, dict[str, Any]] = {
    "ssim": {"higher_is_better": True},
    "pearson": {"higher_is_better": True},
    "spearman": {"higher_is_better": True},
    "psnr": {"higher_is_better": True},
    "mae": {"higher_is_better": False},
    "mse": {"higher_is_better": False},
    "rmse": {"higher_is_better": False},
    "nmse": {"higher_is_better": False},
    "r2": {"higher_is_better": True},
    "lpips": {"higher_is_better": False},
}


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    if a.size < 2:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt(np.sum(a * a) * np.sum(b * b))
    return float("nan") if denom <= 1e-12 else float(np.sum(a * b) / denom)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    if a.size < 2:
        return float("nan")
    try:
        from scipy.stats import spearmanr
        return float(spearmanr(a, b).correlation)
    except Exception:
        ra = np.argsort(np.argsort(a)).astype(np.float64)
        rb = np.argsort(np.argsort(b)).astype(np.float64)
        return _pearson(ra, rb)


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.astype(np.float64).ravel()
    y_pred = y_pred.astype(np.float64).ravel()
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float("nan") if ss_tot <= 1e-12 else float(1.0 - np.sum((y_true - y_pred) ** 2) / ss_tot)


def _nmse(real_ch: np.ndarray, fake_ch: np.ndarray) -> float:
    err = fake_ch.astype(np.float64) - real_ch.astype(np.float64)
    var = float(np.var(real_ch.astype(np.float64)))
    return float("nan") if var <= 1e-12 else float(np.mean(err ** 2) / var)


def _channel_metrics(real_ch: np.ndarray, fake_ch: np.ndarray) -> dict[str, float]:
    real_ch = real_ch.astype(np.float64)
    fake_ch = fake_ch.astype(np.float64)
    err = fake_ch - real_ch
    return {
        "ssim": float(ssim(real_ch, fake_ch, data_range=255.0)),
        "pearson": _pearson(real_ch, fake_ch),
        "spearman": _spearman(real_ch, fake_ch),
        "psnr": float(peak_signal_noise_ratio(real_ch, fake_ch, data_range=255.0)),
        "mae": float(np.mean(np.abs(err))),
        "mse": float(np.mean(err ** 2)),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "nmse": _nmse(real_ch, fake_ch),
        "r2": _r2(real_ch, fake_ch),
    }


class _LpipsScorer:
    def __init__(self) -> None:
        self._model = None
        self.available = False
        try:
            import lpips  # type: ignore
            import torch
            self._torch = torch
            self._model = lpips.LPIPS(net="alex")
            self._model.eval()
            self.available = True
        except Exception:
            pass

    def score(self, real_ch: np.ndarray, fake_ch: np.ndarray) -> float:
        if not self.available or self._model is None:
            return float("nan")
        t = self._torch
        def _tensor(ch: np.ndarray):
            x = np.clip(ch, 0, 255).astype(np.float32) / 255.0
            x = np.stack([x, x, x], axis=0) * 2.0 - 1.0
            return t.from_numpy(x).unsqueeze(0)
        with t.no_grad():
            return float(self._model(_tensor(fake_ch), _tensor(real_ch)).item())


def compute_extended_metrics(
    srcdir: str | Path, *, use_lpips: bool = True,
    bootstrap_resamples: int = 10000, seed: int = 42,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    image_dir = resolve_image_dir(srcdir)
    lpips_scorer = _LpipsScorer() if use_lpips else None
    per_tile: list[dict[str, Any]] = []

    for fake_path in list_fake_files(image_dir):
        real, fake, base = load_pair(fake_path)
        row: dict[str, Any] = {"file_name": base}
        channel_metrics: dict[str, dict[str, float]] = {}
        for i, ch in enumerate(HEMIT_CHANNELS):
            m = _channel_metrics(real[..., i], fake[..., i])
            m["lpips"] = lpips_scorer.score(real[..., i], fake[..., i]) if lpips_scorer and lpips_scorer.available else float("nan")
            channel_metrics[ch] = m
            for metric, val in m.items():
                row[f"{ch}_{metric}"] = val
        for metric in METRIC_SPECS:
            row[f"average_{metric}"] = float(np.nanmean([channel_metrics[ch][metric] for ch in HEMIT_CHANNELS]))
        per_tile.append(row)

    summary: dict[str, Any] = {"channels": {}, "average": {}}
    for key in list(HEMIT_CHANNELS) + ["average"]:
        block = summary["average"] if key == "average" else summary["channels"].setdefault(key, {})
        for metric, spec in METRIC_SPECS.items():
            col = f"{key}_{metric}" if key != "average" else f"average_{metric}"
            block[metric] = summarize_values(
                np.array([row[col] for row in per_tile], dtype=np.float64),
                n_resamples=bootstrap_resamples, random_state=seed,
                higher_is_better=spec["higher_is_better"],
            )

    summary["lpips_available"] = bool(lpips_scorer and lpips_scorer.available)
    summary["n_tiles"] = len(per_tile)
    summary["image_dir"] = str(image_dir)
    return per_tile, summary


def write_extended_metrics(
    outdir: str | Path, per_tile: list[dict[str, Any]], summary: dict[str, Any],
) -> dict[str, Path]:
    outdir = Path(outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    per_tile_path = outdir / "extended_metrics_per_tile.csv"
    if per_tile:
        with per_tile_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(per_tile[0].keys()))
            w.writeheader()
            w.writerows(per_tile)
    summary_path = outdir / "extended_metrics_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    flat_path = outdir / "extended_metrics_summary.csv"
    with flat_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scope", "channel", "metric", "n", "mean", "std", "median", "ci_low", "ci_high", "higher_is_better"])
        for channel, metrics in summary["channels"].items():
            for metric, stats in metrics.items():
                w.writerow(["channel", channel, metric, stats["n"], f"{stats['mean']:.6f}", f"{stats['std']:.6f}",
                            f"{stats['median']:.6f}", f"{stats['ci_low']:.6f}", f"{stats['ci_high']:.6f}", stats["higher_is_better"]])
        for metric, stats in summary["average"].items():
            w.writerow(["average", "mean", metric, stats["n"], f"{stats['mean']:.6f}", f"{stats['std']:.6f}",
                        f"{stats['median']:.6f}", f"{stats['ci_low']:.6f}", f"{stats['ci_high']:.6f}", stats["higher_is_better"]])
    return {"per_tile_csv": per_tile_path, "summary_json": summary_path, "summary_csv": flat_path}
