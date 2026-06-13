"""Downstream biology: marker-positive cell proportions (HEMIT Fig. 6 style)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import closing, disk, remove_small_objects

from hemit_eval.image_io import list_fake_files, load_pair, resolve_image_dir
from hemit_eval.statistics import summarize_values

DOWNSTREAM_MARKERS = ("cd3", "panck")
DOWNSTREAM_LABELS = ("CD3e", "Pan-CK")


def segment_nuclei(dapi: np.ndarray, *, min_area: int = 36) -> np.ndarray:
    img = np.clip(dapi, 0, 255).astype(np.float32)
    if img.max() <= 0:
        return np.zeros_like(img, dtype=bool)
    mask = img >= threshold_otsu(img)
    mask = closing(mask, disk(2))
    return remove_small_objects(mask, min_size=min_area)


def _marker_positive_mask(marker: np.ndarray, nuclei_mask: np.ndarray) -> np.ndarray:
    labeled = label(nuclei_mask)
    if labeled.max() == 0:
        return np.zeros_like(nuclei_mask, dtype=bool)
    means = [p.mean_intensity for p in regionprops(labeled, intensity_image=marker)]
    if not means:
        return np.zeros_like(nuclei_mask, dtype=bool)
    thr = max(threshold_otsu(marker[marker > 0]) if np.any(marker > 0) else 0.0, float(np.percentile(means, 60)))
    positive = np.zeros_like(nuclei_mask, dtype=bool)
    for prop in regionprops(labeled, intensity_image=marker):
        if prop.mean_intensity >= thr:
            positive[labeled == prop.label] = True
    return positive


def compute_tile_proportions(real: np.ndarray, fake: np.ndarray) -> dict[str, Any]:
    nuclei = segment_nuclei(real[..., 0])
    n_total = int(label(nuclei).max())
    out: dict[str, Any] = {"n_nuclei": n_total}
    if n_total == 0:
        for marker in DOWNSTREAM_MARKERS:
            out[f"{marker}_p_real"] = float("nan")
            out[f"{marker}_p_gen"] = float("nan")
            out[f"{marker}_count_real"] = 0
            out[f"{marker}_count_gen"] = 0
        out["total_p_real"] = out["total_p_gen"] = float("nan")
        out["total_count_real"] = out["total_count_gen"] = 0
        return out
    idx = {"cd3": 1, "panck": 2}
    total_real = total_gen = 0
    for marker, ch_i in idx.items():
        cr = int(np.sum(_marker_positive_mask(real[..., ch_i], nuclei)))
        cg = int(np.sum(_marker_positive_mask(fake[..., ch_i], nuclei)))
        out[f"{marker}_count_real"], out[f"{marker}_count_gen"] = cr, cg
        out[f"{marker}_p_real"], out[f"{marker}_p_gen"] = cr / n_total, cg / n_total
        total_real += cr
        total_gen += cg
    out["total_count_real"], out["total_count_gen"] = total_real, total_gen
    out["total_p_real"], out["total_p_gen"] = total_real / n_total, total_gen / n_total
    return out


def _mae_ratio(p_real: np.ndarray, p_gen: np.ndarray, eps: float) -> float:
    mask = np.isfinite(p_real) & np.isfinite(p_gen)
    if not np.any(mask):
        return float("nan")
    pr, pg = p_real[mask], p_gen[mask]
    return float(np.mean(np.abs(pr - pg) / np.maximum(np.abs(pr), eps)))


def compute_downstream_biology(
    srcdir: str | Path, *, model_name: str = "model", eps: float = 1e-3,
    bootstrap_resamples: int = 10000, seed: int = 42,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    image_dir = resolve_image_dir(srcdir)
    per_tile: list[dict[str, Any]] = []
    for fake_path in list_fake_files(image_dir):
        real, fake, base = load_pair(fake_path)
        row = {"file_name": base, "model": model_name, **compute_tile_proportions(real, fake)}
        per_tile.append(row)
    summary: dict[str, Any] = {"model": model_name, "markers": {}, "eps": eps}
    for marker in list(DOWNSTREAM_MARKERS) + ["total"]:
        p_real = np.array([r[f"{marker}_p_real"] for r in per_tile], dtype=np.float64)
        p_gen = np.array([r[f"{marker}_p_gen"] for r in per_tile], dtype=np.float64)
        summary["markers"][marker] = {
            "mae_ratio": _mae_ratio(p_real, p_gen, eps),
            "p_real": summarize_values(p_real, n_resamples=bootstrap_resamples, random_state=seed),
            "p_gen": summarize_values(p_gen, n_resamples=bootstrap_resamples, random_state=seed),
            "count_real": summarize_values(np.array([r[f"{marker}_count_real"] for r in per_tile], dtype=np.float64),
                                          n_resamples=bootstrap_resamples, random_state=seed),
            "count_gen": summarize_values(np.array([r[f"{marker}_count_gen"] for r in per_tile], dtype=np.float64),
                                         n_resamples=bootstrap_resamples, random_state=seed),
        }
    summary["n_tiles"] = len(per_tile)
    summary["image_dir"] = str(image_dir)
    return per_tile, summary


def write_downstream_results(outdir: str | Path, per_tile: list[dict[str, Any]], summary: dict[str, Any]) -> dict[str, Path]:
    outdir = Path(outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    per_tile_path = outdir / "downstream_per_tile.csv"
    if per_tile:
        with per_tile_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(per_tile[0].keys()))
            w.writeheader()
            w.writerows(per_tile)
    summary_path = outdir / "downstream_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    flat_path = outdir / "downstream_summary.csv"
    with flat_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "marker", "mae_ratio", "p_real_mean", "p_real_std", "p_real_ci_low", "p_real_ci_high",
                    "p_gen_mean", "p_gen_std", "p_gen_ci_low", "p_gen_ci_high", "count_real_mean", "count_real_std",
                    "count_gen_mean", "count_gen_std"])
        model = summary.get("model", "model")
        for marker, block in summary["markers"].items():
            w.writerow([model, marker, f"{block['mae_ratio']:.6f}",
                        f"{block['p_real']['mean']:.6f}", f"{block['p_real']['std']:.6f}",
                        f"{block['p_real']['ci_low']:.6f}", f"{block['p_real']['ci_high']:.6f}",
                        f"{block['p_gen']['mean']:.6f}", f"{block['p_gen']['std']:.6f}",
                        f"{block['p_gen']['ci_low']:.6f}", f"{block['p_gen']['ci_high']:.6f}",
                        f"{block['count_real']['mean']:.6f}", f"{block['count_real']['std']:.6f}",
                        f"{block['count_gen']['mean']:.6f}", f"{block['count_gen']['std']:.6f}"])
    return {"per_tile_csv": per_tile_path, "summary_json": summary_path, "summary_csv": flat_path}


def write_downstream_leaderboard(summaries: list[dict[str, Any]], outdir: str | Path) -> Path:
    outdir = Path(outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "downstream_leaderboard.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "cd3_mae_ratio", "panck_mae_ratio", "total_mae_ratio"])
        for s in summaries:
            m = s["markers"]
            w.writerow([s["model"], f"{m['cd3']['mae_ratio']:.6f}", f"{m['panck']['mae_ratio']:.6f}", f"{m['total']['mae_ratio']:.6f}"])
    return path


def plot_downstream_single_model(per_tile, summary, outdir, *, model_name: str) -> list[Path]:
    import matplotlib.pyplot as plt
    outdir = Path(outdir).expanduser().resolve()
    plot_dir = outdir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(list(DOWNSTREAM_LABELS), [summary["markers"][m]["mae_ratio"] for m in DOWNSTREAM_MARKERS])
    ax.set_ylabel("MAE ratio")
    ax.set_title(f"{model_name}: proportion error")
    fig.tight_layout()
    p = plot_dir / "mae_ratio_bar.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    saved.append(p)
    return saved


def plot_downstream_model_comparison(summaries: list[dict[str, Any]], outdir: str | Path) -> Path:
    import matplotlib.pyplot as plt
    outdir = Path(outdir).expanduser().resolve()
    models = [s["model"] for s in summaries]
    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(max(6, len(models) * 1.2), 4))
    ax.bar(x - 0.175, [s["markers"]["cd3"]["mae_ratio"] for s in summaries], 0.35, label="CD3e")
    ax.bar(x + 0.175, [s["markers"]["panck"]["mae_ratio"] for s in summaries], 0.35, label="Pan-CK")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylabel("MAE ratio")
    ax.legend()
    fig.tight_layout()
    path = outdir / "mae_ratio_comparison.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path
