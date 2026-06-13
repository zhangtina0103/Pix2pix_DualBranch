"""Per-cell downstream: intensity correlation, co-expression, CD3+ tile subset."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from skimage.measure import label, regionprops

from hemit_eval.downstream_biology import (
    DOWNSTREAM_MARKERS,
    _marker_positive_nuclei_count,
    segment_nuclei,
)
from hemit_eval.image_io import list_fake_files, load_pair, resolve_image_dir
from hemit_eval.statistics import summarize_values

MARKER_CHANNELS = {"cd3": 1, "panck": 2}


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size < 3 or b.size < 3:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt(np.sum(a * a) * np.sum(b * b))
    return float("nan") if denom <= 1e-12 else float(np.sum(a * b) / denom)


def _per_cell_intensities(real: np.ndarray, fake: np.ndarray) -> list[dict[str, float]]:
    nuclei = segment_nuclei(real[..., 0])
    labeled = label(nuclei)
    if labeled.max() == 0:
        return []
    cells: list[dict[str, float]] = []
    for prop in regionprops(labeled):
        mask = labeled == prop.label
        row: dict[str, float] = {"nucleus_id": float(prop.label)}
        for marker, ch in MARKER_CHANNELS.items():
            row[f"{marker}_real"] = float(real[mask, ch].mean())
            row[f"{marker}_gen"] = float(fake[mask, ch].mean())
        cells.append(row)
    return cells


def _tile_metrics(
    real: np.ndarray,
    fake: np.ndarray,
    *,
    cd3_marker_percentile: float = 60,
) -> dict[str, Any]:
    cells = _per_cell_intensities(real, fake)
    nuclei = segment_nuclei(real[..., 0])
    cd3_count_real = _marker_positive_nuclei_count(
        real[..., 1], nuclei, marker_percentile=cd3_marker_percentile,
    )
    cd3_count_gen = _marker_positive_nuclei_count(
        fake[..., 1], nuclei, marker_percentile=cd3_marker_percentile,
    )

    out: dict[str, Any] = {
        "n_nuclei": len(cells),
        "cd3_count_real": cd3_count_real,
        "cd3_count_gen": cd3_count_gen,
        "cd3_count_abs_err": abs(cd3_count_gen - cd3_count_real),
        "is_cd3_positive_tile": cd3_count_real > 0,
    }
    if len(cells) < 3:
        out.update({
            "cd3_percell_pearson": float("nan"),
            "panck_percell_pearson": float("nan"),
            "coexp_real_cd3_panck": float("nan"),
            "coexp_gen_cd3_panck": float("nan"),
            "coexp_abs_err": float("nan"),
        })
        return out

    cd3_r = np.array([c["cd3_real"] for c in cells], dtype=np.float64)
    cd3_g = np.array([c["cd3_gen"] for c in cells], dtype=np.float64)
    pan_r = np.array([c["panck_real"] for c in cells], dtype=np.float64)
    pan_g = np.array([c["panck_gen"] for c in cells], dtype=np.float64)

    coexp_r = _pearson(cd3_r, pan_r)
    coexp_g = _pearson(cd3_g, pan_g)
    out.update({
        "cd3_percell_pearson": _pearson(cd3_r, cd3_g),
        "panck_percell_pearson": _pearson(pan_r, pan_g),
        "coexp_real_cd3_panck": coexp_r,
        "coexp_gen_cd3_panck": coexp_g,
        "coexp_abs_err": abs(coexp_r - coexp_g) if np.isfinite(coexp_r) and np.isfinite(coexp_g) else float("nan"),
    })
    return out


def compute_percell_downstream(
    srcdir: str | Path, *, model_name: str = "model",
    cd3_marker_percentile: float = 60,
    bootstrap_resamples: int = 10000, seed: int = 42,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    image_dir = resolve_image_dir(srcdir)
    per_tile: list[dict[str, Any]] = []
    for fake_path in list_fake_files(image_dir):
        real, fake, base = load_pair(fake_path)
        row = {"file_name": base, "model": model_name, **_tile_metrics(
            real, fake, cd3_marker_percentile=cd3_marker_percentile,
        )}
        per_tile.append(row)

    def _summ(col: str, *, higher: bool = True) -> dict[str, Any]:
        return summarize_values(
            np.array([r[col] for r in per_tile], dtype=np.float64),
            n_resamples=bootstrap_resamples, random_state=seed, higher_is_better=higher,
        )

    cd3pos = [r for r in per_tile if r.get("is_cd3_positive_tile")]
    summary: dict[str, Any] = {
        "model": model_name,
        "n_tiles": len(per_tile),
        "n_cd3_positive_tiles": len(cd3pos),
        "cd3_marker_percentile": cd3_marker_percentile,
        "all_tiles": {
            "cd3_percell_pearson": _summ("cd3_percell_pearson"),
            "panck_percell_pearson": _summ("panck_percell_pearson"),
            "coexp_abs_err": _summ("coexp_abs_err", higher=False),
            "cd3_count_abs_err": _summ("cd3_count_abs_err", higher=False),
        },
        "cd3_positive_tiles": {
            "cd3_percell_pearson": summarize_values(
                np.array([r["cd3_percell_pearson"] for r in cd3pos], dtype=np.float64),
                n_resamples=bootstrap_resamples, random_state=seed,
            ) if cd3pos else {"n": 0, "mean": float("nan"), "std": float("nan"),
                              "ci_low": float("nan"), "ci_high": float("nan")},
            "cd3_count_abs_err": summarize_values(
                np.array([r["cd3_count_abs_err"] for r in cd3pos], dtype=np.float64),
                n_resamples=bootstrap_resamples, random_state=seed, higher_is_better=False,
            ) if cd3pos else {"n": 0, "mean": float("nan"), "std": float("nan"),
                              "ci_low": float("nan"), "ci_high": float("nan")},
            "panck_percell_pearson": summarize_values(
                np.array([r["panck_percell_pearson"] for r in cd3pos], dtype=np.float64),
                n_resamples=bootstrap_resamples, random_state=seed,
            ) if cd3pos else {"n": 0, "mean": float("nan"), "std": float("nan"),
                              "ci_low": float("nan"), "ci_high": float("nan")},
        },
        "image_dir": str(image_dir),
    }
    return per_tile, summary


def write_percell_results(
    outdir: str | Path, per_tile: list[dict[str, Any]], summary: dict[str, Any],
) -> dict[str, Path]:
    outdir = Path(outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    per_tile_path = outdir / "percell_per_tile.csv"
    if per_tile:
        with per_tile_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(per_tile[0].keys()))
            w.writeheader()
            w.writerows(per_tile)
    json_path = outdir / "percell_summary.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    flat_path = outdir / "percell_summary.csv"
    with flat_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "scope", "metric", "mean", "std", "ci_low", "ci_high", "n"])
        model = summary["model"]
        for scope in ("all_tiles", "cd3_positive_tiles"):
            for metric, block in summary[scope].items():
                w.writerow([
                    model, scope, metric,
                    f"{block['mean']:.6f}", f"{block.get('std', 0):.6f}",
                    f"{block['ci_low']:.6f}", f"{block['ci_high']:.6f}", block.get("n", summary["n_tiles"]),
                ])
    return {"per_tile_csv": per_tile_path, "summary_json": json_path, "summary_csv": flat_path}


def write_percell_leaderboard(summaries: list[dict[str, Any]], outdir: str | Path) -> Path:
    outdir = Path(outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "percell_leaderboard.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "model", "n_tiles", "n_cd3_positive_tiles",
            "cd3_percell_pearson", "cd3_percell_ci_low", "cd3_percell_ci_high",
            "panck_percell_pearson", "panck_percell_ci_low", "panck_percell_ci_high",
            "coexp_abs_err", "cd3pos_cd3_percell_pearson", "cd3pos_count_abs_err",
        ])
        for s in summaries:
            allm = s["all_tiles"]
            sub = s["cd3_positive_tiles"]
            w.writerow([
                s["model"], s["n_tiles"], s["n_cd3_positive_tiles"],
                f"{allm['cd3_percell_pearson']['mean']:.6f}",
                f"{allm['cd3_percell_pearson']['ci_low']:.6f}",
                f"{allm['cd3_percell_pearson']['ci_high']:.6f}",
                f"{allm['panck_percell_pearson']['mean']:.6f}",
                f"{allm['panck_percell_pearson']['ci_low']:.6f}",
                f"{allm['panck_percell_pearson']['ci_high']:.6f}",
                f"{allm['coexp_abs_err']['mean']:.6f}",
                f"{sub['cd3_percell_pearson']['mean']:.6f}",
                f"{sub['cd3_count_abs_err']['mean']:.6f}",
            ])
    return path


def plot_percell_leaderboard(summaries: list[dict[str, Any]], outdir: str | Path) -> list[Path]:
    import matplotlib.pyplot as plt

    outdir = Path(outdir).expanduser().resolve()
    plot_dir = outdir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    models = [s["model"] for s in summaries]
    x = np.arange(len(models))
    saved: list[Path] = []

    for key, title, ylabel in (
        ("cd3_percell_pearson", "Per-cell CD3 intensity Pearson (all tiles)", "Pearson r"),
        ("panck_percell_pearson", "Per-cell Pan-CK intensity Pearson (all tiles)", "Pearson r"),
    ):
        means = [s["all_tiles"][key]["mean"] for s in summaries]
        yerr = [
            [m - s["all_tiles"][key]["ci_low"] for m, s in zip(means, summaries)],
            [s["all_tiles"][key]["ci_high"] - m for m, s in zip(means, summaries)],
        ]
        fig, ax = plt.subplots(figsize=(max(6, len(models) * 1.1), 4))
        ax.bar(x, means, yerr=yerr, capsize=3, color="steelblue")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=20, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        fig.tight_layout()
        p = plot_dir / f"{key}_comparison.png"
        fig.savefig(p, dpi=160)
        plt.close(fig)
        saved.append(p)

    # CD3+ tile subset
    fig, ax = plt.subplots(figsize=(max(6, len(models) * 1.1), 4))
    means = [s["cd3_positive_tiles"]["cd3_percell_pearson"]["mean"] for s in summaries]
    ax.bar(x, means, color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylabel("Pearson r")
    ax.set_title(f"Per-cell CD3 Pearson (CD3+ tiles only, n≈{summaries[0]['n_cd3_positive_tiles']})")
    fig.tight_layout()
    p = plot_dir / "cd3pos_percell_pearson_comparison.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    saved.append(p)

    return saved
