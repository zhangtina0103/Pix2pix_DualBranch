"""Compare extended metrics across multiple HEMIT models."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from hemit_eval.extended_metrics import METRIC_SPECS, compute_extended_metrics, write_extended_metrics
from hemit_eval.statistics import compare_paired


def load_manifest(manifest_path: str | Path) -> list[dict[str, str]]:
    path = Path(manifest_path).expanduser().resolve()
    rows: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("model") and row.get("srcdir"):
                rows.append({"model": row["model"].strip(), "srcdir": row["srcdir"].strip()})
    if not rows:
        raise ValueError(f"Empty manifest: {path}")
    return rows


def resolve_reference_model(manifest: list[dict[str, str]], reference: str | None) -> str:
    names = [m["model"] for m in manifest]
    if not names:
        raise ValueError("Empty manifest")
    if reference is None:
        return names[0]
    if reference in names:
        return reference
    matches = [n for n in names if reference in n or n.startswith(reference)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        # prefer exact pix2pix over cyclegan etc.
        for prefer in (f"{reference}_resnet9", "pix2pix_resnet9", reference):
            for n in matches:
                if prefer in n and "orion" not in n:
                    return n
        raise ValueError(f"Ambiguous reference_model '{reference}': {matches}")
    raise ValueError(f"reference_model '{reference}' not in manifest: {names}")


def run_model_comparison(
    manifest: list[dict[str, str]], outdir: str | Path, *,
    reference_model: str | None = None, bootstrap_resamples: int = 10000,
    seed: int = 42, use_lpips: bool = True,
) -> dict[str, Any]:
    outdir = Path(outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    model_results: dict[str, dict[str, Any]] = {}
    per_tile_by_model: dict[str, list[dict[str, Any]]] = {}

    for entry in manifest:
        name = entry["model"]
        per_tile, summary = compute_extended_metrics(
            entry["srcdir"], use_lpips=use_lpips,
            bootstrap_resamples=bootstrap_resamples, seed=seed,
        )
        model_results[name] = summary
        per_tile_by_model[name] = per_tile
        write_extended_metrics(outdir / name, per_tile, summary)

    leaderboard_path = outdir / "leaderboard_extended_metrics.csv"
    with leaderboard_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "scope", "channel", "metric", "mean", "std", "ci_low", "ci_high"])
        for name, summary in model_results.items():
            for channel, metrics in summary["channels"].items():
                for metric, stats in metrics.items():
                    w.writerow([name, "channel", channel, metric, f"{stats['mean']:.6f}", f"{stats['std']:.6f}",
                                f"{stats['ci_low']:.6f}", f"{stats['ci_high']:.6f}"])
            for metric, stats in summary["average"].items():
                w.writerow([name, "average", "mean", metric, f"{stats['mean']:.6f}", f"{stats['std']:.6f}",
                            f"{stats['ci_low']:.6f}", f"{stats['ci_high']:.6f}"])

    ref = resolve_reference_model(manifest, reference_model)
    ref_by_file = {row["file_name"]: row for row in per_tile_by_model[ref]}
    paired_rows: list[dict[str, Any]] = []
    for name, rows in per_tile_by_model.items():
        if name == ref:
            continue
        common = sorted(set(ref_by_file) & {r["file_name"] for r in rows})
        cur_by_file = {r["file_name"]: r for r in rows}
        for metric in METRIC_SPECS:
            col = f"average_{metric}"
            comp = compare_paired(
                np.array([ref_by_file[k][col] for k in common], dtype=np.float64),
                np.array([cur_by_file[k][col] for k in common], dtype=np.float64),
                metric_name=metric, n_resamples=bootstrap_resamples, random_state=seed,
            )
            paired_rows.append({"reference": ref, "model": name, **comp})

    paired_path = outdir / "paired_comparison_vs_reference.csv"
    if paired_rows:
        with paired_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(paired_rows[0].keys()))
            w.writeheader()
            w.writerows(paired_rows)

    report = {
        "reference_model": ref, "models": list(model_results.keys()),
        "leaderboard_csv": str(leaderboard_path),
        "paired_comparison_csv": str(paired_path) if paired_rows else None,
        "summaries": model_results,
    }
    (outdir / "comparison_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
