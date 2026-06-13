#!/usr/bin/env python3
"""Merge extended + downstream robustness CSVs into paper-ready tables (mean ± std, 95% CI)."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


EXTENDED_METRICS = ("ssim", "pearson", "spearman", "psnr", "lpips")
DOWNSTREAM_MARKERS = ("cd3", "panck", "total")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _fmt(mean: float, std: float, lo: float, hi: float, digits: int = 3) -> str:
    return f"{mean:.{digits}f} ± {std:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def _f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def load_extended_from_leaderboard(path: Path) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    """model → channel → metric → {mean, std, ci_low, ci_high, n}."""
    out: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for row in _read_csv(path):
        if row.get("scope") != "channel":
            continue
        model = row["model"]
        ch, metric = row["channel"], row["metric"]
        out.setdefault(model, {}).setdefault(ch, {})[metric] = {
            "mean": _f(row, "mean"),
            "std": _f(row, "std"),
            "ci_low": _f(row, "ci_low"),
            "ci_high": _f(row, "ci_high"),
        }
    return out


def load_extended_from_summaries(extended_dir: Path) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    out: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for summary in sorted(extended_dir.glob("*/extended_metrics_summary.csv")):
        model = summary.parent.name
        for row in _read_csv(summary):
            if row.get("scope") != "channel":
                continue
            ch, metric = row["channel"], row["metric"]
            out.setdefault(model, {}).setdefault(ch, {})[metric] = {
                "mean": _f(row, "mean"),
                "std": _f(row, "std"),
                "ci_low": _f(row, "ci_low"),
                "ci_high": _f(row, "ci_high"),
                "n": float(row.get("n") or 0),
            }
    return out


def load_downstream(downstream_dir: Path) -> dict[str, dict[str, dict[str, Any]]]:
    """model → marker → fields."""
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for summary in sorted(downstream_dir.glob("*/downstream_summary.csv")):
        model = summary.parent.name
        for row in _read_csv(summary):
            marker = row["marker"]
            out.setdefault(model, {})[marker] = {
                "mae_ratio": _f(row, "mae_ratio"),
                "p_real_mean": _f(row, "p_real_mean"),
                "p_real_std": _f(row, "p_real_std"),
                "p_real_ci_low": _f(row, "p_real_ci_low"),
                "p_real_ci_high": _f(row, "p_real_ci_high"),
                "p_gen_mean": _f(row, "p_gen_mean"),
                "p_gen_std": _f(row, "p_gen_std"),
                "p_gen_ci_low": _f(row, "p_gen_ci_low"),
                "p_gen_ci_high": _f(row, "p_gen_ci_high"),
            }
    leaderboard = downstream_dir / "downstream_leaderboard.csv"
    if leaderboard.is_file():
        for row in _read_csv(leaderboard):
            m = row["model"]
            for mk in DOWNSTREAM_MARKERS:
                key = f"{mk}_mae_ratio"
                if key in row and m in out and mk in out[m]:
                    out[m][mk]["mae_ratio_leaderboard"] = float(row[key])
    return out


def write_extended_table(
    models: list[str],
    data: dict[str, dict[str, dict[str, dict[str, float]]]],
    out_path: Path,
    *,
    family: str,
) -> None:
    rows: list[dict[str, str]] = []
    for model in models:
        if model not in data:
            continue
        for ch in ("cd3", "panck", "dapi"):
            if ch not in data[model]:
                continue
            for metric in EXTENDED_METRICS:
                if metric not in data[model][ch]:
                    continue
                s = data[model][ch][metric]
                rows.append({
                    "family": family,
                    "model": model,
                    "channel": ch,
                    "metric": metric,
                    "mean": f"{s['mean']:.6f}",
                    "std": f"{s['std']:.6f}",
                    "ci_low": f"{s['ci_low']:.6f}",
                    "ci_high": f"{s['ci_high']:.6f}",
                    "paper_format": _fmt(s["mean"], s["std"], s["ci_low"], s["ci_high"]),
                })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [
            "family", "model", "channel", "metric", "mean", "std", "ci_low", "ci_high", "paper_format",
        ])
        w.writeheader()
        w.writerows(rows)


def write_downstream_table(
    models: list[str],
    data: dict[str, dict[str, dict[str, Any]]],
    out_path: Path,
    *,
    family: str,
) -> None:
    rows: list[dict[str, str]] = []
    for model in models:
        if model not in data:
            continue
        for marker in DOWNSTREAM_MARKERS:
            if marker not in data[model]:
                continue
            d = data[model][marker]
            rows.append({
                "family": family,
                "model": model,
                "marker": marker,
                "p_real_mean": f"{d['p_real_mean']:.6f}",
                "p_real_std": f"{d['p_real_std']:.6f}",
                "p_real_ci": f"[{d['p_real_ci_low']:.4f}, {d['p_real_ci_high']:.4f}]",
                "p_gen_mean": f"{d['p_gen_mean']:.6f}",
                "p_gen_std": f"{d['p_gen_std']:.6f}",
                "p_gen_ci": f"[{d['p_gen_ci_low']:.4f}, {d['p_gen_ci_high']:.4f}]",
                "p_gen_paper_format": _fmt(
                    d["p_gen_mean"], d["p_gen_std"], d["p_gen_ci_low"], d["p_gen_ci_high"], 4,
                ),
                "mae_ratio": f"{d['mae_ratio']:.6f}",
            })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        w.writeheader()
        w.writerows(rows)


def write_compact_cd3_table(
    hemit_models: list[str],
    hemit_ext: dict[str, dict[str, dict[str, dict[str, float]]]],
    diffusion_models: list[str],
    diff_ext: dict[str, dict[str, dict[str, dict[str, float]]]],
    hemit_ds: dict[str, dict[str, dict[str, Any]]],
    out_path: Path,
) -> None:
    """One-row-per-model summary for CD3 Pearson + LPIPS + downstream p_gen."""
    rows: list[dict[str, str]] = []

    def _add(model: str, family: str, ext: dict) -> None:
        if model not in ext or "cd3" not in ext[model]:
            return
        pearson = ext[model]["cd3"].get("pearson", {})
        lpips = ext[model]["cd3"].get("lpips", {})
        row = {
            "family": family,
            "model": model,
            "cd3_pearson_mean": f"{pearson.get('mean', float('nan')):.4f}",
            "cd3_pearson_std": f"{pearson.get('std', float('nan')):.4f}",
            "cd3_pearson_ci": (
                f"[{pearson.get('ci_low', float('nan')):.4f}, {pearson.get('ci_high', float('nan')):.4f}]"
                if pearson else ""
            ),
            "cd3_lpips_mean": f"{lpips.get('mean', float('nan')):.4f}",
            "cd3_lpips_ci": (
                f"[{lpips.get('ci_low', float('nan')):.4f}, {lpips.get('ci_high', float('nan')):.4f}]"
                if lpips else ""
            ),
        }
        if model in hemit_ds and "cd3" in hemit_ds[model]:
            d = hemit_ds[model]["cd3"]
            row["cd3_p_gen_mean"] = f"{d['p_gen_mean']:.4f}"
            row["cd3_p_gen_ci"] = f"[{d['p_gen_ci_low']:.4f}, {d['p_gen_ci_high']:.4f}]"
        rows.append(row)

    for m in hemit_models:
        _add(m, "hemit", hemit_ext)
    for m in diffusion_models:
        _add(m, "diffusion", diff_ext)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser(description="Export paper-ready metric tables with CI/std.")
    p.add_argument("--hemit-extended", type=str, required=True, help="eval/.../extended/ dir or leaderboard.csv")
    p.add_argument("--hemit-downstream", type=str, default=None)
    p.add_argument("--diffusion-extended", type=str, default=None)
    p.add_argument("--diffusion-downstream", type=str, default=None)
    p.add_argument("--outdir", type=str, required=True)
    p.add_argument(
        "--hemit-models",
        type=str,
        default="asp,cut,cyclegan,cross_attn,pix2pix,vanilla_fm",
        help="Comma-separated model order for main table",
    )
    p.add_argument(
        "--diffusion-models",
        type=str,
        default="dvst",
        help="Comma-separated diffusion models (use dvst not dvst_zero_shot if duplicate)",
    )
    args = p.parse_args()

    outdir = Path(args.outdir).expanduser().resolve()
    hemit_path = Path(args.hemit_extended).expanduser().resolve()

    if hemit_path.is_file():
        hemit_ext = load_extended_from_leaderboard(hemit_path)
    else:
        hemit_ext = load_extended_from_summaries(hemit_path)

    hemit_models = [m.strip() for m in args.hemit_models.split(",") if m.strip()]
    diffusion_models = [m.strip() for m in args.diffusion_models.split(",") if m.strip()]

    hemit_ds: dict[str, dict[str, dict[str, Any]]] = {}
    if args.hemit_downstream:
        hemit_ds = load_downstream(Path(args.hemit_downstream).expanduser().resolve())

    diff_ext: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    if args.diffusion_extended:
        diff_path = Path(args.diffusion_extended).expanduser().resolve()
        if diff_path.is_file():
            diff_ext = load_extended_from_leaderboard(diff_path)
        else:
            diff_ext = load_extended_from_summaries(diff_path)
            lb = diff_path / "leaderboard_extended_metrics.csv"
            if lb.is_file():
                diff_ext.update(load_extended_from_leaderboard(lb))

    write_extended_table(hemit_models, hemit_ext, outdir / "table1_hemit_extended.csv", family="hemit")
    if diff_ext:
        write_extended_table(diffusion_models, diff_ext, outdir / "table2_diffusion_extended.csv", family="diffusion")
    if hemit_ds:
        write_downstream_table(hemit_models, hemit_ds, outdir / "table3_hemit_downstream.csv", family="hemit")
    write_compact_cd3_table(
        hemit_models, hemit_ext, diffusion_models, diff_ext, hemit_ds, outdir / "table4_cd3_summary.csv",
    )

    print(f"Wrote tables → {outdir}")
    for f in sorted(outdir.glob("table*.csv")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
