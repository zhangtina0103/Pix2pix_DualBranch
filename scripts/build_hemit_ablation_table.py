#!/usr/bin/env python3
"""Build HEMIT ablation table (Markdown + LaTeX) from score.csv or extended_metrics_summary.csv."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean
from typing import Dict, List, Tuple

MetricKey = str
Row = Tuple[str, Dict[MetricKey, float]]


def _finite_float(x: str) -> float | None:
    if x is None or x == "" or str(x).lower() == "nan":
        return None
    v = float(x)
    return v if math.isfinite(v) else None


def load_tile_score_csv(path: Path) -> Dict[MetricKey, float]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows in {path}")

    cols = [
        "average_ssim", "average_pearson", "average_psnr",
        "dapi_ssim", "cd3_ssim", "panck_ssim",
        "dapi_pearson", "cd3_pearson", "panck_pearson",
        "dapi_psnr", "cd3_psnr", "panck_psnr",
    ]
    out: Dict[MetricKey, float] = {"n": float(len(rows))}
    for col in cols:
        vals = [_finite_float(r[col]) for r in rows if col in r]
        vals = [v for v in vals if v is not None]
        if vals:
            out[col] = mean(vals)
    return out


def load_extended_summary(path: Path) -> Dict[MetricKey, float]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    out: Dict[MetricKey, float] = {}
    if rows and "n" in rows[0]:
        out["n"] = float(rows[0]["n"])
    for r in rows:
        m = r["metric"]
        val = _finite_float(r["mean"])
        if val is None:
            continue
        if r["scope"] == "average":
            out[m] = val
        elif r["scope"] == "channel":
            out[f"{r['channel']}_{m}"] = val
    return out


def load_metrics(path: Path) -> Dict[MetricKey, float]:
    path = path.expanduser().resolve()
    if path.name.endswith("extended_metrics_summary.csv"):
        return load_extended_summary(path)
    return load_tile_score_csv(path)


DISPLAY_COLS: List[Tuple[MetricKey, str, bool]] = [
    ("average_ssim", "SSIM", True),
    ("average_pearson", "Pearson", True),
    ("average_psnr", "PSNR", True),
    ("cd3_pearson", "CD3 $r$", True),
    ("dapi_pearson", "DAPI $r$", True),
    ("panck_pearson", "panCK $r$", True),
]


def _get(d: Dict[MetricKey, float], key: MetricKey) -> float | None:
    if key in d:
        return d[key]
    alt = key.replace("average_", "")
    return d.get(alt)


def bold_best(rows: List[Row]) -> Dict[MetricKey, str | None]:
    best_name: Dict[MetricKey, str | None] = {}
    for key, _, higher in DISPLAY_COLS:
        scored = []
        for name, d in rows:
            v = _get(d, key)
            if v is not None:
                scored.append((v, name))
        if not scored:
            best_name[key] = None
            continue
        best_val = max(v for v, _ in scored) if higher else min(v for v, _ in scored)
        winners = [n for v, n in scored if abs(v - best_val) < 1e-9]
        best_name[key] = winners[0] if len(winners) == 1 else None
    return best_name


def format_cell(v: float | None, bold: bool) -> str:
    if v is None:
        return "—"
    s = f"{v:.3f}"
    return f"**{s}**" if bold else s


def to_markdown(rows: List[Row], caption: str) -> str:
    best = bold_best(rows)
    lines = [caption, ""]
    hdr = "| Variant | " + " | ".join(c[1] for c in DISPLAY_COLS) + " |"
    sep = "|---|" + "|".join([":---:"] * len(DISPLAY_COLS)) + "|"
    lines.extend([hdr, sep])
    for name, d in rows:
        cells = []
        for key, _, _ in DISPLAY_COLS:
            v = _get(d, key)
            winner = best.get(key)
            cells.append(format_cell(v, winner == name and winner is not None))
        lines.append("| " + name + " | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def to_latex(rows: List[Row], caption: str, label: str) -> str:
    best = bold_best(rows)
    ncol = len(DISPLAY_COLS)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\small",
        r"\begin{tabular}{l" + "r" * ncol + "}",
        r"\toprule",
        "Variant & " + " & ".join(f"{c[1]}$\\uparrow$" if c[2] else c[1] for c in DISPLAY_COLS) + r" \\",
        r"\midrule",
    ]
    for name, d in rows:
        tex_name = name.replace("+ ", "+\\,")
        cells = []
        for key, _, _ in DISPLAY_COLS:
            v = _get(d, key)
            winner = best.get(key)
            if v is None:
                cells.append("---")
            elif winner == name and winner is not None:
                cells.append(f"\\textbf{{{v:.3f}}}")
            else:
                cells.append(f"{v:.3f}")
        lines.append(tex_name + " & " + " & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description="Build HEMIT ablation table from metric CSVs.")
    p.add_argument(
        "--config",
        type=str,
        help="JSON list of {name, path} entries. Default: built-in HEMIT ablation set.",
    )
    p.add_argument("--outdir", type=str, default="eval/hemit/ablation")
    p.add_argument(
        "--caption",
        type=str,
        default=(
            "Component ablation on HEMIT test set ($n=945$ tiles). "
            "All FM rows share joint perceptual loss ($\\lambda_{\\mathrm{perc}}=0.1$) "
            "and fair channel weights $(1,1,1)$ unless noted."
        ),
    )
    p.add_argument("--label", type=str, default="tab:ablation")
    args = p.parse_args()

    if args.config:
        with open(args.config) as f:
            spec = json.load(f)
        rows: List[Row] = [(e["name"], load_metrics(Path(e["path"]))) for e in spec]
    else:
        root = Path(__file__).resolve().parents[1]
        dl = Path.home() / "Downloads"
        rows = [
            ("Vanilla FM (+ joint perc)", load_metrics(dl / "extended/vanilla_fm/extended_metrics_summary.csv")),
            ("+ Cross-attention (ours)", load_metrics(dl / "extended/cross_attn/extended_metrics_summary.csv")),
            ("+ Focal $\\gamma{=}1$ (global)", load_metrics(dl / "score-4.csv")),
            ("+ Focal $\\gamma{=}0.75$ + vel", load_metrics(dl / "score-5.csv")),
            ("+ Focal CD3-only @50", load_metrics(dl / "score-6.csv")),
            ("+ Focal CD3-only @80", load_metrics(dl / "score-7.csv")),
            ("+ CD3 combo @60 (1,2,1)", load_metrics(dl / "score-2.csv")),
        ]

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    md = to_markdown(rows, "# HEMIT ablation table\n")
    tex = to_latex(rows, args.caption, args.label)

    (outdir / "ablation_table.md").write_text(md)
    (outdir / "ablation_table.tex").write_text(tex)
    print(f"Wrote {outdir / 'ablation_table.md'}")
    print(f"Wrote {outdir / 'ablation_table.tex'}")
    print()
    print(md)


if __name__ == "__main__":
    main()
