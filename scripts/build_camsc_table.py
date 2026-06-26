#!/usr/bin/env python3
"""
CaMSC quantitative leaderboard table (the single results table for the case study).

Computes per-tile image metrics for both channels (Hoechst, WT1) + their average,
pooled across ALL k-folds per model, then emits:

  camsc_leaderboard_full.csv   every scope x metric (mean, std) per model
  camsc_leaderboard.tex        booktabs LaTeX: one tabular per scope (WT1, Hoechst,
                               Average), best value per metric column bolded

Self-contained: reads results/<...>/test_<epoch>/images and scores directly, so it
does not depend on having run eval_camsc_kfold.py first. Channels: Hoechst(0), WT1(1).

Example
-------
  python scripts/build_camsc_table.py \
    --results-root results --epoch 110 --k-folds 5 \
    --out-dir figures/camsc/table --with-lpips
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.hemit_eval.extended_metrics import _channel_metrics, _LpipsScorer  # noqa: E402

CHANNELS = (("hoechst", 0), ("wt1", 1))

# Order: Ours first, then weakest -> strongest baselines (matches figure ordering).
DEFAULT_MODELS = [
    ("Ours", "fm_cross_attn_ft"),
    ("Pix2Pix", "pix2pix_ft"),
    ("CycleGAN", "cyclegan_ft"),
    ("CUT", "cut_ft"),
    ("ASP", "asp_ft"),
]

# metric -> (LaTeX header, decimals, higher_is_better)
METRICS = {
    "ssim": ("SSIM $\\uparrow$", 3, True),
    "pearson": ("Pearson $r$ $\\uparrow$", 3, True),
    "spearman": ("Spearman $\\rho$ $\\uparrow$", 3, True),
    "psnr": ("PSNR $\\uparrow$", 2, True),
    "mae": ("MAE $\\downarrow$", 2, False),
    "rmse": ("RMSE $\\downarrow$", 2, False),
    "lpips": ("LPIPS $\\downarrow$", 3, False),
}
SCOPE_TITLES = {"wt1": "WT1 (sparse marker)", "hoechst": "Hoechst (dense marker)",
                "average": "Channel average"}


def _to_uint8(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img, dtype=np.float64)
    if arr.min() < 0:
        arr = (arr + 1.0) / 2.0 * 255.0
    elif arr.max() <= 1.0:
        arr = arr * 255.0
    return np.clip(arr, 0, 255)


def discover(results_root: Path, epoch: int, k_folds: int,
             models: list[tuple[str, str]]) -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = {}
    for label, key in models:
        dirs = []
        for fold in range(k_folds):
            matches = sorted(results_root.glob(f"camsc_bf_{key}_fold{fold}*/test_{epoch}/images"))
            matches = [m for m in matches if any(m.glob("*_fake_B.tif"))]
            if matches:
                dirs.append(matches[0])
        if dirs:
            out[label] = dirs
            print(f"  [ok] {label:<10} {len(dirs)} fold(s)")
        else:
            print(f"  [MISS] {label:<10} camsc_bf_{key}_fold*/test_{epoch}/images", file=sys.stderr)
    return out


def collect(model_dirs: dict[str, list[Path]], lpips_scorer) -> pd.DataFrame:
    rows = []
    for label, dirs in model_dirs.items():
        for d in dirs:
            for fp in sorted(d.glob("*_fake_B.tif")):
                rp = Path(str(fp).replace("_fake_B.tif", "_real_B.tif"))
                if not rp.is_file():
                    continue
                fake = _to_uint8(np.asarray(Image.open(fp)))
                real = _to_uint8(np.asarray(Image.open(rp)))
                row = {"model": label, "tile": fp.stem.replace("_fake_B", "")}
                per_ch = {}
                for name, idx in CHANNELS:
                    m = _channel_metrics(real[..., idx], fake[..., idx])
                    if lpips_scorer is not None and lpips_scorer.available:
                        m["lpips"] = lpips_scorer.score(real[..., idx], fake[..., idx])
                    else:
                        m["lpips"] = float("nan")
                    per_ch[name] = m
                    for met, val in m.items():
                        row[f"{name}_{met}"] = val
                for met in METRICS:
                    row[f"average_{met}"] = float(np.nanmean([per_ch[n][met] for n, _ in CHANNELS]))
                rows.append(row)
    return pd.DataFrame(rows)


def _order(labels: list[str]) -> list[str]:
    pref = [lbl for lbl, _ in DEFAULT_MODELS]
    ordered = [m for m in pref if m in labels]
    return ordered + [m for m in labels if m not in ordered]


def aggregate(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    recs = []
    for m in _order(list(dict.fromkeys(df["model"]))):
        sub = df[df["model"] == m]
        rec = {"model": m, "n_tiles": len(sub)}
        for scope in ("hoechst", "wt1", "average"):
            for met in metrics:
                col = f"{scope}_{met}"
                if col in sub:
                    rec[f"{col}_mean"] = float(sub[col].mean())
                    rec[f"{col}_std"] = float(sub[col].std())
        recs.append(rec)
    return pd.DataFrame(recs)


def _fmt(mean: float, std: float, dec: int) -> str:
    if not np.isfinite(mean):
        return "n/a"
    return f"{mean:.{dec}f}$\\pm${std:.{dec}f}"


def _best_model(agg: pd.DataFrame, col_mean: str, higher: bool) -> str | None:
    vals = agg[["model", col_mean]].dropna()
    if vals.empty:
        return None
    idx = vals[col_mean].idxmax() if higher else vals[col_mean].idxmin()
    return agg.loc[idx, "model"]


def write_latex(agg: pd.DataFrame, metrics: list[str], scopes: list[str], out_path: Path) -> None:
    lines = ["% Auto-generated by scripts/build_camsc_table.py",
             "% Requires \\usepackage{booktabs}"]
    for scope in scopes:
        present = [m for m in metrics if f"{scope}_{m}_mean" in agg.columns]
        if not present:
            continue
        ncol = 1 + len(present)
        lines += [
            "\\begin{table}[t]", "\\centering",
            f"\\caption{{CaMSC {SCOPE_TITLES.get(scope, scope)}: pooled 5-fold test metrics "
            f"(mean$\\pm$std). Best per column in \\textbf{{bold}}.}}",
            f"\\label{{tab:camsc_{scope}}}",
            "\\begin{tabular}{l" + "c" * len(present) + "}",
            "\\toprule",
            "Model & " + " & ".join(METRICS[m][0] for m in present) + " \\\\",
            "\\midrule",
        ]
        best = {m: _best_model(agg, f"{scope}_{m}_mean", METRICS[m][2]) for m in present}
        for _, r in agg.iterrows():
            cells = [r["model"]]
            for m in present:
                dec = METRICS[m][1]
                s = _fmt(r.get(f"{scope}_{m}_mean", float("nan")),
                         r.get(f"{scope}_{m}_std", float("nan")), dec)
                if r["model"] == best[m] and s != "n/a":
                    s = f"\\textbf{{{s}}}"
                cells.append(s)
            lines.append(" & ".join(cells) + " \\\\")
        lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    out_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="CaMSC leaderboard table (LaTeX + CSV)")
    p.add_argument("--results-root", default="results")
    p.add_argument("--epoch", type=int, default=110)
    p.add_argument("--k-folds", type=int, default=5)
    p.add_argument("--model", action="append", default=[], help="Label=key (overrides defaults)")
    p.add_argument("--metrics", default="ssim,pearson,spearman,psnr,mae,rmse")
    p.add_argument("--scopes", default="wt1,hoechst,average")
    p.add_argument("--with-lpips", action="store_true", help="compute LPIPS (slower; needs lpips pkg)")
    p.add_argument("--out-dir", default="figures/camsc/table")
    args = p.parse_args()

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    if args.with_lpips and "lpips" not in metrics:
        metrics.append("lpips")
    scopes = [s.strip() for s in args.scopes.split(",") if s.strip()]
    bad = [m for m in metrics if m not in METRICS]
    if bad:
        raise SystemExit(f"Unknown metric(s): {bad}. Valid: {list(METRICS)}")

    results_root = Path(args.results_root).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.model:
        models = []
        for spec in args.model:
            lbl, key = spec.split("=", 1)
            models.append((lbl.strip(), key.strip()))
    else:
        models = DEFAULT_MODELS

    print(f"Discovering models under {results_root} (epoch {args.epoch}, {args.k_folds} folds):")
    model_dirs = discover(results_root, args.epoch, args.k_folds, models)
    if not model_dirs:
        raise SystemExit("No model dirs found.")

    lpips_scorer = _LpipsScorer() if args.with_lpips else None
    if lpips_scorer and lpips_scorer.available:
        print(f"LPIPS: on ({lpips_scorer.device})")

    print("Scoring tiles (both channels, pooled across folds)...")
    df = collect(model_dirs, lpips_scorer)
    if df.empty:
        raise SystemExit("No tiles scored.")
    df.to_csv(out_dir / "camsc_per_tile_all.csv", index=False)

    agg = aggregate(df, metrics)
    agg.to_csv(out_dir / "camsc_leaderboard_full.csv", index=False)
    print(f"Wrote {out_dir / 'camsc_leaderboard_full.csv'}")
    write_latex(agg, metrics, scopes, out_dir / "camsc_leaderboard.tex")

    print("\n=== WT1 (key marker) summary ===")
    for _, r in agg.iterrows():
        print(f"  {r['model']:<10} SSIM={r.get('wt1_ssim_mean', float('nan')):.3f}  "
              f"Pearson={r.get('wt1_pearson_mean', float('nan')):.3f}  "
              f"PSNR={r.get('wt1_psnr_mean', float('nan')):.2f}")


if __name__ == "__main__":
    main()
