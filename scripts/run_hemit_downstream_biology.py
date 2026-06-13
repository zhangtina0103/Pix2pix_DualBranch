#!/usr/bin/env python3
"""
HEMIT downstream biology (Fig. 6 style): marker-positive cell proportions.

Segments nuclei from Hoechst, counts CD3e/Pan-CK positive nuclei on real vs generated,
reports MAE ratio, cell counts, and diagnostic plots.

Examples:
  python scripts/run_hemit_downstream_biology.py \\
    --srcdir results/hemit_fm_cross_attn_scratch/test_80/images \\
    --outdir eval/hemit/cross_attn_ep80/downstream \\
    --model-name cross_attn

  python scripts/run_hemit_downstream_biology.py \\
    --manifest eval/hemit/manifest.csv \\
    --outdir eval/hemit/downstream_comparison
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.hemit_eval_path import setup_hemit_eval_path

setup_hemit_eval_path()

from hemit_eval.compare_models import load_manifest
from hemit_eval.downstream_biology import (
    compute_downstream_biology,
    plot_downstream_model_comparison,
    plot_downstream_single_model,
    write_downstream_leaderboard,
    write_downstream_results,
)


def main() -> None:
    p = argparse.ArgumentParser(description="HEMIT downstream biology evaluation.")
    p.add_argument("--srcdir", type=str, default=None)
    p.add_argument("--manifest", type=str, default=None,
                   help="CSV with model,srcdir for multi-model comparison plots")
    p.add_argument("--outdir", type=str, required=True)
    p.add_argument("--model-name", type=str, default="model")
    p.add_argument("--bootstrap-resamples", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--plots", action="store_true",
                   help="Generate figures (default: CSV only)")
    args = p.parse_args()

    outdir = Path(args.outdir).expanduser().resolve()

    if args.manifest:
        manifest = load_manifest(args.manifest)
        summaries = []
        for entry in manifest:
            model_out = outdir / entry["model"]
            per_tile, summary = compute_downstream_biology(
                entry["srcdir"],
                model_name=entry["model"],
                bootstrap_resamples=args.bootstrap_resamples,
                seed=args.seed,
            )
            write_downstream_results(model_out, per_tile, summary)
            if args.plots:
                plot_downstream_single_model(per_tile, summary, model_out, model_name=entry["model"])
            summaries.append(summary)
        leaderboard = write_downstream_leaderboard(summaries, outdir)
        cmp_plot = None
        if args.plots and len(summaries) > 1:
            cmp_plot = str(plot_downstream_model_comparison(summaries, outdir))
        print(json.dumps({
            "mode": "comparison",
            "outdir": str(outdir),
            "models": [s["model"] for s in summaries],
            "leaderboard_csv": str(leaderboard),
            "mae_ratio_comparison_plot": cmp_plot,
        }, indent=2))
        return

    if not args.srcdir:
        p.error("Provide --srcdir or --manifest")

    per_tile, summary = compute_downstream_biology(
        args.srcdir,
        model_name=args.model_name,
        bootstrap_resamples=args.bootstrap_resamples,
        seed=args.seed,
    )
    paths = write_downstream_results(outdir, per_tile, summary)
    plot_paths = []
    if args.plots:
        plot_paths = [str(p) for p in plot_downstream_single_model(
            per_tile, summary, outdir, model_name=args.model_name
        )]
    print(json.dumps({
        "mode": "single",
        "outdir": str(outdir),
        "mae_ratio_cd3": summary["markers"]["cd3"]["mae_ratio"],
        "mae_ratio_panck": summary["markers"]["panck"]["mae_ratio"],
        **{k: str(v) for k, v in paths.items()},
        "plots": plot_paths,
    }, indent=2))


if __name__ == "__main__":
    main()
