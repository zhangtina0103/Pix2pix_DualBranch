#!/usr/bin/env python3
"""
Phase-1 per-cell downstream on HEMIT test TIFFs.

Metrics:
  - Per-nucleus CD3 / Pan-CK intensity Pearson (real vs generated)
  - CD3–Pan-CK co-expression preservation (|r_real - r_gen| per tile)
  - CD3-positive tile subset (tiles with ≥1 CD3+ nucleus)

Examples:
  python scripts/run_hemit_percell_downstream.py \\
    --manifest eval/hemit/manifest.csv \\
    --outdir eval/hemit/percell_downstream

  python scripts/run_hemit_percell_downstream.py \\
    --srcdir results/hemit_fm_cross_attn_scratch/test_80/images \\
    --outdir eval/hemit/cross_attn/percell \\
    --model-name cross_attn \\
    --plots
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
from hemit_eval.percell_downstream import (
    compute_percell_downstream,
    plot_percell_leaderboard,
    write_percell_leaderboard,
    write_percell_results,
)


def main() -> None:
    p = argparse.ArgumentParser(description="HEMIT per-cell downstream (Phase 1).")
    p.add_argument("--srcdir", type=str, default=None)
    p.add_argument("--manifest", type=str, default=None)
    p.add_argument("--outdir", type=str, required=True)
    p.add_argument("--model-name", type=str, default="model")
    p.add_argument("--cd3-marker-percentile", type=float, default=60,
                   help="Lower (e.g. 50) for more CD3+ tiles — run analyze_cd3_positive_coverage.py first")
    p.add_argument("--bootstrap-resamples", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--plots", action="store_true")
    args = p.parse_args()

    outdir = Path(args.outdir).expanduser().resolve()

    if args.manifest:
        manifest = load_manifest(args.manifest)
        summaries = []
        for entry in manifest:
            model_out = outdir / entry["model"]
            per_tile, summary = compute_percell_downstream(
                entry["srcdir"],
                model_name=entry["model"],
                cd3_marker_percentile=args.cd3_marker_percentile,
                bootstrap_resamples=args.bootstrap_resamples,
                seed=args.seed,
            )
            write_percell_results(model_out, per_tile, summary)
            summaries.append(summary)
        leaderboard = write_percell_leaderboard(summaries, outdir)
        plot_paths = [str(p) for p in plot_percell_leaderboard(summaries, outdir)] if args.plots else []
        print(json.dumps({
            "mode": "comparison",
            "outdir": str(outdir),
            "models": [s["model"] for s in summaries],
            "leaderboard_csv": str(leaderboard),
            "plots": plot_paths,
        }, indent=2))
        return

    if not args.srcdir:
        p.error("Provide --srcdir or --manifest")

    per_tile, summary = compute_percell_downstream(
        args.srcdir,
        model_name=args.model_name,
        cd3_marker_percentile=args.cd3_marker_percentile,
        bootstrap_resamples=args.bootstrap_resamples,
        seed=args.seed,
    )
    paths = write_percell_results(outdir, per_tile, summary)
    print(json.dumps({
        "mode": "single",
        "model": args.model_name,
        "cd3_percell_pearson": summary["all_tiles"]["cd3_percell_pearson"]["mean"],
        "panck_percell_pearson": summary["all_tiles"]["panck_percell_pearson"]["mean"],
        **{k: str(v) for k, v in paths.items()},
    }, indent=2))


if __name__ == "__main__":
    main()
