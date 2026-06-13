#!/usr/bin/env python3
"""
Extended HEMIT metrics on test.py outputs.

Computes per-tile SSIM, Pearson, PSNR, MAE, MSE, RMSE, R², LPIPS (if installed)
with mean/std/median and 95% bootstrap CIs.

Examples:
  python scripts/run_hemit_extended_metrics.py \\
    --srcdir results/hemit_fm_cross_attn_scratch/test_80/images \\
    --outdir eval/hemit/cross_attn_ep80

  python scripts/run_hemit_extended_metrics.py \\
    --manifest eval/hemit/manifest.csv \\
    --outdir eval/hemit/comparison \\
    --reference-model pix2pix
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hemit.eval.compare_models import load_manifest, run_model_comparison
from hemit.eval.extended_metrics import compute_extended_metrics, write_extended_metrics


def main() -> None:
    p = argparse.ArgumentParser(description="HEMIT extended metrics with bootstrap CIs.")
    p.add_argument("--srcdir", type=str, default=None, help="Single model test images dir")
    p.add_argument("--manifest", type=str, default=None,
                   help="CSV with columns model,srcdir for multi-model comparison")
    p.add_argument("--outdir", type=str, required=True)
    p.add_argument("--reference-model", type=str, default=None,
                   help="Reference model for paired comparison (default: first in manifest)")
    p.add_argument("--no-lpips", action="store_true", help="Skip LPIPS even if lpips is installed")
    p.add_argument("--bootstrap-resamples", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    outdir = Path(args.outdir).expanduser().resolve()
    use_lpips = not args.no_lpips

    if args.manifest:
        manifest = load_manifest(args.manifest)
        report = run_model_comparison(
            manifest,
            outdir,
            reference_model=args.reference_model,
            bootstrap_resamples=args.bootstrap_resamples,
            seed=args.seed,
            use_lpips=use_lpips,
        )
        print(json.dumps({
            "mode": "comparison",
            "outdir": str(outdir),
            "reference_model": report["reference_model"],
            "leaderboard_csv": report["leaderboard_csv"],
            "paired_comparison_csv": report["paired_comparison_csv"],
        }, indent=2))
        return

    if not args.srcdir:
        p.error("Provide --srcdir for single model or --manifest for comparison")

    per_tile, summary = compute_extended_metrics(
        args.srcdir,
        use_lpips=use_lpips,
        bootstrap_resamples=args.bootstrap_resamples,
        seed=args.seed,
    )
    paths = write_extended_metrics(outdir, per_tile, summary)
    print(json.dumps({
        "mode": "single",
        "n_tiles": summary["n_tiles"],
        "lpips_available": summary["lpips_available"],
        **{k: str(v) for k, v in paths.items()},
        "average_pearson_mean": summary["average"]["pearson"]["mean"],
        "average_ssim_mean": summary["average"]["ssim"]["mean"],
    }, indent=2))


if __name__ == "__main__":
    main()
