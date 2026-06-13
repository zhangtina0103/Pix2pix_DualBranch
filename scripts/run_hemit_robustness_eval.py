#!/usr/bin/env python3
"""Run extended metrics + downstream biology for one or more HEMIT models."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Full HEMIT robustness eval pipeline.")
    p.add_argument("--srcdir", type=str, default=None)
    p.add_argument("--manifest", type=str, default=None)
    p.add_argument("--outdir", type=str, required=True)
    p.add_argument("--model-name", type=str, default=None,
                   help="Label for downstream plots (single-model mode)")
    p.add_argument("--reference-model", type=str, default="pix2pix")
    p.add_argument("--skip-extended", action="store_true")
    p.add_argument("--skip-downstream", action="store_true")
    p.add_argument("--no-lpips", action="store_true")
    p.add_argument("--plots", action="store_true",
                   help="Generate downstream figures (default: CSV only)")
    args = p.parse_args()

    py = sys.executable
    outdir = Path(args.outdir).expanduser().resolve()

    if not args.skip_extended:
        cmd = [py, str(ROOT / "scripts/run_hemit_extended_metrics.py"), "--outdir", str(outdir / "extended")]
        if args.manifest:
            cmd += ["--manifest", args.manifest, "--reference-model", args.reference_model]
        else:
            if not args.srcdir:
                p.error("Provide --srcdir or --manifest")
            cmd += ["--srcdir", args.srcdir]
        if args.no_lpips:
            cmd.append("--no-lpips")
        _run(cmd)

    if not args.skip_downstream:
        cmd = [py, str(ROOT / "scripts/run_hemit_downstream_biology.py"), "--outdir", str(outdir / "downstream")]
        if args.manifest:
            cmd += ["--manifest", args.manifest]
        else:
            cmd += ["--srcdir", args.srcdir, "--model-name", args.model_name or "model"]
        if args.plots:
            cmd.append("--plots")
        _run(cmd)

    print(f"\nHEMIT robustness eval complete → {outdir}")


if __name__ == "__main__":
    main()
