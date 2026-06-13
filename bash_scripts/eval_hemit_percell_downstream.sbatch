#!/bin/bash
# Phase-1 per-cell downstream: intensity Pearson + CD3+ / enriched subsets.
#
#   export HEMIT_MANIFEST=eval/hemit/manifest.csv
#   export PERCELL_OUTDIR=eval/hemit/percell_downstream_enriched
#   export PERCELL_CD3_ENRICH_TOP_FRAC=0.10   # top 10% by mean real CD3 (~95 tiles)
#   sbatch bash_scripts/eval_hemit_percell_downstream.sbatch
#
# Coverage first: sbatch bash_scripts/analyze_cd3_coverage.sbatch
# Plots: PLOTS=1 sbatch ...
#SBATCH --job-name=hemit_percell
#SBATCH --output=logs/hemit_percell_%j.out
#SBATCH --error=logs/hemit_percell_%j.err
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/bash_scripts/_common.sh"
mkdir -p logs eval/hemit

source /home/zhangtin/miniforge3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate "${CONDA_ENV:-pix2pix_cuda}" 2>/dev/null || true

export HEMIT_MANIFEST="${HEMIT_MANIFEST:-eval/hemit/manifest.csv}"
export PERCELL_OUTDIR="${PERCELL_OUTDIR:-eval/hemit/percell_downstream}"
export PERCELL_CD3_PCT="${PERCELL_CD3_PCT:-60}"
export PERCELL_CD3_ENRICH_TOP_FRAC="${PERCELL_CD3_ENRICH_TOP_FRAC:-0.10}"
PLOT_FLAG=()
[[ "${PLOTS:-0}" == "1" ]] && PLOT_FLAG=(--plots)

[[ -f "${HEMIT_MANIFEST}" ]] || { echo "ERROR: missing ${HEMIT_MANIFEST}" >&2; exit 1; }

python scripts/run_hemit_percell_downstream.py \
  --manifest "${HEMIT_MANIFEST}" \
  --outdir "${PERCELL_OUTDIR}" \
  --cd3-marker-percentile "${PERCELL_CD3_PCT}" \
  --cd3-enrichment-top-frac "${PERCELL_CD3_ENRICH_TOP_FRAC}" \
  "${PLOT_FLAG[@]}"

echo "Done → ${PERCELL_OUTDIR}/percell_leaderboard.csv"
