#!/bin/bash
# Count CD3+ test tiles at multiple pseudo-label percentiles (power analysis).
# CPU-only; run before lowering YOLO_CD3_MARKER_PERCENTILE or PERCELL_CD3_PCT.
#
#   export CD3_COVERAGE_SRCDIR=results/hemit_fm_cross_attn_scratch_512/test_80/images
#   export CD3_COVERAGE_OUTDIR=eval/hemit/cd3_coverage
#   sbatch bash_scripts/analyze_cd3_coverage.sbatch
#
#SBATCH --job-name=cd3_coverage
#SBATCH --output=logs/cd3_coverage_%j.out
#SBATCH --error=logs/cd3_coverage_%j.err
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/bash_scripts/_common.sh"
mkdir -p logs eval/hemit/cd3_coverage

source /home/zhangtin/miniforge3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate "${CONDA_ENV:-pix2pix_cuda}" 2>/dev/null || true

CD3_COVERAGE_SRCDIR="${CD3_COVERAGE_SRCDIR:-results/hemit_fm_cross_attn_scratch_512/test_80/images}"
CD3_COVERAGE_OUTDIR="${CD3_COVERAGE_OUTDIR:-eval/hemit/cd3_coverage}"
CD3_COVERAGE_PCTS="${CD3_COVERAGE_PCTS:-50,55,60,65,70}"

[[ -d "${CD3_COVERAGE_SRCDIR}" ]] || {
  echo "ERROR: missing ${CD3_COVERAGE_SRCDIR} — run test inference @80 first" >&2
  exit 1
}

python scripts/analyze_cd3_positive_coverage.py \
  --srcdir "${CD3_COVERAGE_SRCDIR}" \
  --outdir "${CD3_COVERAGE_OUTDIR}" \
  --percentiles "${CD3_COVERAGE_PCTS}"

echo "Done → ${CD3_COVERAGE_OUTDIR}/cd3_positive_coverage.csv"
