#!/bin/bash
# CaMSC extended WT1 downstream (per-cell + enriched subsets + nucleus SSIM).
#
#   export TEST_EPOCH=110
#   sbatch bash_scripts/eval_camsc_wt1_downstream.sbatch
#
# Outputs: figures/camsc/downstream_wt1/ (or CAMSC_WT1_DOWNSTREAM_OUT)
#SBATCH --job-name=camsc_wt1_ds
#SBATCH --output=logs/camsc_wt1_downstream_%j.out
#SBATCH --error=logs/camsc_wt1_downstream_%j.err
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/bash_scripts/_common.sh"
mkdir -p logs

source /home/zhangtin/miniforge3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate "${CONDA_ENV:-pix2pix_cuda}" 2>/dev/null || true

export TEST_EPOCH="${TEST_EPOCH:-110}"
export CAMSC_WT1_DOWNSTREAM_OUT="${CAMSC_WT1_DOWNSTREAM_OUT:-figures/camsc/downstream_wt1}"
export WT1_ENRICH_TOP_FRAC="${WT1_ENRICH_TOP_FRAC:-0.10}"

python scripts/camsc_wt1_percell_downstream.py \
  --results-root "${REPO_ROOT}/results" \
  --epoch "${TEST_EPOCH}" \
  --k-folds 5 \
  --wt1-enrichment-top-frac "${WT1_ENRICH_TOP_FRAC}" \
  --out-dir "${CAMSC_WT1_DOWNSTREAM_OUT}"

echo "Done → ${CAMSC_WT1_DOWNSTREAM_OUT}/wt1_subset_leaderboard.csv"
