#!/bin/bash
# HEMIT robustness: extended metrics + downstream biology CSVs (plots optional).
#
# Multi-model (after submit_hemit_full_robustness.sbatch or manual manifest):
#   export HEMIT_MANIFEST=eval/hemit/manifest.csv
#   sbatch bash_scripts/eval_hemit_robustness.sbatch
#
# Single model:
#   export TRAIN_NAME=hemit_fm_cross_attn_scratch
#   unset HEMIT_MANIFEST
#   export TEST_EPOCH=80
#   sbatch bash_scripts/eval_hemit_robustness.sbatch
#
# Plots later:  PLOTS=1 sbatch bash_scripts/eval_hemit_robustness.sbatch
#SBATCH --job-name=hemit_robust
#SBATCH --output=logs/hemit_robustness_%j.out
#SBATCH --error=logs/hemit_robustness_%j.err
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/bash_scripts/_common.sh"
mkdir -p logs eval/hemit

source /home/zhangtin/miniforge3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate "${CONDA_ENV:-pix2pix_cuda}" 2>/dev/null || true

export TEST_EPOCH="${TEST_EPOCH:-80}"
export REFERENCE_MODEL="${REFERENCE_MODEL:-pix2pix}"
export ROBUST_OUTDIR="${ROBUST_OUTDIR:-eval/hemit/robustness_comparison}"
PLOT_FLAG=()
[[ "${PLOTS:-0}" == "1" ]] && PLOT_FLAG=(--plots)

if [[ -n "${TRAIN_NAME:-}" && -z "${HEMIT_MANIFEST:-}" ]]; then
  SRCDIR="results/${TRAIN_NAME}/test_${TEST_EPOCH}/images"
  [[ -d "${SRCDIR}" ]] || { echo "ERROR: missing ${SRCDIR} — run test.py first" >&2; exit 1; }
  MODEL_LABEL="${MODEL_LABEL:-${TRAIN_NAME}}"
  OUT="${ROBUST_OUTDIR:-eval/hemit/${MODEL_LABEL}_ep${TEST_EPOCH}}"
  python scripts/run_hemit_robustness_eval.py \
    --srcdir "${SRCDIR}" \
    --outdir "${OUT}" \
    --model-name "${MODEL_LABEL}" \
    "${PLOT_FLAG[@]}"
elif [[ -n "${HEMIT_MANIFEST:-}" ]]; then
  [[ -f "${HEMIT_MANIFEST}" ]] || { echo "ERROR: missing ${HEMIT_MANIFEST}" >&2; exit 1; }
  if [[ "${REBUILD_MANIFEST:-0}" == "1" ]]; then
    python scripts/build_hemit_robustness_manifest.py --require-images --test-epoch "${TEST_EPOCH}"
  fi
  python scripts/run_hemit_robustness_eval.py \
    --manifest "${HEMIT_MANIFEST}" \
    --outdir "${ROBUST_OUTDIR}" \
    --reference-model "${REFERENCE_MODEL}" \
    "${PLOT_FLAG[@]}"
else
  echo "ERROR: set TRAIN_NAME (single) or HEMIT_MANIFEST=eval/hemit/manifest.csv (multi)" >&2
  exit 1
fi

echo "Done. See ${ROBUST_OUTDIR:-eval/hemit}/ for extended + downstream CSVs"
