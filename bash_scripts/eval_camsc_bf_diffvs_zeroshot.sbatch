#!/bin/bash
# DiffVS zero-shot on CaMSC test (HEMIT-trained checkpoint, no CaMSC training).
#
#   export CAMSC_KFOLD_ROOT=~/orcd/scratch/camsc/datasets/camsc_bf_kfold_aug
#   export CAMSC_ENABLE_AUG=1
#   sbatch bash_scripts/eval_camsc_bf_diffvs_zeroshot.sbatch
#
# Requires: DiffVS repo at DIFFVS_ROOT, HEMIT stage-2 ckpt (see DIFFVS_CHECKPOINT_DIR).
#SBATCH --job-name=camsc_dvs_zs
#SBATCH --output=logs/camsc_diffvs_zeroshot_%j.out
#SBATCH --error=logs/camsc_diffvs_zeroshot_%j.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/bash_scripts/_common.sh"
mkdir -p logs

source /home/zhangtin/miniforge3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-pix2pix_cuda}"
source "${REPO_ROOT}/bash_scripts/_cuda.sh"

export CAMSC_KFOLD_ROOT="${CAMSC_KFOLD_ROOT:-${HOME}/orcd/scratch/camsc/datasets/camsc_bf_kfold_aug}"
export CAMSC_ENABLE_AUG="${CAMSC_ENABLE_AUG:-1}"
export CAMSC_EVAL_FOLDS="${CAMSC_EVAL_FOLDS:-}"

bash scripts/run_camsc_diffvs_zeroshot.sh

# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/camsc_scratch_env.sh"
export CAMSC_MODEL=diffvs_zeroshot

EVAL_EPOCH="${CAMSC_DIFFVS_EVAL_EPOCH:-80}"
NAME_SUFFIX="$(camsc_eval_name_suffix 0)"
OUT="${CAMSC_KFOLD_ROOT}/eval/diffvs_zeroshot"
mkdir -p "${OUT}"

python scripts/eval_camsc_kfold.py \
  --kfold-root "${CAMSC_KFOLD_ROOT}" \
  --results-root "${REPO_ROOT}/results" \
  --name-prefix "$(camsc_results_name_prefix)" \
  --name-suffix "${NAME_SUFFIX}" \
  --epoch "${EVAL_EPOCH}" \
  --k-folds "${CAMSC_KFOLDS}" \
  --out-dir "${OUT}"

echo "diffvs_zeroshot metrics: ${OUT}/kfold_aggregate.json"
