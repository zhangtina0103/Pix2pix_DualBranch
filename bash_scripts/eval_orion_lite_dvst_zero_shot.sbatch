#!/bin/bash
# D-VST zero-shot on Orion-Lite test @ 512² (pretrained HE2mIHC.ckpt — no Orion train).
#
#   sbatch bash_scripts/eval_orion_lite_dvst_zero_shot.sbatch
#
# Separate table in paper (~610M params, GT-blurred reference by default).
# 6h may be tight for full test split — re-submit or set DVST_MAX_PAIRS for smoke test.
#SBATCH --job-name=orion_dvst_zs
#SBATCH --output=logs/orion_dvst_zero_shot_%j.out
#SBATCH --error=logs/orion_dvst_zero_shot_%j.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/bash_scripts/_common.sh"
mkdir -p logs results

source /home/zhangtin/miniforge3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-D_VST}"
source "${REPO_ROOT}/bash_scripts/_cuda.sh"

# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/orion_scratch_env.sh"

export MODEL=dvst
export MODE="prepare|test|metrics"
export IMAGE_SIZE="${IMAGE_SIZE:-512}"
export DVST_ROOT="${DVST_ROOT:-${REPO_ROOT}/../D-VST}"
export DVST_EVAL_CKPT="${DVST_EVAL_CKPT:-${DVST_ROOT}/weights/dvst_pretrained/HE2mIHC.ckpt}"
export DVST_PREPARE_SPLITS=test
export DVST_EVAL_PREPARE_ONLY=1
export DVST_REF_MODE="${DVST_REF_MODE:-paired_gt}"

echo "DATAROOT=${DATAROOT}"
echo "DVST_DATA_ROOT=${ORION_SCRATCH_ROOT}/datasets/orion_lite_dvst"
echo "DVST_EVAL_CKPT=${DVST_EVAL_CKPT}"

bash scripts/run_orion_dvst.sh
