#!/bin/bash
# Orion-Lite cross-attn eval: full test split → score.csv
#
#   export TEST_EPOCH=80
#   sbatch bash_scripts/eval_orion_lite_fm_cross_attn_scratch.sbatch
#SBATCH --job-name=orion_xattn_ev
#SBATCH --output=logs/orion_fm_cross_attn_eval_%j.out
#SBATCH --error=logs/orion_fm_cross_attn_eval_%j.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/bash_scripts/_common.sh"
mkdir -p logs results

source /home/zhangtin/miniforge3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-pix2pix_cuda}"
source "${REPO_ROOT}/bash_scripts/_cuda.sh"

export MODEL=vanilla_fm
export DATAROOT="${DATAROOT:-./datasets/orion_lite}"
export TEST_EPOCH="${TEST_EPOCH:-80}"
export GPU_IDS=0
export DISPLAY_ID=-1

source "${REPO_ROOT}/bash_scripts/_vanilla_fm_sbatch_preamble.sh"
source "${REPO_ROOT}/scripts/vanilla_fm_env.sh"
vanilla_fm_apply_orion_fm_cross_attn_scratch_env
unset FM_CROSS_ATTN_DECODER
export FM_CROSS_ATTN_HEADS="${FM_CROSS_ATTN_HEADS:-4}"
export VANILLA_FM_ENV_LOCKED=1
source "${REPO_ROOT}/scripts/orion_model_profiles.sh"
vanilla_fm_verify_locked_env
vanilla_fm_print_train_env

CKPT="${REPO_ROOT}/checkpoints/${TRAIN_NAME}/${TEST_EPOCH}_net_G.pth"
echo "TRAIN_NAME=${TRAIN_NAME} TEST_EPOCH=${TEST_EPOCH} NUM_TEST=${NUM_TEST}"
[[ -f "${CKPT}" ]] || { echo "ERROR: missing ${CKPT}" >&2; exit 1; }

export MODE=test
bash scripts/run_orion_native.sh
export MODE=metrics
bash scripts/run_orion_native.sh

echo "Done. score.csv → results/${TRAIN_NAME}/test_${TEST_EPOCH}/images/score.csv"
