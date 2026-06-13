#!/bin/bash
# Cross-attn FM eval @80 on Hoechst+CD3e+CD4 panel.
#   export ORION_PANEL=cd3_cd4 TEST_EPOCH=80
#   sbatch bash_scripts/eval_orion_cd3_cd4_fm_cross_attn_scratch.sbatch
#SBATCH --job-name=orion_cd4_xattn_ev
#SBATCH --output=logs/orion_cd3_cd4_fm_cross_attn_eval_%j.out
#SBATCH --error=logs/orion_cd3_cd4_fm_cross_attn_eval_%j.err
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

source "${REPO_ROOT}/bash_scripts/_orion_cd3_cd4_preamble.sh"
_orion_cd3_cd4_check_dataroot

export MODEL=vanilla_fm
export TEST_EPOCH="${TEST_EPOCH:-80}"
export GPU_IDS=0

source "${REPO_ROOT}/bash_scripts/_vanilla_fm_sbatch_preamble.sh"
source "${REPO_ROOT}/scripts/vanilla_fm_env.sh"
vanilla_fm_apply_orion_fm_cross_attn_scratch_env
export VANILLA_FM_ENV_LOCKED=1
source "${REPO_ROOT}/scripts/orion_model_profiles.sh"
vanilla_fm_verify_locked_env

CKPT="${REPO_ROOT}/checkpoints/${TRAIN_NAME}/${TEST_EPOCH}_net_G.pth"
echo "cross-attn FM TRAIN_NAME=${TRAIN_NAME} epoch=${TEST_EPOCH}"
[[ -f "${CKPT}" ]] || { echo "ERROR: missing ${CKPT}" >&2; exit 1; }

export MODE=test
bash scripts/run_orion_native.sh
export MODE=metrics
bash scripts/run_orion_native.sh

echo "score.csv → results/${TRAIN_NAME}/test_${TEST_EPOCH}/images/score.csv"
