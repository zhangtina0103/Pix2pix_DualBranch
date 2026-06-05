#!/bin/bash
#   export TEST_EPOCH=130
#   sbatch bash_scripts/test_hemit_cond_fm_advanced_scratch.sbatch
#SBATCH --job-name=hemit_adv_t
#SBATCH --output=logs/hemit_cond_fm_advanced_test_%j.out
#SBATCH --error=logs/hemit_cond_fm_advanced_test_%j.err
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

export MODEL=vanilla_fm MODE="${MODE:-test|metrics}"
export GPU_IDS=0 DISPLAY_ID=-1
export DATAROOT="${DATAROOT:-./datasets/hemit}"
export TEST_EPOCH="${TEST_EPOCH:-130}" NUM_TEST=100

source "${REPO_ROOT}/bash_scripts/_vanilla_fm_sbatch_preamble.sh"
source "${REPO_ROOT}/scripts/vanilla_fm_env.sh"
vanilla_fm_apply_cond_fm_advanced_scratch_env
export VANILLA_FM_ENV_LOCKED=1
hemit_require_seg_masks "${DATAROOT}"
CKPT="${REPO_ROOT}/checkpoints/${TRAIN_NAME}/${TEST_EPOCH}_net_G.pth"
[[ -f "${CKPT}" ]] || { echo "ERROR: missing ${CKPT}" >&2; exit 1; }
echo "Using checkpoint: ${CKPT}"
vanilla_fm_print_train_env
bash "${REPO_ROOT}/scripts/run_hemit_all.sh"
