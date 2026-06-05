#!/bin/bash
# Screen test @ TEST_EPOCH (default 130), 100 tiles. Final: eval_hemit_cond_fm_seg_only_scratch.sbatch
#
#   export TEST_EPOCH=30
#   sbatch bash_scripts/test_hemit_cond_fm_seg_only_scratch.sbatch
#SBATCH --job-name=hemit_cfm_so_t
#SBATCH --output=logs/hemit_cond_fm_seg_only_scratch_test_%j.out
#SBATCH --error=logs/hemit_cond_fm_seg_only_scratch_test_%j.err
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
export MODE="${MODE:-test|metrics}"
export GPU_IDS=0
export DISPLAY_ID=-1
export DATAROOT="${DATAROOT:-./datasets/hemit}"
export TEST_EPOCH="${TEST_EPOCH:-130}"
export NUM_TEST="${NUM_TEST:-100}"

source "${REPO_ROOT}/bash_scripts/_vanilla_fm_sbatch_preamble.sh"
source "${REPO_ROOT}/scripts/vanilla_fm_env.sh"
vanilla_fm_apply_cond_seg_only_scratch_env
export VANILLA_FM_ENV_LOCKED=1
export MODEL=vanilla_fm
source "${REPO_ROOT}/scripts/hemit_model_profiles.sh"
export VANILLA_FM_EXPECTED_TRAIN_NAME="${TRAIN_NAME}"
vanilla_fm_verify_locked_env
hemit_require_seg_masks "${DATAROOT}"
CKPT_G="${REPO_ROOT}/checkpoints/${TRAIN_NAME}/${TEST_EPOCH}_net_G.pth"
[[ -f "${CKPT_G}" ]] || {
  echo "ERROR: missing ${CKPT_G}" >&2
  ls -1 "${REPO_ROOT}/checkpoints/${TRAIN_NAME}/"*_net_G.pth 2>/dev/null || true
  exit 1
}
echo "Using checkpoint: ${CKPT_G}"
vanilla_fm_print_train_env
bash "${REPO_ROOT}/scripts/run_hemit_all.sh"
