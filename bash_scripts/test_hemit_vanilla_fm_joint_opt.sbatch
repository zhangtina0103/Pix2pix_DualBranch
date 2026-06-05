#!/bin/bash
# Test optimized conditional FM @ 130 (seg + init_cond_v2 + tuned losses).
#   sbatch bash_scripts/test_hemit_vanilla_fm_joint_opt.sbatch
# Optional ODE sweep after train: bash scripts/sweep_fm_ode_steps.sh hemit_vanilla_fm_joint_opt 130
#SBATCH --job-name=hemit_vfm_opt_t
#SBATCH --output=logs/hemit_vfm_joint_opt_test_%j.out
#SBATCH --error=logs/hemit_vfm_joint_opt_test_%j.err
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
export HEMIT_SRC="${HEMIT_SRC:-/home/zhangtin/virtual-staining/data/hemit}"
export DATAROOT="${DATAROOT:-./datasets/hemit}"
export TEST_EPOCH="${TEST_EPOCH:-130}"
export NUM_TEST=100

# shellcheck source=/dev/null
source "${REPO_ROOT}/bash_scripts/_vanilla_fm_sbatch_preamble.sh"
source "${REPO_ROOT}/scripts/vanilla_fm_env.sh"
vanilla_fm_apply_joint_fm_opt_env
export VANILLA_FM_ENV_LOCKED=1
hemit_require_seg_masks "${DATAROOT}"
[[ -f "${REPO_ROOT}/checkpoints/${TRAIN_NAME}/${TEST_EPOCH}_net_G.pth" ]] || {
  echo "ERROR: train joint_opt first (need ${TRAIN_NAME}/${TEST_EPOCH}_net_G.pth)" >&2
  exit 1
}
vanilla_fm_print_train_env
bash "${REPO_ROOT}/scripts/run_hemit_all.sh"
