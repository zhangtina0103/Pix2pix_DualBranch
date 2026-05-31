#!/bin/bash
# Test hemit_cond_fm_init_only @ 130
#   sbatch bash_scripts/test_hemit_cond_fm_init_only.sbatch
#SBATCH --job-name=hemit_cfm_init_t
#SBATCH --output=logs/hemit_cond_fm_init_only_test_%j.out
#SBATCH --error=logs/hemit_cond_fm_init_only_test_%j.err
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
export NUM_TEST=100

# shellcheck source=/dev/null
source "${REPO_ROOT}/bash_scripts/_vanilla_fm_sbatch_preamble.sh"
source "${REPO_ROOT}/scripts/vanilla_fm_env.sh"
vanilla_fm_apply_cond_init_only_env
export VANILLA_FM_ENV_LOCKED=1
CKPT_G="${REPO_ROOT}/checkpoints/${TRAIN_NAME}/${TEST_EPOCH}_net_G.pth"
if [[ ! -f "${CKPT_G}" ]]; then
  echo "ERROR: missing checkpoint:" >&2
  echo "  ${CKPT_G}" >&2
  ls -1 "${REPO_ROOT}/checkpoints/${TRAIN_NAME}/"*_net_G.pth 2>/dev/null >&2 || true
  exit 1
fi
echo "Using checkpoint: ${CKPT_G}"
vanilla_fm_print_train_env
bash "${REPO_ROOT}/scripts/run_hemit_all.sh"
