#!/bin/bash
# Full train: seg + informed init, no ODE-aux, scratch 1→130.
#
#   sbatch bash_scripts/train_hemit_cond_fm_seg_init_scratch.sbatch
#   sbatch bash_scripts/eval_hemit_cond_fm_seg_init_scratch.sbatch
#SBATCH --job-name=hemit_cfm_si
#SBATCH --output=logs/hemit_cond_fm_seg_init_scratch_%j.out
#SBATCH --error=logs/hemit_cond_fm_seg_init_scratch_%j.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/bash_scripts/_common.sh"
mkdir -p logs
source /home/zhangtin/miniforge3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-pix2pix_cuda}"
source "${REPO_ROOT}/bash_scripts/_cuda.sh"

export MODEL=vanilla_fm MODE=train
export DATAROOT="${DATAROOT:-./datasets/hemit}"
export DISPLAY_ID=-1 NUM_THREADS=8
export BATCH_SIZE="${BATCH_SIZE:-4}"

source "${REPO_ROOT}/bash_scripts/_vanilla_fm_sbatch_preamble.sh"
source "${REPO_ROOT}/scripts/vanilla_fm_env.sh"
vanilla_fm_apply_cond_seg_init_scratch_env
export VANILLA_FM_ENV_LOCKED=1
hemit_require_seg_masks "${DATAROOT}"
vanilla_fm_print_train_env
bash "${REPO_ROOT}/scripts/run_hemit_vanilla_fm.sh"
