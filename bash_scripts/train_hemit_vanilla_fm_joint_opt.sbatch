#!/bin/bash
# Optimized conditional FM: seg + informed init + perc/vel; finetune joint_perc/80 → 130.
# Target: beat 0.831 FM + ~0.854 pix2pix (not vanilla-only — conditioning is part of the stack).
#   sbatch bash_scripts/train_hemit_vanilla_fm_joint_opt.sbatch
#SBATCH --job-name=hemit_vfm_opt
#SBATCH --output=logs/hemit_vfm_joint_opt_%j.out
#SBATCH --error=logs/hemit_vfm_joint_opt_%j.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/bash_scripts/_common.sh"
mkdir -p logs
source /home/zhangtin/miniforge3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-pix2pix_cuda}"
source "${REPO_ROOT}/bash_scripts/_cuda.sh"

export MODE=train
export HEMIT_SRC="${HEMIT_SRC:-/home/zhangtin/virtual-staining/data/hemit}"
export DATAROOT="${DATAROOT:-./datasets/hemit}"
export BATCH_SIZE=1
export DISPLAY_ID=-1
export TRAIN_LR=0.00005

# shellcheck source=/dev/null
source "${REPO_ROOT}/bash_scripts/_vanilla_fm_sbatch_preamble.sh"
source "${REPO_ROOT}/scripts/vanilla_fm_env.sh"
vanilla_fm_apply_joint_fm_opt_env
export VANILLA_FM_ENV_LOCKED=1
hemit_require_seg_masks "${DATAROOT}"
source "${REPO_ROOT}/bash_scripts/_vanilla_fm_finetune_from_joint_perc.sh"
vanilla_fm_finetune_joint_opt_from_perc_80
vanilla_fm_print_train_env
bash scripts/run_hemit_vanilla_fm.sh
