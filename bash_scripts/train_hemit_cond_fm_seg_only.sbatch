#!/bin/bash
# Phase A @ 130: seg concat only (no init/ODE-aux). Finetune joint_perc/80 → 130.
#   sbatch bash_scripts/train_hemit_cond_fm_seg_only.sbatch
#SBATCH --job-name=hemit_cfm_seg
#SBATCH --output=logs/hemit_cond_fm_seg_only_%j.out
#SBATCH --error=logs/hemit_cond_fm_seg_only_%j.err
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

export MODEL=vanilla_fm
export MODE=train
export HEMIT_SRC="${HEMIT_SRC:-/home/zhangtin/virtual-staining/data/hemit}"
export DATAROOT="${DATAROOT:-./datasets/hemit}"
export BATCH_SIZE=1
export DISPLAY_ID=-1
export TRAIN_LR=0.00005

# shellcheck source=/dev/null
source "${REPO_ROOT}/bash_scripts/_vanilla_fm_sbatch_preamble.sh"
source "${REPO_ROOT}/scripts/vanilla_fm_env.sh"
vanilla_fm_apply_cond_seg_only_env
export VANILLA_FM_ENV_LOCKED=1
python "${REPO_ROOT}/scripts/generate_hemit_seg_masks.py" --dataroot "${REPO_ROOT}/${DATAROOT}"
source "${REPO_ROOT}/bash_scripts/_vanilla_fm_finetune_from_joint_perc.sh"
vanilla_fm_finetune_joint_opt_from_perc_80
vanilla_fm_print_train_env
echo "Train epochs: ${EPOCH_COUNT:-?} .. $(( ${N_EPOCHS:-0} + ${N_EPOCHS_DECAY:-0} )) (expect 81..130)"
bash "${REPO_ROOT}/scripts/run_hemit_vanilla_fm.sh"
