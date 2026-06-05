#!/bin/bash
# cond_consistent from epoch 1 (no joint_perc/80 seed). ~30h on L40S; request 48h.
#
#   sbatch bash_scripts/train_hemit_cond_fm_consistent_scratch.sbatch
#SBATCH --job-name=hemit_cfm_scr
#SBATCH --output=logs/hemit_cond_fm_consistent_scratch_%j.out
#SBATCH --error=logs/hemit_cond_fm_consistent_scratch_%j.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00

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

# shellcheck source=/dev/null
source "${REPO_ROOT}/bash_scripts/_vanilla_fm_sbatch_preamble.sh"
source "${REPO_ROOT}/scripts/vanilla_fm_env.sh"
vanilla_fm_apply_cond_consistent_scratch_env
export VANILLA_FM_ENV_LOCKED=1
hemit_require_seg_masks "${DATAROOT}"
vanilla_fm_print_train_env
echo "Train epochs: ${EPOCH_COUNT} .. $(( N_EPOCHS + N_EPOCHS_DECAY )) (scratch, expect 1..130)"
bash "${REPO_ROOT}/scripts/run_hemit_vanilla_fm.sh"
