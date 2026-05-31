#!/bin/bash
# cond_consistent v2: 8-step ODE-aux, perc=0.15 — faster per epoch than v1 scratch.
#
#   sbatch bash_scripts/train_hemit_cond_fm_consistent_v2.sbatch
# After 6h: RESUME_PROFILE=v2 sbatch bash_scripts/resume_hemit_fm_scratch.sbatch
#SBATCH --job-name=hemit_cfm_v2
#SBATCH --output=logs/hemit_cond_fm_consistent_v2_%j.out
#SBATCH --error=logs/hemit_cond_fm_consistent_v2_%j.err
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
vanilla_fm_apply_cond_consistent_v2_env
export VANILLA_FM_ENV_LOCKED=1
python "${REPO_ROOT}/scripts/generate_hemit_seg_masks.py" --dataroot "${REPO_ROOT}/${DATAROOT}"
vanilla_fm_print_train_env
echo "Train epochs: ${EPOCH_COUNT} .. $(( N_EPOCHS + N_EPOCHS_DECAY )) (v2 scratch, expect 1..130)"
bash "${REPO_ROOT}/scripts/run_hemit_vanilla_fm.sh"
