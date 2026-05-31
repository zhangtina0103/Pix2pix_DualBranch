#!/bin/bash
# Resume cond_consistent after time limit (do NOT re-run train_hemit_cond_fm_consistent.sbatch).
#
#   export RESUME_FROM_EPOCH=105   # highest saved epoch
#   export EPOCH_COUNT=$((RESUME_FROM_EPOCH + 1))
#   sbatch bash_scripts/resume_hemit_cond_fm_consistent.sbatch
#SBATCH --job-name=hemit_cfm_res
#SBATCH --output=logs/hemit_cond_fm_consistent_resume_%j.out
#SBATCH --error=logs/hemit_cond_fm_consistent_resume_%j.err
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
export TRAIN_LR="${TRAIN_LR:-0.00005}"
export N_EPOCHS=100
export N_EPOCHS_DECAY=30

export RESUME_FROM_EPOCH="${RESUME_FROM_EPOCH:-105}"
export EPOCH_COUNT="${EPOCH_COUNT:-$((RESUME_FROM_EPOCH + 1))}"
export CONTINUE_TRAIN=1

# shellcheck source=/dev/null
source "${REPO_ROOT}/bash_scripts/_vanilla_fm_sbatch_preamble.sh"
source "${REPO_ROOT}/scripts/vanilla_fm_env.sh"
vanilla_fm_apply_cond_consistent_env
export VANILLA_FM_ENV_LOCKED=1
vanilla_fm_print_train_env
echo "Resume: load epoch ${RESUME_FROM_EPOCH}, train ${EPOCH_COUNT}..$((N_EPOCHS + N_EPOCHS_DECAY))"
bash "${REPO_ROOT}/scripts/run_hemit_vanilla_fm.sh"
