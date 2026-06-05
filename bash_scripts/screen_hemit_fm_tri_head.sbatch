#!/bin/bash
# Screen: tri-head scratch 30 epochs (arch ablation, no seg).
#
#   sbatch bash_scripts/screen_hemit_fm_tri_head.sbatch
#   export TEST_EPOCH=30 && sbatch bash_scripts/test_hemit_fm_tri_head_scratch.sbatch
#SBATCH --job-name=hemit_tri_sc
#SBATCH --output=logs/hemit_fm_tri_head_screen_%j.out
#SBATCH --error=logs/hemit_fm_tri_head_screen_%j.err
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

export MODEL=vanilla_fm MODE=train
export DATAROOT="${DATAROOT:-./datasets/hemit}"
export BATCH_SIZE=1 DISPLAY_ID=-1
export N_EPOCHS=25
export N_EPOCHS_DECAY=5

source "${REPO_ROOT}/bash_scripts/_vanilla_fm_sbatch_preamble.sh"
source "${REPO_ROOT}/scripts/vanilla_fm_env.sh"
vanilla_fm_apply_fm_tri_head_scratch_env
export VANILLA_FM_ENV_LOCKED=1
vanilla_fm_print_train_env
bash "${REPO_ROOT}/scripts/run_hemit_vanilla_fm.sh"
