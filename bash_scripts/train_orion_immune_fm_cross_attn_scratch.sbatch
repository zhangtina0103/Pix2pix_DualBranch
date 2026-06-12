#!/bin/bash
# Orion immune panel cross-attn FM @80.
#   sbatch bash_scripts/train_orion_immune_fm_cross_attn_scratch.sbatch
# Resume: sbatch bash_scripts/resume_orion_immune_fm_cross_attn_scratch.sbatch
#SBATCH --job-name=orion_imm_xattn
#SBATCH --output=logs/orion_immune_fm_cross_attn_%j.out
#SBATCH --error=logs/orion_immune_fm_cross_attn_%j.err
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

source "${REPO_ROOT}/bash_scripts/_orion_immune_preamble.sh"
_orion_immune_check_dataroot

export MODEL=vanilla_fm MODE=train
export BATCH_SIZE="${BATCH_SIZE:-4}"
export DISPLAY_ID=-1

source "${REPO_ROOT}/bash_scripts/_vanilla_fm_sbatch_preamble.sh"
source "${REPO_ROOT}/scripts/vanilla_fm_env.sh"
vanilla_fm_apply_orion_fm_cross_attn_scratch_env
unset FM_CROSS_ATTN_DECODER
export FM_CROSS_ATTN_HEADS="${FM_CROSS_ATTN_HEADS:-4}"
export VANILLA_FM_ENV_LOCKED=1
source "${REPO_ROOT}/scripts/orion_model_profiles.sh"
vanilla_fm_verify_locked_env
vanilla_fm_print_train_env

bash scripts/run_orion_native.sh
