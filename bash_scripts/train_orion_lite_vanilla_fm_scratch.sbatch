#!/bin/bash
# Orion-Lite vanilla FM (no cross-attn) @80.
#   sbatch bash_scripts/train_orion_lite_vanilla_fm_scratch.sbatch
# After 6h timeout: sbatch bash_scripts/resume_orion_lite_vanilla_fm_scratch.sbatch
#SBATCH --job-name=orion_vfm
#SBATCH --output=logs/orion_vanilla_fm_%j.out
#SBATCH --error=logs/orion_vanilla_fm_%j.err
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

# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/orion_scratch_env.sh"

export MODEL=vanilla_fm MODE=train
export BATCH_SIZE="${BATCH_SIZE:-4}"
export DISPLAY_ID=-1

source "${REPO_ROOT}/bash_scripts/_vanilla_fm_sbatch_preamble.sh"
source "${REPO_ROOT}/scripts/vanilla_fm_env.sh"
vanilla_fm_apply_orion_joint_perc_scratch_env
export VANILLA_FM_ENV_LOCKED=1
source "${REPO_ROOT}/scripts/orion_model_profiles.sh"
vanilla_fm_verify_locked_env
vanilla_fm_print_train_env

bash scripts/run_orion_native.sh
