#!/bin/bash
# Orion-Lite cross-attn FM @80 (fair compare, HEMIT-matched recipe).
#   sbatch bash_scripts/train_orion_lite_fm_cross_attn_scratch.sbatch
#SBATCH --job-name=orion_xattn
#SBATCH --output=logs/orion_fm_cross_attn_%j.out
#SBATCH --error=logs/orion_fm_cross_attn_%j.err
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
export DATAROOT="${DATAROOT:-./datasets/orion_lite}"
export ORION_SRC="${ORION_SRC:-${REPO_ROOT}/data/orion/ORIONCRC_dataset_tile_20x}"
export HEMIT_TRAIN_SIZE="${HEMIT_TRAIN_SIZE:-512}"
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

[[ -d "${DATAROOT}/testA" ]] || { echo "ERROR: run prepare_orion_lite.sbatch first" >&2; exit 1; }
bash scripts/run_orion_native.sh
