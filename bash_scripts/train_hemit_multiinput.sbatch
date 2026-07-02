#!/bin/bash
# Train one HEMIT multi-input variant on scratch (FM + cross-attn, 80 epochs).
#
#   source scripts/hemit_scratch_env.sh   # optional — applied automatically
#   HEMIT_MULTI_VARIANT=he_dapi sbatch bash_scripts/train_hemit_multiinput.sbatch
#
# All three: sbatch bash_scripts/train_hemit_multiinput_array.sbatch
#SBATCH --job-name=hemit_multi_tr
#SBATCH --output=logs/hemit_multiinput_train_%j.out
#SBATCH --error=logs/hemit_multiinput_train_%j.err
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

export MODEL=vanilla_fm MODE=train
export HEMIT_MULTI_VARIANT="${HEMIT_MULTI_VARIANT:?Set HEMIT_MULTI_VARIANT=he|dapi|he_dapi}"
export HEMIT_TRAIN_SIZE="${HEMIT_TRAIN_SIZE:-512}"
export BATCH_SIZE="${BATCH_SIZE:-4}"
export DISPLAY_ID=-1

source "${REPO_ROOT}/bash_scripts/_vanilla_fm_sbatch_preamble.sh"
source "${REPO_ROOT}/scripts/hemit_multiinput_env.sh"
hemit_multiinput_apply_env
vanilla_fm_print_train_env
echo "DATAROOT=${DATAROOT} CHECKPOINTS_DIR=${CHECKPOINTS_DIR}"
bash "${REPO_ROOT}/scripts/run_hemit_vanilla_fm.sh"
