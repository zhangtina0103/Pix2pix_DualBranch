#!/bin/bash
# Fix multi-input channel weights without a full 80-epoch rerun.
#
# Finetunes from epoch-80 checkpoint with FM_CHANNEL_WEIGHTS=1,1,0 (+15 epochs default).
#
#   HEMIT_MULTI_VARIANT=he sbatch bash_scripts/finetune_hemit_multiinput_rebalance.sbatch
#   HEMIT_MULTI_VARIANT=dapi sbatch bash_scripts/finetune_hemit_multiinput_rebalance.sbatch
#
# Eval @95 (or set HEMIT_MULTI_REBALANCE_EPOCHS):
#   HEMIT_MULTI_VARIANT=he TEST_EPOCH=95 sbatch bash_scripts/eval_hemit_multiinput.sbatch
#SBATCH --job-name=hemit_multi_rb
#SBATCH --output=logs/hemit_multiinput_rebalance_%j.out
#SBATCH --error=logs/hemit_multiinput_rebalance_%j.err
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

export DISPLAY_ID=-1
export GPU_IDS="${GPU_IDS:-0}"
export BATCH_SIZE="${BATCH_SIZE:-4}"
export HEMIT_MULTI_VARIANT="${HEMIT_MULTI_VARIANT:-he}"
export HEMIT_MULTI_REBALANCE_EPOCHS="${HEMIT_MULTI_REBALANCE_EPOCHS:-15}"

source "${REPO_ROOT}/bash_scripts/_vanilla_fm_sbatch_preamble.sh"
bash "${REPO_ROOT}/scripts/hemit_multiinput_rebalance.sh"
