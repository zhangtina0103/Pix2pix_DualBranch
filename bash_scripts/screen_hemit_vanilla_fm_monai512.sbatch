#!/bin/bash
# Screen: MONAI UNet @ 512², 30 epochs (~same walltime as other FM screens).
# Full 130-epoch monai512 @ bs=1 was ~3h/epoch — not viable on 6h chunks.
#
#   sbatch bash_scripts/screen_hemit_vanilla_fm_monai512.sbatch
# Eval @ 945:
#   export TEST_EPOCH=30
#   sbatch bash_scripts/eval_hemit_vanilla_fm_monai512_screen.sbatch
# OOM: BATCH_SIZE=1 sbatch ...  or  FM_USE_CHECKPOINT=1 BATCH_SIZE=2 sbatch ...
#SBATCH --job-name=hemit_monai_sc
#SBATCH --output=logs/hemit_monai512_screen_%j.out
#SBATCH --error=logs/hemit_monai512_screen_%j.err
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
hemit_gpu_smoke_test

export MODEL=vanilla_fm MODE=train
export DATAROOT="${DATAROOT:-./datasets/hemit}"
export DISPLAY_ID=-1 NUM_THREADS=8
export BATCH_SIZE="${BATCH_SIZE:-2}"

source "${REPO_ROOT}/bash_scripts/_vanilla_fm_sbatch_preamble.sh"
source "${REPO_ROOT}/scripts/vanilla_fm_env.sh"
vanilla_fm_apply_monai512_screen_env
export VANILLA_FM_ENV_LOCKED=1
vanilla_fm_print_train_env
bash "${REPO_ROOT}/scripts/run_hemit_vanilla_fm.sh"
