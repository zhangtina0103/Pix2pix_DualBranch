#!/bin/bash
# Tri-head + pooled cross-attn + seg + init (no train ODE-aux — OOM on L40S 44GB with ODE backward).
# 6h chunks → resume. git pull required.
# OOM on FM pass: export FM_CROSS_ATTN_DECODER=0 FM_CROSS_ATTN_HEADS=4
#   sbatch bash_scripts/train_hemit_cond_fm_advanced_scratch.sbatch
#   export RESUME_PROFILE=advanced && sbatch bash_scripts/resume_hemit_fm_scratch.sbatch
#SBATCH --job-name=hemit_adv
#SBATCH --output=logs/hemit_cond_fm_advanced_scratch_%j.out
#SBATCH --error=logs/hemit_cond_fm_advanced_scratch_%j.err
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

source "${REPO_ROOT}/bash_scripts/_vanilla_fm_sbatch_preamble.sh"
source "${REPO_ROOT}/scripts/vanilla_fm_env.sh"
vanilla_fm_apply_cond_fm_advanced_scratch_env
export VANILLA_FM_ENV_LOCKED=1
hemit_require_seg_masks "${DATAROOT}"
vanilla_fm_print_train_env
bash "${REPO_ROOT}/scripts/run_hemit_vanilla_fm.sh"
