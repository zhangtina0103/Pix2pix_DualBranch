#!/bin/bash
#   sbatch bash_scripts/resume_orion_immune_fm_cross_attn_scratch.sbatch
#SBATCH --job-name=orion_imm_xattn_r
#SBATCH --output=logs/orion_immune_fm_cross_attn_resume_%j.out
#SBATCH --error=logs/orion_immune_fm_cross_attn_resume_%j.err
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

export ORION_PANEL=immune
export RESUME_PROFILE=cross_attn
export BATCH_SIZE="${BATCH_SIZE:-4}"
export VAL_FREQ="${VAL_FREQ:-999}"

source "${REPO_ROOT}/bash_scripts/_vanilla_fm_sbatch_preamble.sh"
bash "${REPO_ROOT}/scripts/resume_orion_fm_scratch.sh"
