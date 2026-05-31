#!/bin/bash
# Resume scratch FM in 6h chunks (re-submit until epoch 130).
#
#   export RESUME_PROFILE=advanced
#   export RESUME_FROM_EPOCH=25   # optional; auto = latest numbered ckpt
#   sbatch bash_scripts/resume_hemit_fm_scratch.sbatch
#SBATCH --job-name=hemit_fm_res
#SBATCH --output=logs/hemit_fm_resume_%j.out
#SBATCH --error=logs/hemit_fm_resume_%j.err
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
export BATCH_SIZE=1
export DATAROOT="${DATAROOT:-./datasets/hemit}"

source "${REPO_ROOT}/bash_scripts/_vanilla_fm_sbatch_preamble.sh"
bash "${REPO_ROOT}/scripts/resume_hemit_fm_scratch.sh"
