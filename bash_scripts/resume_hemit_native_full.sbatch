#!/bin/bash
# Resume native baseline until epoch 80 (6h walltime per submit).
#
#   export MODEL=pix2pix
#   sbatch bash_scripts/resume_hemit_native_full.sbatch
#SBATCH --job-name=hemit_res
#SBATCH --output=logs/hemit_resume_%x_%j.out
#SBATCH --error=logs/hemit_resume_%x_%j.err
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

export MODEL="${MODEL:?Set MODEL before sbatch}"
export HEMIT_SRC="${HEMIT_SRC:-/home/zhangtin/HEMIT}"
export DATAROOT="${DATAROOT:-./datasets/hemit}"
export GPU_IDS=0

bash scripts/resume_hemit_native_full.sh
