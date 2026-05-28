#!/bin/bash
# Resume Vanilla FM epochs 76–80 (load 75_net_G.pth). Short GPU job.
#
#   sbatch bash_scripts/resume_hemit_vanilla_fm.sbatch
#
# After it finishes:
#   MODEL=vanilla_fm MODE=test TEST_EPOCH=80 sbatch bash_scripts/run_hemit_all.sbatch
#SBATCH --job-name=hemit_vfm_resume
#SBATCH --output=logs/hemit_vanilla_fm_resume_%j.out
#SBATCH --error=logs/hemit_vanilla_fm_resume_%j.err
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

export HEMIT_SRC="${HEMIT_SRC:-/home/zhangtin/virtual-staining/data/hemit}"
export DATAROOT="${DATAROOT:-./datasets/hemit}"
export GPU_IDS=0

echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Started: $(date)"

bash scripts/resume_hemit_vanilla_fm.sh

echo "Finished: $(date)"
