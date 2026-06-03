#!/bin/bash
# Train one native HEMIT baseline on full data (1024², joint 3ch).
#
#   export HEMIT_SRC=/home/zhangtin/HEMIT
#   MODEL=pix2pix    sbatch bash_scripts/train_hemit_native_full.sbatch
#   MODEL=pix2pixhd  sbatch bash_scripts/train_hemit_native_full.sbatch
#   MODEL=cut        sbatch bash_scripts/train_hemit_native_full.sbatch
#   MODEL=asp        sbatch bash_scripts/train_hemit_native_full.sbatch
#   MODEL=cyclegan   sbatch bash_scripts/train_hemit_native_full.sbatch
#SBATCH --job-name=hemit_tr
#SBATCH --output=logs/hemit_train_%x_%j.out
#SBATCH --error=logs/hemit_train_%x_%j.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/bash_scripts/_common.sh"
mkdir -p logs

source /home/zhangtin/miniforge3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-pix2pix_cuda}"
source "${REPO_ROOT}/bash_scripts/_cuda.sh"

export MODEL="${MODEL:?Set MODEL=pix2pix|pix2pixhd|cut|asp|cyclegan before sbatch}"
export MODE=train
export HEMIT_SRC="${HEMIT_SRC:-/home/zhangtin/HEMIT}"
export DATAROOT="${DATAROOT:-./datasets/hemit}"
export GPU_IDS=0

echo "MODEL=${MODEL} MODE=${MODE} HEMIT_SRC=${HEMIT_SRC}"
ls "${DATAROOT}/trainA" | wc -l
bash scripts/run_hemit_all.sh
