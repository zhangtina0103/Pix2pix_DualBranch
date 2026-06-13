#!/bin/bash
# Full fair GAN baselines @80 on Hoechst+CD3e+CD4 (only after pix2pix gate PASS).
#   sbatch bash_scripts/train_orion_cd3_cd4_gan_array.sbatch
# One model:  sbatch --array=0 bash_scripts/train_orion_cd3_cd4_gan_array.sbatch
#SBATCH --job-name=orion_cd4_gan
#SBATCH --output=logs/orion_cd3_cd4_gan_%A_%a.out
#SBATCH --error=logs/orion_cd3_cd4_gan_%A_%a.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --array=0-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/bash_scripts/_common.sh"
mkdir -p logs
source /home/zhangtin/miniforge3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-pix2pix_cuda}"
source "${REPO_ROOT}/bash_scripts/_cuda.sh"

source "${REPO_ROOT}/bash_scripts/_orion_cd3_cd4_preamble.sh"
_orion_cd3_cd4_check_dataroot

MODELS=(pix2pix cut asp)
export MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"
export MODE=train
export N_EPOCHS=50
export N_EPOCHS_DECAY=30

echo "Task ${SLURM_ARRAY_TASK_ID}: MODEL=${MODEL} DATAROOT=${DATAROOT}"
bash scripts/run_orion_all.sh
