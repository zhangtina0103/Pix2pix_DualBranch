#!/bin/bash
# HNSCC GAN baselines @80: pix2pix, CUT, ASP, CycleGAN.
#   sbatch bash_scripts/train_hnscc_gan_array.sbatch
#SBATCH --job-name=hnscc_gan
#SBATCH --output=logs/hnscc_gan_%A_%a.out
#SBATCH --error=logs/hnscc_gan_%A_%a.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --array=0-3
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

# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/hnscc_scratch_env.sh"

MODELS=(pix2pix cut asp cyclegan)
export MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"
export MODE=train
export N_EPOCHS=50
export N_EPOCHS_DECAY=30

echo "Task ${SLURM_ARRAY_TASK_ID}: MODEL=${MODEL}"
[[ -d "${DATAROOT}/testA" ]] || { echo "ERROR: run prepare_hnscc.sbatch first" >&2; exit 1; }
bash scripts/run_hnscc_all.sh
