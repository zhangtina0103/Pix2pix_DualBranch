#!/bin/bash
# Pix2pix-only gate @50 epochs on Hoechst+CD3e+CD4 (same recipe as Orion GAN baselines).
# Do NOT run FM until eval_orion_cd3_cd4_pix2pix_screen passes (mean cd3_pearson < 0.05).
#
#   sbatch bash_scripts/train_orion_cd3_cd4_pix2pix_screen.sbatch
#SBATCH --job-name=orion_cd4_p2p
#SBATCH --output=logs/orion_cd3_cd4_pix2pix_screen_%j.out
#SBATCH --error=logs/orion_cd3_cd4_pix2pix_screen_%j.err
#SBATCH --partition=mit_normal_gpu
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

export MODEL=pix2pix MODE=train
export N_EPOCHS=30
export N_EPOCHS_DECAY=20
export SCREEN_EPOCH=50

echo "pix2pix screen: DATAROOT=${DATAROOT} total_epochs=${SCREEN_EPOCH}"
bash scripts/run_orion_all.sh
echo "Eval: sbatch bash_scripts/eval_orion_cd3_cd4_pix2pix_screen.sbatch"
