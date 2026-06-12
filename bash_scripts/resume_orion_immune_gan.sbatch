#!/bin/bash
# Resume one immune-panel GAN until epoch 80.
#   export MODEL=pix2pix
#   sbatch bash_scripts/resume_orion_immune_gan.sbatch
#   sbatch --array=0 bash_scripts/resume_orion_immune_gan.sbatch
#SBATCH --job-name=orion_imm_gan_r
#SBATCH --output=logs/orion_immune_gan_resume_%x_%j.out
#SBATCH --error=logs/orion_immune_gan_resume_%x_%j.err
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

export ORION_PANEL=immune
if [[ -z "${MODEL:-}" && -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  MODELS=(pix2pix cut asp)
  export MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"
fi
export MODEL="${MODEL:?Set MODEL or use --array=N}"
export VAL_FREQ="${VAL_FREQ:-999}"

bash "${REPO_ROOT}/scripts/resume_orion_native.sh"
