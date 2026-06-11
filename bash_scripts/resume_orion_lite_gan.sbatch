#!/bin/bash
# Resume one Orion GAN baseline until epoch 80 (do NOT resubmit full array).
#
#   export MODEL=cyclegan
#   sbatch bash_scripts/resume_orion_lite_gan.sbatch
#
# Or by array index (0=pix2pix 1=cut 2=asp 3=cyclegan):
#   sbatch --array=3 bash_scripts/resume_orion_lite_gan.sbatch
#SBATCH --job-name=orion_gan_res
#SBATCH --output=logs/orion_gan_resume_%x_%j.out
#SBATCH --error=logs/orion_gan_resume_%x_%j.err
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

# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/orion_scratch_env.sh"

if [[ -z "${MODEL:-}" && -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  MODELS=(pix2pix cut asp cyclegan)
  export MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"
fi
export MODEL="${MODEL:?Set MODEL or use --array=N}"
export VAL_FREQ="${VAL_FREQ:-999}"

bash "${REPO_ROOT}/scripts/resume_orion_native.sh"
