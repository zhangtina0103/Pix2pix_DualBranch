#!/bin/bash
# Upload all 6 HEMIT runs to Hugging Face (one folder per model).
# Login once: huggingface-cli login
#
#   sbatch bash_scripts/upload_hemit_phased_to_hf.sbatch
#   FREE_DISK=1 sbatch bash_scripts/upload_hemit_phased_to_hf.sbatch
#SBATCH --job-name=hemit_hf
#SBATCH --output=logs/hemit_hf_upload_%j.out
#SBATCH --error=logs/hemit_hf_upload_%j.err
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/bash_scripts/_common.sh"
mkdir -p logs

source /home/zhangtin/miniforge3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-pix2pix_cuda}"

export HF_REPO="${HF_REPO:-zhangtin/fixed}"
export TEST_EPOCH="${TEST_EPOCH:-80}"
export FREE_DISK="${FREE_DISK:-0}"

for MODEL in pix2pix pix2pixhd cut asp cyclegan vanilla_fm; do
  echo "===== HF upload: ${MODEL} ====="
  bash scripts/upload_hemit_ckpt_to_hf.sh
done

echo "All uploads finished: https://huggingface.co/${HF_REPO}"
