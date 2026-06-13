#!/bin/bash
# Resume pix2pix on Hoechst+CD3e+CD4 from epoch 50 → 80 (after screen @50).
#
#   sbatch bash_scripts/resume_orion_cd3_cd4_pix2pix.sbatch
#
# Eval @80 when done:
#   export TEST_EPOCH=80
#   sbatch bash_scripts/eval_orion_cd3_cd4_pix2pix.sbatch
#SBATCH --job-name=orion_cd4_p2p_r
#SBATCH --output=logs/orion_cd3_cd4_pix2pix_resume_%j.out
#SBATCH --error=logs/orion_cd3_cd4_pix2pix_resume_%j.err
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

export MODEL=pix2pix
export N_EPOCHS=50
export N_EPOCHS_DECAY=30
export RESUME_FROM_EPOCH="${RESUME_FROM_EPOCH:-50}"
export VAL_FREQ="${VAL_FREQ:-999}"

echo "Resume pix2pix: DATAROOT=${DATAROOT} from epoch ${RESUME_FROM_EPOCH} to 80"
echo "Checkpoint dir: checkpoints/orion_lite_pix2pix_resnet9_cd3_cd4_512/"

bash "${REPO_ROOT}/scripts/resume_orion_native.sh"
