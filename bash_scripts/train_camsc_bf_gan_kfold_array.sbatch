#!/bin/bash
# Train CUT / ASP / CycleGAN on CaMSC k-fold (one fold per array task).
#
#   export CAMSC_KFOLD_ROOT=~/orcd/scratch/camsc/datasets/camsc_bf_kfold_aug
#   export CAMSC_ENABLE_AUG=1
#   CAMSC_MODEL=cut  sbatch bash_scripts/train_camsc_bf_gan_kfold_array.sbatch
#   CAMSC_MODEL=asp  sbatch bash_scripts/train_camsc_bf_gan_kfold_array.sbatch
#   CAMSC_MODEL=cyclegan sbatch bash_scripts/train_camsc_bf_gan_kfold_array.sbatch
#SBATCH --job-name=camsc_gan
#SBATCH --output=logs/camsc_%x_%A_%a.out
#SBATCH --error=logs/camsc_%x_%A_%a.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --array=0-4
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

# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/camsc_scratch_env.sh"

export CAMSC_MODEL="${CAMSC_MODEL:?Set CAMSC_MODEL=cut|asp|cyclegan before sbatch}"
case "${CAMSC_MODEL}" in
  cut|asp|cyclegan) ;;
  *) echo "ERROR: CAMSC_MODEL=${CAMSC_MODEL} (need cut|asp|cyclegan)" >&2; exit 1 ;;
esac

export CAMSC_ENABLE_AUG="${CAMSC_ENABLE_AUG:-1}"
camsc_apply_train_data_env

export MODE=train
export GPU_IDS=0
export DISPLAY_ID="${DISPLAY_ID:--1}"
export N_EPOCHS="${N_EPOCHS:-100}"
export N_EPOCHS_DECAY="${N_EPOCHS_DECAY:-20}"
export LR_DECAY_ITERS="${LR_DECAY_ITERS:-100}"

if [[ "${CAMSC_MODEL}" == "cut" || "${CAMSC_MODEL}" == "asp" ]]; then
  export BATCH_SIZE="${BATCH_SIZE:-2}"
else
  export BATCH_SIZE="${BATCH_SIZE:-4}"
fi

camsc_apply_model_env

FOLD="${SLURM_ARRAY_TASK_ID}"
echo "${CAMSC_MODEL} fold ${FOLD} @ ${CAMSC_KFOLD_ROOT}  BATCH_SIZE=${BATCH_SIZE}"
camsc_run_fold_train "${FOLD}"
