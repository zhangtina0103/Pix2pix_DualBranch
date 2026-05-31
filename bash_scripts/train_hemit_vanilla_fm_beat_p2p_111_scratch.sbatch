#!/bin/bash
# FM + strong perc + ODE sample L1 (train on inference path), weights 1,1,1, scratch 1→130.
# Slower than joint_perc; targets SSIM gap to pix2pix. OOM → FM_PERC_SIZE=256 FM_SAMPLE_L1_PROB=0.25
#
#   sbatch bash_scripts/train_hemit_vanilla_fm_beat_p2p_111_scratch.sbatch
#SBATCH --job-name=hemit_beat_p2p
#SBATCH --output=logs/hemit_beat_p2p_111_scratch_%j.out
#SBATCH --error=logs/hemit_beat_p2p_111_scratch_%j.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/bash_scripts/_common.sh"
mkdir -p logs
source /home/zhangtin/miniforge3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-pix2pix_cuda}"
source "${REPO_ROOT}/bash_scripts/_cuda.sh"

export MODEL=vanilla_fm
export MODE=train
export HEMIT_SRC="${HEMIT_SRC:-/home/zhangtin/virtual-staining/data/hemit}"
export DATAROOT="${DATAROOT:-./datasets/hemit}"
export BATCH_SIZE=1
export DISPLAY_ID=-1

# shellcheck source=/dev/null
source "${REPO_ROOT}/bash_scripts/_vanilla_fm_sbatch_preamble.sh"
source "${REPO_ROOT}/scripts/vanilla_fm_env.sh"
vanilla_fm_apply_beat_pix2pix_111_scratch_env
export VANILLA_FM_ENV_LOCKED=1
vanilla_fm_print_train_env
echo "Train epochs: ${EPOCH_COUNT} .. $(( N_EPOCHS + N_EPOCHS_DECAY )) (beat_p2p_111 scratch)"
bash "${REPO_ROOT}/scripts/run_hemit_vanilla_fm.sh"
