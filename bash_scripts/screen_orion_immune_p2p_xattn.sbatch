#!/bin/bash
# Screen Orion immune panel: pix2pix vs cross-attn FM only (@80).
# Run CUT/ASP/vanilla FM after score.csv looks good.
#
#   sbatch bash_scripts/prepare_orion_immune.sbatch          # once
#   sbatch bash_scripts/screen_orion_immune_p2p_xattn.sbatch # this file
#
# Eval @80 (after both finish):
#   export ORION_PANEL=immune TEST_EPOCH=80
#   sbatch --array=0 bash_scripts/eval_orion_immune_gan_array.sbatch
#   sbatch bash_scripts/eval_orion_immune_fm_cross_attn_scratch.sbatch
#SBATCH --job-name=orion_imm_scr
#SBATCH --output=logs/orion_immune_screen_submit_%j.out
#SBATCH --error=logs/orion_immune_screen_submit_%j.err
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/bash_scripts/_common.sh"
mkdir -p logs

cd "${REPO_ROOT}"
[[ -d "${HOME}/orcd/scratch/orion/datasets/orion_immune_cd3_cd8_foxp3/testA" ]] || \
[[ -d "./datasets/orion_immune_cd3_cd8_foxp3/testA" ]] || {
  echo "ERROR: immune dataroot missing. Run: sbatch bash_scripts/prepare_orion_immune.sbatch" >&2
  exit 1
}

j1=$(sbatch --parsable --array=0 bash_scripts/train_orion_immune_gan_array.sbatch)
j2=$(sbatch --parsable bash_scripts/train_orion_immune_fm_cross_attn_scratch.sbatch)

echo "Submitted immune screen:"
echo "  pix2pix:    job ${j1}  (train_orion_immune_gan_array --array=0)"
echo "  cross-attn: job ${j2}  (train_orion_immune_fm_cross_attn_scratch)"
echo ""
echo "Checkpoints:"
echo "  checkpoints/orion_lite_pix2pix_resnet9_immune_512/"
echo "  checkpoints/orion_lite_fm_cross_attn_scratch_immune_512/"
echo ""
echo "Resume if 6h timeout:"
echo "  export MODEL=pix2pix && sbatch bash_scripts/resume_orion_immune_gan.sbatch"
echo "  sbatch bash_scripts/resume_orion_immune_fm_cross_attn_scratch.sbatch"
