#!/bin/bash
# Full fair sweep @80 after pix2pix gate PASS: pix2pix/CUT/ASP + cross-attn FM.
# Comparison bar = best GAN (typically pix2pix), not vanilla FM.
#
#   sbatch bash_scripts/screen_orion_cd3_cd4_full.sbatch
#SBATCH --job-name=orion_cd4_full
#SBATCH --output=logs/orion_cd3_cd4_full_submit_%j.out
#SBATCH --error=logs/orion_cd3_cd4_full_submit_%j.err
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/bash_scripts/_common.sh"
mkdir -p logs

cd "${REPO_ROOT}"

j_gan=$(sbatch --parsable bash_scripts/train_orion_cd3_cd4_gan_array.sbatch)
j_fm=$(sbatch --parsable bash_scripts/train_orion_cd3_cd4_fm_cross_attn_scratch.sbatch)

echo "Submitted full Hoechst+CD3e+CD4 sweep @80:"
echo "  GAN array:  job ${j_gan}  (pix2pix,CUT,ASP)"
echo "  cross-attn: job ${j_fm}"
echo ""
echo "Eval @80 when done:"
echo "  export ORION_PANEL=cd3_cd4 TEST_EPOCH=80"
echo "  sbatch --array=0-2 bash_scripts/eval_orion_cd3_cd4_gan_array.sbatch"
echo "  sbatch bash_scripts/eval_orion_cd3_cd4_fm_cross_attn_scratch.sbatch"
