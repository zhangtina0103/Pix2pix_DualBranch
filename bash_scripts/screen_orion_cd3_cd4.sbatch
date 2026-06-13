#!/bin/bash
# Prepare Hoechst+CD3e+CD4 dataroot, then pix2pix-only screen train.
#
#   sbatch bash_scripts/screen_orion_cd3_cd4.sbatch
#
# After train finishes:
#   sbatch bash_scripts/eval_orion_cd3_cd4_pix2pix_screen.sbatch
#SBATCH --job-name=orion_cd4_scr
#SBATCH --output=logs/orion_cd3_cd4_screen_submit_%j.out
#SBATCH --error=logs/orion_cd3_cd4_screen_submit_%j.err
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/bash_scripts/_common.sh"
mkdir -p logs

cd "${REPO_ROOT}"

j_prep=$(sbatch --parsable bash_scripts/prepare_orion_cd3_cd4.sbatch)
j_train=$(sbatch --parsable --dependency=afterok:"${j_prep}" \
  bash_scripts/train_orion_cd3_cd4_pix2pix_screen.sbatch)

echo "Submitted Hoechst+CD3e+CD4 pix2pix screen:"
echo "  prepare:  job ${j_prep}"
echo "  pix2pix:  job ${j_train}  (depends on ${j_prep}, 50 epochs)"
echo ""
echo "After pix2pix finishes, eval + gate:"
echo "  export SCREEN_EPOCH=50"
echo "  sbatch bash_scripts/eval_orion_cd3_cd4_pix2pix_screen.sbatch"
echo ""
echo "Resume pix2pix 50→80, then eval @80:"
echo "  sbatch bash_scripts/resume_orion_cd3_cd4_pix2pix.sbatch"
echo "  export TEST_EPOCH=80 && sbatch bash_scripts/eval_orion_cd3_cd4_pix2pix.sbatch"
echo ""
echo "Panel: Hoechst,CD3e,CD4 — Pan-CK replaced; comparison bar = pix2pix (not vanilla FM)."
