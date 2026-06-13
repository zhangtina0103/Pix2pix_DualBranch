#!/bin/bash
# Full HNSCC pipeline: prepare → train all models → eval all → leaderboard.
#
# Prereq: copy hnscc.zip to ~/Downloads/ OR set HNSCC_SRC to extracted Case1..Case8
#
#   sbatch bash_scripts/run_hnscc_full_pipeline.sbatch
#
# Skip prepare if dataroot exists:
#   export SKIP_PREP=1 && sbatch bash_scripts/run_hnscc_full_pipeline.sbatch
#SBATCH --job-name=hnscc_full
#SBATCH --output=logs/hnscc_full_%j.out
#SBATCH --error=logs/hnscc_full_%j.err
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/bash_scripts/_common.sh"
mkdir -p logs

# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/hnscc_scratch_env.sh"

echo "===== HNSCC full pipeline ====="
echo "DATAROOT=${DATAROOT}"
echo "HNSCC_SRC=${HNSCC_SRC}"
echo "HNSCC_ZIP=${HNSCC_ZIP}"

j_prep=""
if [[ "${SKIP_PREP:-0}" != "1" ]]; then
  j_prep=$(sbatch --parsable bash_scripts/prepare_hnscc.sbatch)
  echo "Submitted prepare: job ${j_prep}"
  j_gan=$(sbatch --parsable --dependency=afterok:${j_prep} bash_scripts/train_hnscc_gan_array.sbatch)
else
  [[ -d "${DATAROOT}/testA" ]] || { echo "ERROR: SKIP_PREP=1 but ${DATAROOT}/testA missing" >&2; exit 1; }
  j_gan=$(sbatch --parsable bash_scripts/train_hnscc_gan_array.sbatch)
fi

j_vfm=$(sbatch --parsable --dependency=afterok:${j_gan} bash_scripts/train_hnscc_vanilla_fm_scratch.sbatch)
j_xattn=$(sbatch --parsable --dependency=afterok:${j_gan} bash_scripts/train_hnscc_fm_cross_attn_scratch.sbatch)

j_gan_ev=$(sbatch --parsable --dependency=afterok:${j_gan} bash_scripts/eval_hnscc_gan_array.sbatch)
j_vfm_ev=$(sbatch --parsable --dependency=afterok:${j_vfm} bash_scripts/eval_hnscc_vanilla_fm_scratch.sbatch)
j_xattn_ev=$(sbatch --parsable --dependency=afterok:${j_xattn} bash_scripts/eval_hnscc_fm_cross_attn_scratch.sbatch)

j_leader=$(sbatch --parsable --dependency=afterok:${j_gan_ev}:${j_vfm_ev}:${j_xattn_ev} \
  --wrap="cd ${REPO_ROOT} && source /home/zhangtin/miniforge3/etc/profile.d/conda.sh && conda activate pix2pix_cuda && python scripts/summarize_hnscc_leaderboard.py --epoch 80")

echo ""
echo "Submitted jobs:"
echo "  prepare:     ${j_prep:-skipped}"
echo "  train GAN:   ${j_gan}"
echo "  train vFM:   ${j_vfm}"
echo "  train xattn: ${j_xattn}"
echo "  eval GAN:    ${j_gan_ev}"
echo "  eval vFM:    ${j_vfm_ev}"
echo "  eval xattn:  ${j_xattn_ev}"
echo "  leaderboard: ${j_leader}"
