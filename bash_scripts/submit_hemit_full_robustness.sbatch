#!/bin/bash
# Full HEMIT robustness pipeline for all paper models + baselines.
#
# Phase 1 — generate test TIFFs (skip models that already have images):
#   GAN array: pix2pix, CUT, ASP, CycleGAN
#   FM jobs:   vanilla_fm (joint_perc), cross_attn
#
# Phase 2 — extended metrics + downstream biology CSVs for every model with TIFFs
#
# Usage:
#   export TEST_EPOCH=80
#   sbatch bash_scripts/submit_hemit_full_robustness.sbatch
#
# CSV-only (no plots). Results → eval/hemit/robustness_comparison/
#SBATCH --job-name=hemit_full_rb
#SBATCH --output=logs/hemit_full_robustness_submit_%j.out
#SBATCH --error=logs/hemit_full_robustness_submit_%j.err
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/bash_scripts/_common.sh"
mkdir -p logs eval/hemit

cd "${REPO_ROOT}"
export TEST_EPOCH="${TEST_EPOCH:-80}"
export HEMIT_SRC="${HEMIT_SRC:-/home/zhangtin/HEMIT}"
ROBUST_OUTDIR="${ROBUST_OUTDIR:-eval/hemit/robustness_comparison}"

need_test() {
  local train_name="$1"
  local epoch="$2"
  local imgdir="results/${train_name}/test_${epoch}/images"
  [[ -d "${imgdir}" ]] && compgen -G "${imgdir}/*_fake_B.tif" > /dev/null
}

echo "===== HEMIT full robustness pipeline @ epoch ${TEST_EPOCH} ====="
python scripts/build_hemit_robustness_manifest.py --status --test-epoch "${TEST_EPOCH}"

deps=()

# GAN baselines (array); only submit if any GAN missing TIFFs
gan_need=0
for spec in "pix2pix:hemit_pix2pix_resnet9" "cut:hemit_cut_joint" "asp:hemit_asp_joint" "cyclegan:hemit_cyclegan_joint"; do
  train_name="${spec#*:}"
  if ! need_test "${train_name}" "${TEST_EPOCH}"; then
    gan_need=1
    break
  fi
done
if [[ "${gan_need}" == "1" ]]; then
  j_gan=$(sbatch --parsable --export=ALL,TEST_EPOCH="${TEST_EPOCH}",HEMIT_SRC="${HEMIT_SRC}" \
    --array=0-3 bash_scripts/eval_hemit_paper_gan_array.sbatch)
  deps+=("${j_gan}")
  echo "Submitted GAN test array: ${j_gan}"
else
  echo "GAN baselines: all have test TIFFs @${TEST_EPOCH} — skip"
fi

# vanilla FM @80 (joint_perc fair baseline)
if ! need_test "hemit_vanilla_fm_joint_perc" "${TEST_EPOCH}"; then
  j_vfm=$(sbatch --parsable --export=ALL,TEST_EPOCH="${TEST_EPOCH}",HEMIT_SRC="${HEMIT_SRC}",FM_PROFILE=joint_perc \
    bash_scripts/eval_hemit_paper_fm.sbatch)
  deps+=("${j_vfm}")
  echo "Submitted vanilla_fm test: ${j_vfm}"
else
  echo "vanilla_fm: TIFFs exist — skip"
fi

# cross-attn FM
if ! need_test "hemit_fm_cross_attn_scratch" "${TEST_EPOCH}"; then
  j_xattn=$(sbatch --parsable --export=ALL,TEST_EPOCH="${TEST_EPOCH}",HEMIT_SRC="${HEMIT_SRC}",FM_PROFILE=fm_cross_attn_scratch \
    bash_scripts/eval_hemit_paper_fm.sbatch)
  deps+=("${j_xattn}")
  echo "Submitted cross_attn test: ${j_xattn}"
else
  echo "cross_attn: TIFFs exist — skip"
fi

# Build manifest (require images) + robustness eval
dep_arg=""
if [[ ${#deps[@]} -gt 0 ]]; then
  dep_str=$(IFS=:
    echo "${deps[*]}")
  dep_arg="--dependency=afterany:${dep_str}"
fi

manifest_job=$(sbatch --parsable ${dep_arg} --wrap="set -euo pipefail
  cd '${REPO_ROOT}'
  source /home/zhangtin/miniforge3/etc/profile.d/conda.sh 2>/dev/null || true
  conda activate '${CONDA_ENV:-pix2pix_cuda}' 2>/dev/null || true
  python scripts/build_hemit_robustness_manifest.py --require-images --test-epoch '${TEST_EPOCH}'
  export HEMIT_MANIFEST='eval/hemit/manifest.csv'
  export REFERENCE_MODEL='pix2pix'
  export ROBUST_OUTDIR='${ROBUST_OUTDIR}'
  python scripts/run_hemit_robustness_eval.py \
    --manifest eval/hemit/manifest.csv \
    --outdir '${ROBUST_OUTDIR}' \
    --reference-model pix2pix
  echo 'Robustness CSVs → ${ROBUST_OUTDIR}'
")

echo ""
echo "Submitted robustness job: ${manifest_job}"
if [[ -n "${dep_arg}" ]]; then
  echo "  (runs after test jobs: ${deps[*]})"
fi
echo ""
echo "Outputs when done:"
echo "  ${ROBUST_OUTDIR}/extended/leaderboard_extended_metrics.csv"
echo "  ${ROBUST_OUTDIR}/extended/paired_comparison_vs_reference.csv"
echo "  ${ROBUST_OUTDIR}/downstream/downstream_leaderboard.csv"
echo "  per-model: ${ROBUST_OUTDIR}/<model>/extended/*.csv"
echo "             ${ROBUST_OUTDIR}/<model>/downstream/*.csv"
