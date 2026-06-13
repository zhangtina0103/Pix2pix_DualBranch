#!/bin/bash
# Diffusion robustness: extended metrics + downstream biology (D-VST, DiffVS, etc.)
#
# Prereq: TIFF pairs exported (e.g. MODEL=dvst MODE=metrics)
#   results/diffusion/dvst/images/*_{real,fake}_B.tif
#
#   export DIFFUSION_MANIFEST=eval/diffusion/manifest.csv
#   sbatch bash_scripts/eval_diffusion_robustness.sbatch
#
# Merge with HEMIT GAN/FM afterward:
#   export CONCAT_HEMIT=1
#   sbatch bash_scripts/eval_diffusion_robustness.sbatch
#SBATCH --job-name=diff_robust
#SBATCH --output=logs/diffusion_robustness_%j.out
#SBATCH --error=logs/diffusion_robustness_%j.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/bash_scripts/_common.sh"
mkdir -p logs eval/diffusion eval/combined

source /home/zhangtin/miniforge3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate "${CONDA_ENV:-pix2pix_cuda}" 2>/dev/null || true

export REFERENCE_MODEL="${REFERENCE_MODEL:-pix2pix}"
export DIFFUSION_OUTDIR="${DIFFUSION_OUTDIR:-eval/diffusion/robustness_comparison}"
export DIFFUSION_MANIFEST="${DIFFUSION_MANIFEST:-eval/diffusion/manifest.csv}"
PLOT_FLAG=()
[[ "${PLOTS:-0}" == "1" ]] && PLOT_FLAG=(--plots)

if [[ "${REBUILD_MANIFEST:-1}" == "1" ]]; then
  python scripts/build_diffusion_robustness_manifest.py --require-images
fi

[[ -f "${DIFFUSION_MANIFEST}" ]] || { echo "ERROR: missing ${DIFFUSION_MANIFEST}" >&2; exit 1; }

python scripts/run_hemit_robustness_eval.py \
  --manifest "${DIFFUSION_MANIFEST}" \
  --outdir "${DIFFUSION_OUTDIR}" \
  --reference-model "${REFERENCE_MODEL}" \
  "${PLOT_FLAG[@]}"

if [[ "${CONCAT_HEMIT:-0}" == "1" ]]; then
  [[ -f eval/hemit/manifest.csv ]] || python scripts/build_hemit_robustness_manifest.py --require-images --test-epoch "${TEST_EPOCH:-80}"
  python scripts/concat_robustness_manifests.py \
    --manifest eval/hemit/manifest.csv \
    --manifest "${DIFFUSION_MANIFEST}" \
    --out eval/combined/manifest.csv
  python scripts/run_hemit_robustness_eval.py \
    --manifest eval/combined/manifest.csv \
    --outdir "${COMBINED_OUTDIR:-eval/combined/robustness_comparison}" \
    --reference-model "${REFERENCE_MODEL}" \
    "${PLOT_FLAG[@]}"
fi

echo "Done. Diffusion → ${DIFFUSION_OUTDIR}"
