#!/usr/bin/env bash
# Upload one HEMIT run to Hugging Face (zhangtin/fixed/<hf_folder>/).
#
#   huggingface-cli login   # once
#   HF_REPO=zhangtin/fixed MODEL=pix2pix bash scripts/upload_hemit_ckpt_to_hf.sh
#   HF_REPO=zhangtin/fixed MODEL=vanilla_fm bash scripts/upload_hemit_ckpt_to_hf.sh
#
# Optional disk cleanup after upload:
#   FREE_DISK=1 HF_REPO=zhangtin/fixed MODEL=cut bash scripts/upload_hemit_ckpt_to_hf.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

HF_REPO="${HF_REPO:-zhangtin/fixed}"
TEST_EPOCH="${TEST_EPOCH:-80}"
FREE_DISK="${FREE_DISK:-0}"

export MODEL="${MODEL:?Set MODEL=pix2pix|pix2pixhd|cut|asp|cyclegan|vanilla_fm}"

case "${MODEL}" in
  pix2pix)    HF_FOLDER=pix2pix;    export MODEL=pix2pix ;;
  pix2pixhd)  HF_FOLDER=pix2pixhd;  export MODEL=pix2pixhd ;;
  cut)        HF_FOLDER=cut;        export MODEL=cut ;;
  asp)        HF_FOLDER=asp;        export MODEL=asp ;;
  cyclegan)   HF_FOLDER=cyclegan;   export MODEL=cyclegan ;;
  vanilla_fm)
    HF_FOLDER=vanilla_fm_joint_perc
    source "${ROOT}/bash_scripts/_vanilla_fm_sbatch_preamble.sh"
    vanilla_fm_apply_joint_perc_env
    export VANILLA_FM_ENV_LOCKED=1
    ;;
  *)
    echo "ERROR: unknown MODEL=${MODEL}" >&2
    exit 1
    ;;
esac

if [[ "${MODEL}" != "vanilla_fm" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/scripts/hemit_model_profiles.sh"
fi

CKPT_DIR="${ROOT}/checkpoints/${TRAIN_NAME}"
G_CKPT="${CKPT_DIR}/${TEST_EPOCH}_net_G.pth"
SCORE="${ROOT}/results/${TRAIN_NAME}/test_${TEST_EPOCH}/images/score.csv"

[[ -f "${G_CKPT}" ]] || { echo "ERROR: missing ${G_CKPT}" >&2; exit 1; }

command -v huggingface-cli >/dev/null 2>&1 || {
  echo "ERROR: huggingface-cli not found. pip install huggingface_hub && huggingface-cli login" >&2
  exit 1
}

echo "Upload ${TRAIN_NAME} → ${HF_REPO}/${HF_FOLDER}/"
huggingface-cli upload "${HF_REPO}" "${G_CKPT}" "${HF_FOLDER}/${TEST_EPOCH}_net_G.pth"

if [[ -f "${CKPT_DIR}/latest_net_G.pth" ]]; then
  huggingface-cli upload "${HF_REPO}" "${CKPT_DIR}/latest_net_G.pth" "${HF_FOLDER}/latest_net_G.pth"
fi

for d_ckpt in "${CKPT_DIR}"/*_net_D.pth; do
  [[ -f "${d_ckpt}" ]] || continue
  base=$(basename "${d_ckpt}")
  huggingface-cli upload "${HF_REPO}" "${d_ckpt}" "${HF_FOLDER}/${base}"
done

if [[ -f "${SCORE}" ]]; then
  huggingface-cli upload "${HF_REPO}" "${SCORE}" "${HF_FOLDER}/score.csv"
fi

echo "Done: https://huggingface.co/${HF_REPO}/tree/main/${HF_FOLDER}"

if [[ "${FREE_DISK}" == "1" ]]; then
  echo "FREE_DISK=1: removing local test images (keeps score.csv + checkpoints on HF)"
  rm -rf "${ROOT}/results/${TRAIN_NAME}/test_${TEST_EPOCH}/images/"*.png \
         "${ROOT}/results/${TRAIN_NAME}/test_${TEST_EPOCH}/images/"*.tif 2>/dev/null || true
  # Uncomment to also drop local .pth after HF upload:
  # rm -f "${CKPT_DIR}"/*.pth
fi
