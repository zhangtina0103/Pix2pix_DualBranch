#!/usr/bin/env bash
# Sweep CFG guidance scale for joint_cfg_v2 @ TEST_EPOCH (GPU node / interactive).
# Saves score_cfg_w<scale>.csv under results/<TRAIN_NAME>/test_<epoch>/
#
#   cd Pix2pix_DualBranch && bash scripts/sweep_fm_cfg_scale.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export MODEL=vanilla_fm
export GPU_IDS="${GPU_IDS:-0}"
export DISPLAY_ID=-1
export HEMIT_SRC="${HEMIT_SRC:-/home/zhangtin/virtual-staining/data/hemit}"
export DATAROOT="${DATAROOT:-./datasets/hemit}"
export TEST_EPOCH="${TEST_EPOCH:-100}"
export NUM_TEST="${NUM_TEST:-100}"

# shellcheck source=/dev/null
source "${ROOT}/bash_scripts/_vanilla_fm_sbatch_preamble.sh"
source "${ROOT}/scripts/vanilla_fm_env.sh"
vanilla_fm_apply_joint_cfg_v2_env
export VANILLA_FM_ENV_LOCKED=1

CKPT="${ROOT}/checkpoints/${TRAIN_NAME}/${TEST_EPOCH}_net_G.pth"
[[ -f "${CKPT}" ]] || {
  echo "ERROR: missing ${CKPT}" >&2
  exit 1
}

OUT_DIR="${ROOT}/results/${TRAIN_NAME}/test_${TEST_EPOCH}"
mkdir -p "${OUT_DIR}"

for w in 1.0 1.05 1.1 1.15 1.2; do
  echo "===== CFG scale w=${w} ====="
  export FM_CFG_SCALE="${w}"
  export MODE=test
  bash "${ROOT}/scripts/run_hemit_all.sh"
  export MODE=metrics
  bash "${ROOT}/scripts/run_hemit_all.sh"
  tag="${w//./}"
  cp -f "${OUT_DIR}/score.csv" "${OUT_DIR}/score_cfg_w${tag}.csv"
  echo "w=${w} -> ${OUT_DIR}/score_cfg_w${tag}.csv"
done

echo "Done. Compare average_ssim column in score_cfg_w*.csv"
