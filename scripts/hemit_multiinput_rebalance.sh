#!/usr/bin/env bash
# Rebalance finished multi-input runs: same checkpoint @80, finetune with fair 1,1,0 weights.
#
# Use when training finished under old FM_CHANNEL_WEIGHTS=2,1,0 (CD3 was 2× L1 vs panCK).
# Keeps TRAIN_NAME / architecture / dataroot; extends training 81→95 at low LR.
#
#   HEMIT_MULTI_VARIANT=he bash scripts/hemit_multiinput_rebalance.sh
#   HEMIT_MULTI_VARIANT=dapi bash scripts/hemit_multiinput_rebalance.sh
#
# Cluster:
#   HEMIT_MULTI_VARIANT=he sbatch bash_scripts/finetune_hemit_multiinput_rebalance.sbatch
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export MODEL=vanilla_fm
export MODE=train
export CONTINUE_TRAIN=1
export HEMIT_MULTI_VARIANT="${HEMIT_MULTI_VARIANT:-he}"

# shellcheck source=/dev/null
source "${ROOT}/scripts/hemit_multiinput_env.sh"
hemit_multiinput_apply_env
export VANILLA_FM_ENV_LOCKED=1

# Fair weights (hemit_multiinput_env pins 1,1,0 on [CD3, panCK, pad])
export FM_CHANNEL_WEIGHTS=1,1,0

src_epoch="${RESUME_FROM_EPOCH:-80}"
extra="${HEMIT_MULTI_REBALANCE_EPOCHS:-15}"
end_epoch=$((src_epoch + extra))

export RESUME_FROM_EPOCH="${src_epoch}"
export EPOCH_COUNT=$((src_epoch + 1))
export N_EPOCHS="${end_epoch}"
export N_EPOCHS_DECAY=0
export TRAIN_LR="${TRAIN_LR:-5e-5}"
export LR_DECAY_ITERS="${LR_DECAY_ITERS:-10}"
export SAVE_EPOCH_FREQ="${SAVE_EPOCH_FREQ:-5}"

# shellcheck source=/dev/null
source "${ROOT}/scripts/hemit_model_profiles.sh"
export VANILLA_FM_EXPECTED_TRAIN_NAME="${TRAIN_NAME}"
vanilla_fm_verify_locked_env

ckpt="${CHECKPOINTS_DIR}/${TRAIN_NAME}/${src_epoch}_net_G.pth"
[[ -f "${ckpt}" ]] || {
  echo "ERROR: missing ${ckpt}" >&2
  echo "  Finish @${src_epoch} first, or set RESUME_FROM_EPOCH to your latest epoch." >&2
  exit 1
}

vanilla_fm_print_train_env
echo "===== multi-input rebalance: variant=${HEMIT_MULTI_VARIANT} ${TRAIN_NAME} ====="
echo "  load@${src_epoch} → train epochs ${EPOCH_COUNT}..${end_epoch}"
echo "  FM_CHANNEL_WEIGHTS=${FM_CHANNEL_WEIGHTS}  TRAIN_LR=${TRAIN_LR}"
echo "  eval when done: TEST_EPOCH=${end_epoch} HEMIT_MULTI_VARIANT=${HEMIT_MULTI_VARIANT} sbatch bash_scripts/eval_hemit_multiinput.sbatch"

exec bash "${ROOT}/scripts/run_hemit_vanilla_fm.sh"
