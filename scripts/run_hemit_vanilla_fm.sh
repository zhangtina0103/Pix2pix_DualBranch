#!/usr/bin/env bash
# Vanilla flow matching only: train.py → test.py → post_process.py (joint 3ch).
#
# Fresh train required after sample-L1 fix (rm old checkpoints or new TRAIN_NAME):
#   rm -rf checkpoints/hemit_vanilla_fm_joint
#   TRAIN_NAME=hemit_vanilla_fm_joint MODE=train sbatch bash_scripts/train_hemit_vanilla_fm.sbatch
#   MODEL=vanilla_fm MODE=test TEST_EPOCH=80 bash scripts/run_hemit_vanilla_fm.sh
#   # still soft? FM_TEST_STEPS=100 MODE=test ...
#
# Cluster:
#   sbatch bash_scripts/train_hemit_vanilla_fm.sbatch
#
# Tune size (~11.38M like resnet_9blocks G):
#   python scripts/count_fm_params.py --fm_channels 96,192,256 --fm_num_res_blocks 2
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export MODEL=vanilla_fm
export FM_CHANNELS="${FM_CHANNELS:-104,208,288}"
export FM_RESBLOCKS="${FM_RESBLOCKS:-2}"
export FM_STEPS="${FM_STEPS:-25}"
export FM_LAMBDA_L1="${FM_LAMBDA_L1:-10}"
export FM_LAMBDA_SAMPLE_L1="${FM_LAMBDA_SAMPLE_L1:-100}"
export FM_SAMPLE_L1_PROB="${FM_SAMPLE_L1_PROB:-1.0}"

# shellcheck source=/dev/null
source "${ROOT}/scripts/hemit_model_profiles.sh"
if [[ "${PY_MODEL}" != "vanilla_fm" ]]; then
  echo "ERROR: expected PY_MODEL=vanilla_fm, got PY_MODEL=${PY_MODEL}" >&2
  exit 1
fi
echo "vanilla_fm profile: PY_MODEL=${PY_MODEL} TRAIN_NAME=${TRAIN_NAME}"
echo "  FM_CHANNELS=${FM_CHANNELS} FM_RESBLOCKS=${FM_RESBLOCKS} FM_STEPS=${FM_STEPS}"
echo "  FM_STEPS=${FM_STEPS} FM_SAMPLE_METHOD=${FM_SAMPLE_METHOD:-heun} FM_LAMBDA_SAMPLE_L1=${FM_LAMBDA_SAMPLE_L1}"

exec bash "${ROOT}/scripts/run_hemit_all.sh"
