#!/usr/bin/env bash
# Shared env for DiffVS + D-VST diffusion baselines on HEMIT @ 512².
# Source from run_hemit_diffvs.sh / run_hemit_dvst.sh (not directly).

# Repo roots (override on cluster)
export DIFFVS_ROOT="${DIFFVS_ROOT:-${REPO_ROOT:-}/../DiffVS}"
export DVST_ROOT="${DVST_ROOT:-${REPO_ROOT:-}/../D-VST}"

export HEMIT_SRC="${HEMIT_SRC:-/home/zhangtin/HEMIT}"
export HEMIT_TRAIN_SIZE="${HEMIT_TRAIN_SIZE:-512}"
export IMAGE_SIZE="${IMAGE_SIZE:-${HEMIT_TRAIN_SIZE}}"

# DiffVS layout: {root}/{train,val,test}/{input,label}/
export DIFFVS_DATA_ROOT="${DIFFVS_DATA_ROOT:-${REPO_ROOT:-}/datasets/hemit_diffvs}"
export DIFFVS_OUTPUT_ROOT="${DIFFVS_OUTPUT_ROOT:-${DIFFVS_ROOT}/outputs/hemit512}"
export DIFFVS_STAGE1_DIR="${DIFFVS_STAGE1_DIR:-${DIFFVS_OUTPUT_ROOT}/stage1}"
export DIFFVS_STAGE2_DIR="${DIFFVS_STAGE2_DIR:-${DIFFVS_OUTPUT_ROOT}/stage2}"
export DIFFVS_INFER_DIR="${DIFFVS_INFER_DIR:-${DIFFVS_OUTPUT_ROOT}/inference}"
export DIFFVS_PRETRAINED="${DIFFVS_PRETRAINED:-Manojb/stable-diffusion-2-1-base}"
export DIFFVS_STAGE1_EPOCH="${DIFFVS_STAGE1_EPOCH:-100}"
export DIFFVS_STAGE2_EPOCH="${DIFFVS_STAGE2_EPOCH:-5}"

# D-VST layout: {root}/HE/{slide}/* and mIHC/{slide}/*
export DVST_DATA_ROOT="${DVST_DATA_ROOT:-${REPO_ROOT:-}/datasets/hemit_dvst}"
export DVST_WEIGHTS="${DVST_WEIGHTS:-${DVST_ROOT}/weights/dvst_pretrained}"
export DVST_CKPT="${DVST_CKPT:-${DVST_WEIGHTS}/HE2mIHC.ckpt}"
export DVST_OUTPUT_ROOT="${DVST_OUTPUT_ROOT:-${DVST_ROOT}/TrainResult/hemit512}"
export DVST_INFER_DIR="${DVST_INFER_DIR:-${DVST_ROOT}/DVST_samples/hemit512_eval}"

export DIFFVS_TRAIN_BS="${DIFFVS_TRAIN_BS:-4}"
export DIFFVS_STAGE1_EPOCHS="${DIFFVS_STAGE1_EPOCHS:-100}"
export DIFFVS_STAGE2_EPOCHS="${DIFFVS_STAGE2_EPOCHS:-5}"

export DVST_TRAIN_BS="${DVST_TRAIN_BS:-2}"
export DVST_VIDEO_LENGTH="${DVST_VIDEO_LENGTH:-4}"
export DVST_MAX_EPOCH="${DVST_MAX_EPOCH:-100}"

# Results symlinked here for unified lookup
export DIFFUSION_RESULTS_ROOT="${DIFFUSION_RESULTS_ROOT:-${REPO_ROOT:-}/results/diffusion}"

hemit_diffusion_die() { echo "ERROR: $*" >&2; exit 1; }

hemit_diffusion_check_diffvs() {
  [[ -d "${DIFFVS_ROOT}" ]] || hemit_diffusion_die "DiffVS not found at ${DIFFVS_ROOT}. Set DIFFVS_ROOT."
  [[ -f "${DIFFVS_ROOT}/src/diffvs/train_stage1_marigold.py" ]] || hemit_diffusion_die "Invalid DiffVS root: ${DIFFVS_ROOT}"
}

hemit_diffusion_check_dvst() {
  [[ -d "${DVST_ROOT}" ]] || hemit_diffusion_die "D-VST not found at ${DVST_ROOT}. Set DVST_ROOT."
  [[ -f "${DVST_ROOT}/train.py" ]] || hemit_diffusion_die "Invalid D-VST root: ${DVST_ROOT}"
}
