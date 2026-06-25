# CaMSC brightfield paths on ORCD Engaging (override before sourcing if needed).
#
#   source scripts/camsc_scratch_env.sh
#
# Raw TIFs:  ~/orcd/scratch/camsc/20260504/
# K-fold:     ~/orcd/scratch/camsc/datasets/camsc_bf_kfold/fold{0..4}/

if [[ -z "${CAMSC_SCRATCH_ROOT:-}" ]]; then
  if [[ -n "${SCRATCH:-}" ]]; then
    CAMSC_SCRATCH_ROOT="${SCRATCH}/camsc"
  else
    CAMSC_SCRATCH_ROOT="${HOME}/orcd/scratch/camsc"
  fi
fi

export CAMSC_SCRATCH_ROOT
export CAMSC_SRC="${CAMSC_SRC:-${CAMSC_SCRATCH_ROOT}/20260504}"
export CAMSC_KFOLD_ROOT="${CAMSC_KFOLD_ROOT:-${CAMSC_SCRATCH_ROOT}/datasets/camsc_bf_kfold}"
export CAMSC_KFOLDS="${CAMSC_KFOLDS:-5}"
export FM_CHANNEL_WEIGHTS="${FM_CHANNEL_WEIGHTS:-1,1,0}"

camsc_fold_dataroot() {
  local fold="${1:?fold index required}"
  echo "${CAMSC_KFOLD_ROOT}/fold${fold}"
}

camsc_fold_train_name() {
  local fold="${1:?fold index required}"
  local model="${CAMSC_MODEL:-vanilla_fm}"
  local size="${HEMIT_TRAIN_SIZE:-512}"
  echo "camsc_bf_${model}_fold${fold}_${size}"
}
