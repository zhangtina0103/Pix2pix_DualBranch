# HNSCC paths on ORCD Engaging (override before sourcing if needed).
#
# Raw tiles:  HNSCC_SRC → folder with Case1/, Case2/, ... *.png
# Processed:   DATAROOT  → trainA/B, valA/B, testA/B (from prepare_hnscc.py)
#
# Put raw data on scratch (not home — ~200G quota):
#   ~/orcd/scratch/hnscc/raw/Case1/ ...
#   ~/orcd/scratch/hnscc/raw/Case2/ ...

if [[ -z "${HNSCC_SCRATCH_ROOT:-}" ]]; then
  if [[ -n "${SCRATCH:-}" ]]; then
    HNSCC_SCRATCH_ROOT="${SCRATCH}/hnscc"
  else
    HNSCC_SCRATCH_ROOT="${HOME}/orcd/scratch/hnscc"
  fi
fi

export HNSCC_SCRATCH_ROOT
export HNSCC_SRC="${HNSCC_SRC:-${HNSCC_SCRATCH_ROOT}/raw}"
export HNSCC_ZIP="${HNSCC_ZIP:-${HNSCC_SCRATCH_ROOT}/hnscc.zip}"
export DATAROOT="${DATAROOT:-${HNSCC_SCRATCH_ROOT}/datasets/hnscc}"
export HNSCC_MODE="${HNSCC_MODE:-max}"
export HNSCC_OUTPUT_NC="${HNSCC_OUTPUT_NC:-4}"
export HNSCC_MARKERS="${HNSCC_MARKERS:-CD3,CD8,FoxP3,PanCK}"
