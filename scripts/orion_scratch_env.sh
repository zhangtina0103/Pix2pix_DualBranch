# Orion paths on ORCD Engaging scratch (override before sourcing if needed).
# Home quota ~200G — do NOT download 118G Orion zip to home.
#
# ORCD scratch: ~/orcd/scratch → /orcd/scratch/orcd/NNN/$USER (1 TB, not backed up)
# Always use ~/orcd/scratch — do not hardcode /orcd/scratch/orcd/002/...

if [[ -z "${ORION_SCRATCH_ROOT:-}" ]]; then
  if [[ -n "${SCRATCH:-}" ]]; then
    ORION_SCRATCH_ROOT="${SCRATCH}/orion"
  else
    ORION_SCRATCH_ROOT="${HOME}/orcd/scratch/orion"
  fi
fi

export ORION_SCRATCH_ROOT
export ORION_DATA_DIR="${ORION_DATA_DIR:-${ORION_SCRATCH_ROOT}}"
export ORION_SRC="${ORION_SRC:-${ORION_SCRATCH_ROOT}/ORIONCRC_dataset_tile_20x}"
export DATAROOT="${DATAROOT:-${ORION_SCRATCH_ROOT}/datasets/orion_lite}"
export ORION_ZIP="${ORION_ZIP:-${ORION_SCRATCH_ROOT}/ORIONCRC_dataset_tile_20x.zip}"

ZENODO_ORION_URL="https://zenodo.org/records/15340874/files/ORIONCRC_dataset_tile_20x.zip?download=1"
