# Orion paths on ORCD scratch (override before sourcing if needed).
# Home quota ~200G — do NOT download 118G Orion zip to home.

ORION_SCRATCH_ROOT="${ORION_SCRATCH_ROOT:-/orcd/scratch/orcd/002/zhangtin/orion}"
export ORION_DATA_DIR="${ORION_DATA_DIR:-${ORION_SCRATCH_ROOT}}"
export ORION_SRC="${ORION_SRC:-${ORION_SCRATCH_ROOT}/ORIONCRC_dataset_tile_20x}"
export DATAROOT="${DATAROOT:-${ORION_SCRATCH_ROOT}/datasets/orion_lite}"
export ORION_ZIP="${ORION_ZIP:-${ORION_SCRATCH_ROOT}/ORIONCRC_dataset_tile_20x.zip}"

ZENODO_ORION_URL="https://zenodo.org/records/15340874/files/ORIONCRC_dataset_tile_20x.zip?download=1"
