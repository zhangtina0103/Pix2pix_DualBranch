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
export ORION_ZIP="${ORION_ZIP:-${ORION_SCRATCH_ROOT}/ORIONCRC_dataset_tile_20x.zip}"

# ORION_PANEL=hemit  → Hoechst,CD3e,Pan-CK  (datasets/orion_lite)
# ORION_PANEL=immune → CD3e,CD8a,FOXP3       (datasets/orion_immune_cd3_cd8_foxp3)
export ORION_PANEL="${ORION_PANEL:-hemit}"
case "${ORION_PANEL}" in
  immune)
    export ORION_MARKERS="${ORION_MARKERS:-CD3e,CD8a,FOXP3}"
    export DATAROOT="${DATAROOT:-${ORION_SCRATCH_ROOT}/datasets/orion_immune_cd3_cd8_foxp3}"
    ;;
  hemit|*)
    export ORION_MARKERS="${ORION_MARKERS:-Hoechst,CD3e,Pan-CK}"
    export DATAROOT="${DATAROOT:-${ORION_SCRATCH_ROOT}/datasets/orion_lite}"
    ;;
esac
export DATAROOT

ZENODO_ORION_URL="https://zenodo.org/records/15340874/files/ORIONCRC_dataset_tile_20x.zip?download=1"
