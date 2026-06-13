# Source after _common.sh on Hoechst+CD3e+CD4 Orion panel jobs.
#   export ORION_PANEL=cd3_cd4
#   source bash_scripts/_orion_cd3_cd4_preamble.sh
export ORION_PANEL=cd3_cd4
# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/orion_scratch_env.sh"
export HEMIT_TRAIN_SIZE="${HEMIT_TRAIN_SIZE:-512}"
export ORION_N_TRAIN="${ORION_N_TRAIN:-1480}"
export ORION_N_VAL="${ORION_N_VAL:-500}"
export ORION_N_TEST="${ORION_N_TEST:-500}"
export ORION_TILE_SIZE="${ORION_TILE_SIZE:-512}"

_orion_cd3_cd4_check_dataroot() {
  [[ -d "${DATAROOT}/testA" ]] || {
    echo "ERROR: missing ${DATAROOT}/testA" >&2
    echo "Run: sbatch bash_scripts/prepare_orion_cd3_cd4.sbatch" >&2
    exit 1
  }
}
