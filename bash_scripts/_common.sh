# Shared setup for SLURM/bash entrypoints.
# Submit .sbatch from Pix2pix_DualBranch repo root:
#   sbatch bash_scripts/run_hemit_reproduce.sbatch

_submit="${SLURM_SUBMIT_DIR:-$PWD}"
if [[ -f "${_submit}/bash_scripts/_common.sh" ]]; then
  REPO_ROOT="${_submit}"
elif [[ -f "${_submit}/_common.sh" ]]; then
  REPO_ROOT="$(cd "${_submit}/.." && pwd)"
else
  echo "ERROR: cannot find bash_scripts/_common.sh (submit from repo root?)" >&2
  exit 1
fi

cd "$REPO_ROOT"
export PYTHONUNBUFFERED=1

# Seg masks: verify only in train/test jobs — never auto-regenerate.
# One-time prep: python scripts/generate_hemit_seg_masks.py --dataroot ./datasets/hemit
# Force regen:    FORCE_HEMIT_SEG=1 sbatch ...
hemit_seg_masks_ready() {
  local dataroot="$1"
  local suffix="${2:-${FM_SEG_SUFFIX:-}}"
  local split n_a n_seg seg_dir
  for split in train val test; do
    seg_dir="${dataroot}/${split}Seg${suffix}"
    [[ -d "${dataroot}/${split}A" && -d "${seg_dir}" ]] || return 1
    n_a=$(ls -1 "${dataroot}/${split}A" 2>/dev/null | wc -l | tr -d ' ')
    n_seg=$(ls -1 "${seg_dir}" 2>/dev/null | wc -l | tr -d ' ')
    [[ "${n_a}" -gt 0 && "${n_seg}" -ge "${n_a}" ]] || return 1
  done
  return 0
}

hemit_require_seg_masks() {
  local dataroot="${1:-./datasets/hemit}"
  local suffix="${2:-${FM_SEG_SUFFIX:-}}"
  dataroot="${REPO_ROOT}/${dataroot#./}"
  local seg_label="Seg${suffix}"
  if [[ "${FORCE_HEMIT_SEG:-0}" == "1" ]]; then
    echo "FORCE_HEMIT_SEG=1: regenerating *${seg_label} under ${dataroot}"
    local gen_args=(--dataroot "${dataroot}")
    [[ -n "${suffix}" ]] && gen_args+=(--suffix "${suffix}")
    [[ "${FM_SEG_METHOD:-}" == "cellpose" || "${suffix}" == "_cellpose" ]] && gen_args+=(--method cellpose)
    python "${REPO_ROOT}/scripts/generate_hemit_seg_masks.py" "${gen_args[@]}"
    return 0
  fi
  if hemit_seg_masks_ready "${dataroot}" "${suffix}"; then
    echo "Seg masks OK (${dataroot}, *${seg_label}): train/val/test match *A — no regeneration"
    return 0
  fi
  echo "ERROR: seg masks missing or incomplete under ${dataroot} (*${seg_label})" >&2
  echo "  Generate once:" >&2
  echo "    python scripts/generate_hemit_seg_masks.py --dataroot ${dataroot}${suffix:+ --suffix ${suffix}}" >&2
  echo "  Intentional regen only: FORCE_HEMIT_SEG=1 sbatch ..." >&2
  for split in train val test; do
    local n_a n_seg
    n_a=$(ls -1 "${dataroot}/${split}A" 2>/dev/null | wc -l | tr -d ' ')
    n_seg=$(ls -1 "${dataroot}/${split}Seg${suffix}" 2>/dev/null | wc -l | tr -d ' ')
    echo "    ${split}: A=${n_a}  ${seg_label}=${n_seg}" >&2
  done
  exit 1
}

# Deprecated alias — same as hemit_require_seg_masks (never auto-regenerates).
hemit_maybe_generate_seg_masks() {
  hemit_require_seg_masks "$@"
}
