# Orion-Lite paths for DiffVS / D-VST (zero-shot eval on scratch).
# Source from run_orion_dvst.sh — not directly.

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/orion_scratch_env.sh"
# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/hemit_diffusion_env.sh"

export ORION_DATAROOT="${ORION_DATAROOT:-${DATAROOT}}"
export IMAGE_SIZE="${IMAGE_SIZE:-512}"

# D-VST / DiffVS staging on scratch (not home)
export DVST_DATA_ROOT="${DVST_DATA_ROOT:-${ORION_SCRATCH_ROOT}/datasets/orion_lite_dvst}"
export DIFFVS_DATA_ROOT="${DIFFVS_DATA_ROOT:-${ORION_SCRATCH_ROOT}/datasets/orion_lite_diffvs}"
export DIFFUSION_RESULTS_ROOT="${DIFFUSION_RESULTS_ROOT:-${REPO_ROOT}/results/orion_lite_diffusion}"
export DVST_INFER_PATTERN="${DVST_INFER_PATTERN:-infer_ORION_lite_test-*}"
