#!/bin/bash
# Schedule delayed HEMIT multi-input eval (default: he_dapi @80 in 2h).
#
#   bash bash_scripts/schedule_hemit_multiinput_eval.sbatch
#
# All variants @80 in 2h:
#   HEMIT_EVAL_VARIANTS=he,dapi,he_dapi bash bash_scripts/schedule_hemit_multiinput_eval.sbatch
#
# Custom delay / epoch:
#   SLURM_BEGIN=now+3hours TEST_EPOCH=80 HEMIT_EVAL_VARIANTS=he_dapi bash ...
#
#   SLURM_BEGIN=now+2hours  (default)
#   HEMIT_EVAL_VARIANTS=he_dapi  (default — only variant likely needing eval)

set -euo pipefail
cd "$(dirname "$0")/.."

BEGIN="${SLURM_BEGIN:-now+2hours}"
TEST_EPOCH="${TEST_EPOCH:-80}"
VARIANTS="${HEMIT_EVAL_VARIANTS:-he_dapi}"

IFS=',' read -r -a _variants <<< "${VARIANTS}"

echo "Scheduling HEMIT multi-input eval"
echo "  begin=${BEGIN}  TEST_EPOCH=${TEST_EPOCH}  variants=${VARIANTS}"

for variant in "${_variants[@]}"; do
  variant="$(echo "${variant}" | xargs)"
  [[ -n "${variant}" ]] || continue
  echo "  queue eval: ${variant}"
  sbatch --begin="${BEGIN}" \
    --job-name="hemit_${variant}_ev" \
    --export=ALL,HEMIT_MULTI_VARIANT="${variant}",TEST_EPOCH="${TEST_EPOCH}" \
    bash_scripts/eval_hemit_multiinput.sbatch
done

echo ""
echo "Queued ${#_variants[@]} eval job(s). Check: squeue -u \$USER --start"
