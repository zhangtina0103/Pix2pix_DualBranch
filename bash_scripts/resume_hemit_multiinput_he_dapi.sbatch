#!/bin/bash
# Resume HEMIT multi-input he + dapi (two jobs, auto-detect latest checkpoint each).
#
#   sbatch bash_scripts/resume_hemit_multiinput_he_dapi.sbatch
#
# Single variant:
#   HEMIT_MULTI_VARIANT=he sbatch bash_scripts/resume_hemit_multiinput.sbatch
#   HEMIT_MULTI_VARIANT=dapi sbatch bash_scripts/resume_hemit_multiinput.sbatch
#
# Delayed (now+6h / now+12h):
#   bash bash_scripts/schedule_hemit_multiinput_resume.sbatch
#SBATCH --job-name=hemit_multi_res2
#SBATCH --output=logs/hemit_multiinput_resume_he_dapi_%j.out
#SBATCH --error=logs/hemit_multiinput_resume_he_dapi_%j.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --time=00:05:00
#SBATCH --mem=1G
#SBATCH --cpus-per-task=1

set -euo pipefail
cd "$(dirname "$0")/.."

for variant in he dapi; do
  echo "==> queue resume: ${variant}"
  sbatch \
    --job-name="hemit_${variant}_res" \
    --export=ALL,HEMIT_MULTI_VARIANT="${variant}" \
    bash_scripts/resume_hemit_multiinput.sbatch
done

echo ""
echo "Queued 2 resume jobs (he + dapi). Check: squeue -u \$USER"
