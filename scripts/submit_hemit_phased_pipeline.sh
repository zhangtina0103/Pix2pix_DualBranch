#!/usr/bin/env bash
# Phased HEMIT pipeline (saves disk vs train+eval all at once):
#   1) train 4 models (pix2pix, pix2pixhd, cut, asp)
#   2) eval first 2
#   3) train 2 models (cyclegan, vanilla_fm joint_perc)
#   4) eval cut, asp
#   5) eval cyclegan, vanilla_fm
#   6) upload all checkpoints to Hugging Face
#
# Prereqs: datasets/hemit OK (0 broken symlinks), huggingface-cli login for step 6
#
#   cd ~/Pix2pix_DualBranch && git pull
#   export HEMIT_SRC=/home/zhangtin/HEMIT
#   bash scripts/submit_hemit_phased_pipeline.sh
#   bash scripts/submit_hemit_phased_pipeline.sh --upload-only   # after manual train/eval
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p logs

export HEMIT_SRC="${HEMIT_SRC:-/home/zhangtin/HEMIT}"
UPLOAD_ONLY=0
[[ "${1:-}" == "--upload-only" ]] && UPLOAD_ONLY=1

if [[ "${UPLOAD_ONLY}" == "0" ]]; then
  echo "HEMIT_SRC=${HEMIT_SRC}"
  broken=$(find datasets/hemit -type l ! -exec test -s {} \; -print 2>/dev/null | wc -l | tr -d ' ')
  if [[ "${broken}" != "0" ]]; then
    echo "ERROR: ${broken} broken symlinks in datasets/hemit — rm -rf datasets/hemit && prep first" >&2
    exit 1
  fi

  J_W1_TR=$(sbatch --parsable bash_scripts/train_hemit_phased_wave1.sbatch)
  echo "Wave1 train (4): job ${J_W1_TR}"

  J_W1A_EV=$(sbatch --parsable --dependency=afterok:"${J_W1_TR}" \
    bash_scripts/eval_hemit_phased_wave1a.sbatch)
  echo "Wave1a eval pix2pix+pix2pixhd (2): job ${J_W1A_EV} (after ${J_W1_TR})"

  J_W2_TR=$(sbatch --parsable --dependency=afterok:"${J_W1_TR}" \
    bash_scripts/train_hemit_phased_wave2.sbatch)
  echo "Wave2 train cyclegan+vanilla_fm (2): job ${J_W2_TR} (after ${J_W1_TR})"

  J_W1B_EV=$(sbatch --parsable --dependency=afterok:"${J_W1_TR}" \
    bash_scripts/eval_hemit_phased_wave1b.sbatch)
  echo "Wave1b eval cut+asp (2): job ${J_W1B_EV} (after ${J_W1_TR})"

  J_W2_EV=$(sbatch --parsable --dependency=afterok:"${J_W2_TR}" \
    bash_scripts/eval_hemit_phased_wave2.sbatch)
  echo "Wave2 eval cyclegan+vanilla_fm (2): job ${J_W2_EV} (after ${J_W2_TR})"

  J_HF=$(sbatch --parsable --dependency=afterok:"${J_W1A_EV}:${J_W1B_EV}:${J_W2_EV}" \
    bash_scripts/upload_hemit_phased_to_hf.sbatch)
  echo "HF upload (6 models): job ${J_HF} (after all evals)"
  echo ""
  echo "Monitor: squeue -u \$USER"
  echo "Resume train if 6h timeout:"
  echo "  MODEL=pix2pix sbatch bash_scripts/resume_hemit_native_full.sbatch"
  echo "  RESUME_PROFILE=joint_perc sbatch bash_scripts/resume_hemit_fm_scratch.sbatch"
else
  J_HF=$(sbatch --parsable bash_scripts/upload_hemit_phased_to_hf.sbatch)
  echo "HF upload only: job ${J_HF}"
fi
