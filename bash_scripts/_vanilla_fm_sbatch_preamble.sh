# shellcheck shell=bash
# Source after bash_scripts/_common.sh (REPO_ROOT must be set).
# Clears login-shell / prior-job FM flags before vanilla_fm_apply_*_env.
#
# Do NOT submit with: sbatch --export=ALL ...

# shellcheck source=/dev/null
source "${REPO_ROOT:?source _common.sh before preamble}/scripts/vanilla_fm_env.sh"
vanilla_fm_clear_stale_env
unset TRAIN_NAME PRETRAINED_NAME FM_STEPS FM_CHANNELS FM_UP_MODE
unset FM_USE_CFG FM_USE_FILM FM_USE_GAN FM_USE_ODE_TRAIN
unset FM_NULL_MODE FM_FILM_WHERE FM_FILM_REG FM_CFG_SCALE FM_CFG_DROPOUT
unset VANILLA_FM_EXPECTED_TRAIN_NAME
