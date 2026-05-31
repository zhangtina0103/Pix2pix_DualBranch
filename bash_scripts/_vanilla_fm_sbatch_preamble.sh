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
unset FM_USE_SEG FM_FLOW_PATH FM_INIT_FROM_COND FM_INIT_NOISE_SIGMA DATASET_MODE
unset FM_HE_PROJ_INIT FM_BRIDGE_X0_SIGMA FM_BRIDGE_NOISE_PROB
unset FM_SAMPLE_L1_PROB FM_TRAIN_SAMPLE_STEPS FM_TRAIN_SAMPLE_METHOD
unset VANILLA_FM_EXPECTED_TRAIN_NAME VANILLA_FM_COND_PROFILE
