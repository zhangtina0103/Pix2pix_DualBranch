#!/bin/bash
# Reference: submit CaMSC supplementary benchmark (6 models).
# Does NOT auto-submit — copy/paste commands below on cluster.
#
# Prerequisites:
#   export CAMSC_KFOLD_ROOT=~/orcd/scratch/camsc/datasets/camsc_bf_kfold_aug
#   export CAMSC_ENABLE_AUG=1
#   hemit_fm_cross_attn_scratch_512/80_net_G.pth exists
#   DiffVS HEMIT stage-2 ckpt (DIFFVS_CHECKPOINT_DIR)
#
# === ALREADY DONE (your checkpoints/) ===
#   pix2pix folds 0-4 @ camsc_bf_pix2pix_fold*_512_aug_512
#
# === RECOMMENDED ORDER ===
#
# 1. Diagnostics (parallel, ~1h each)
#   CAMSC_EVAL_FOLDS=0 sbatch bash_scripts/eval_camsc_bf_fm_cross_attn_zeroshot.sbatch
#   sbatch bash_scripts/eval_camsc_bf_diffvs_zeroshot.sbatch
#
# 2. Cross-attn finetune pilot (~6h) — gate before full FM + GAN sweep
#   sbatch bash_scripts/train_camsc_bf_fm_cross_attn_finetune_fold0.sbatch
#   CAMSC_EVAL_FOLDS=0 TEST_EPOCH=110 sbatch bash_scripts/eval_camsc_bf_fm_cross_attn_finetune.sbatch
#
# 3. If fold0 FM-ft beats pix2pix → train all folds + GAN baselines (parallel)
#   sbatch bash_scripts/train_camsc_bf_fm_cross_attn_finetune_kfold_array.sbatch
#   CAMSC_MODEL=cut      sbatch -J camsc_cut  bash_scripts/train_camsc_bf_gan_kfold_array.sbatch
#   CAMSC_MODEL=asp      sbatch -J camsc_asp  bash_scripts/train_camsc_bf_gan_kfold_array.sbatch
#   CAMSC_MODEL=cyclegan sbatch -J camsc_cg   bash_scripts/train_camsc_bf_gan_kfold_array.sbatch
#
# 4. Eval after training (~2h each)
#   TEST_EPOCH=80 sbatch bash_scripts/eval_camsc_bf_pix2pix.sbatch
#   TEST_EPOCH=110 sbatch bash_scripts/eval_camsc_bf_fm_cross_attn_finetune.sbatch
#   CAMSC_MODEL=cut      TEST_EPOCH=80 sbatch bash_scripts/eval_camsc_bf_gan.sbatch
#   CAMSC_MODEL=asp      TEST_EPOCH=80 sbatch bash_scripts/eval_camsc_bf_gan.sbatch
#   CAMSC_MODEL=cyclegan TEST_EPOCH=80 sbatch bash_scripts/eval_camsc_bf_gan.sbatch
#
# Metrics live under: ${CAMSC_KFOLD_ROOT}/eval/<model>/kfold_aggregate.json
#
# Models: pix2pix | fm_cross_attn_ft | cut | asp | cyclegan | diffvs_zeroshot
# (skip fm_cross_attn scratch — use ft for paper row)

echo "This file is a runbook only. See comments above." >&2
exit 0
