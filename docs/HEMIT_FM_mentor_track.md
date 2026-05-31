# HEMIT FM — mentor track (joint_perc + one change)

**Floor:** `hemit_vanilla_fm_joint_perc` @ **80** → **0.831** avg SSIM.  
**Target:** ~**0.854** (pix2pix).  
**Rule:** Same FM recipe as joint_perc (noise path, x1+perc, conv_transpose decoder, 25 Heun). Change **one** thing per run. Finetune **81→100** from `joint_perc/80` unless noted.

Do **not** use `decoder_only` (bilinear) — that ablation already failed vs conv_transpose perc.

## Runs

| Job | `TRAIN_NAME` | Change | Finetune from perc/80? |
|-----|----------------|--------|-------------------------|
| **Cellpose seg** | `hemit_vanilla_fm_joint_perc_cellpose` | Better masks → `trainSeg_cellpose` | Yes |
| **PatchNCE** | `hemit_vanilla_fm_joint_perc_patchnce` | CUT-style structure loss H&E ↔ x1_hat | Yes (new NCE head) |
| **Res3** | `hemit_vanilla_fm_joint_perc_res3` | 3 ResBlocks/level (deeper conv) | Yes (strict=False) |
| **MONAI 512** | `hemit_vanilla_fm_monai512` | MONAI UNet, 512² crop, mid attention | **No** — train from scratch |

`cond_consistent` / bridge / init-only are deprioritized; use this table instead.

## Cluster commands

```bash
cd ~/Pix2pix_DualBranch
git pull
test -f checkpoints/hemit_vanilla_fm_joint_perc/80_net_G.pth && echo OK

# 1) Cellpose masks (slow; once)
pip install cellpose   # if needed
python scripts/generate_hemit_seg_masks.py --dataroot ./datasets/hemit \
  --method cellpose --suffix _cellpose

bash scripts/verify_fm_cond_env.sh mentor_cellpose   # after we add alias - use:
# bash -c 'source scripts/vanilla_fm_env.sh; vanilla_fm_apply_joint_perc_cellpose_env; ...'

sbatch bash_scripts/train_hemit_vanilla_fm_joint_perc_cellpose.sbatch
sbatch bash_scripts/train_hemit_vanilla_fm_joint_perc_patchnce.sbatch
sbatch bash_scripts/train_hemit_vanilla_fm_joint_perc_res3.sbatch

# Optional: new backbone (longer, no perc seed)
sbatch bash_scripts/train_hemit_vanilla_fm_monai512.sbatch
```

Tests @ **100** (default `TEST_EPOCH`):

```bash
sbatch bash_scripts/test_hemit_vanilla_fm_joint_perc_cellpose.sbatch
sbatch bash_scripts/test_hemit_vanilla_fm_joint_perc_patchnce.sbatch
sbatch bash_scripts/test_hemit_vanilla_fm_joint_perc_res3.sbatch
```

## Verify env (login node)

```bash
bash scripts/verify_fm_cond_env.sh cellpose   # maps to mentor_cellpose
bash scripts/verify_fm_cond_env.sh patchnce
bash scripts/verify_fm_cond_env.sh res3
```

## Future (not wired yet)

- **NATTEN** window attention in custom U-Net @ 1024² — needs `natten` + new module; try after PatchNCE/cellpose scores.
- **SEU-Net** — separate backbone port.
- **Full CUT** (`MODEL=cut`) — different generator; compare as non-FM row, not perc finetune.
