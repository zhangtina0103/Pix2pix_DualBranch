# HEMIT FM — improvement plan (after cond_consistent @95)

## Where we are

| Run | Avg SSIM | Notes |
|-----|----------|--------|
| **joint_perc @80** | **0.831** | FM baseline to beat |
| **cond_consistent @95** (finetune) | **0.827** | DAPI 0.75, CD3 0.89, panCK 0.84 — **−0.004** vs perc |
| pix2pix | ~0.854 | SSIM target |

Finetune from `joint_perc/80` at **lr=5e-5** got close but **panCK lagged**. Full scratch co-trains seg + init + ODE-aux from epoch 1 at **lr=2e-4**.

## Priority (pick 1–2 GPUs)

### A — **cond_consistent_scratch** (main bet)

Same recipe as finetune, **no perc/80 seed**, epochs **1→130**.

```bash
git pull
bash scripts/verify_fm_cond_env.sh consistent_scratch
sbatch bash_scripts/train_hemit_cond_fm_consistent_scratch.sbatch
# test @ 130 (or best val epoch):
export TEST_EPOCH=130
sbatch bash_scripts/test_hemit_cond_fm_consistent_scratch.sbatch
```

`TRAIN_NAME=hemit_cond_fm_consistent_scratch` — **48h** walltime.

### B — **cond_consistent_v2** (faster epochs)

8-step ODE-aux (vs 12), perc=0.15, λ_sample=22. Use if scratch is too slow or as second GPU.

```bash
sbatch bash_scripts/train_hemit_cond_fm_consistent_v2.sbatch
```

### C — **beat_p2p_111_scratch** (SSIM-aligned FM)

Trains on **ODE sample L1** + strong perc; **1,1,1** weights. Different hypothesis than cond.

```bash
sbatch bash_scripts/train_hemit_vanilla_fm_beat_p2p_111_scratch.sbatch
```

OOM: `FM_PERC_SIZE=256 FM_SAMPLE_L1_PROB=0.25`.

### D — **Resume** old finetune (cheap, low upside)

Only if you want a “@130” row without a full scratch:

```bash
export RESUME_FROM_EPOCH=105
sbatch bash_scripts/resume_hemit_cond_fm_consistent.sbatch
export TEST_EPOCH=130
sbatch bash_scripts/test_hemit_cond_fm_consistent.sbatch
```

## Do not repeat

- PatchNCE / FiLM finetune from perc  
- `train_hemit_cond_fm_consistent.sbatch` from scratch (copies perc/80 again)  
- `sbatch --export=ALL`

## Success

- **Beat FM baseline:** avg SSIM **> 0.831**, weights **1,1,1**  
- **Beat pix2pix:** avg SSIM **≥ 0.854** (hard; beat_p2p track is the SSIM-focused attempt)
