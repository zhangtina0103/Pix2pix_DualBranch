# Beating pix2pix with conditional flow matching

## Table roles (fixed)

| Row | Name | What it is |
|-----|------|------------|
| **Vanilla FM baseline** | `hemit_vanilla_fm_joint_perc` @ **80** | **0.831** — noise path, x1+perc, no extra cond |
| **Non-FM baseline** | pix2pix ResNet9 | **~0.854** — target to beat |
| **Our method** | `hemit_cond_fm_*` | FM + **right conditioning + consistency** |

`vanilla_fm` in code = framework class only. **joint_perc is the vanilla FM number.**

## Why earlier conditioning looked “broken”

| Run | Problem |
|-----|---------|
| init_cond v1 | **Inconsistent**: train on noise `x₀`, test on 70% H&E-colored `z₀` → green soup |
| bridge v1 | Wrong `x₀` semantics + train path change |
| GAN | Not conditioning — hurts DAPI |
| seg alone @ 100 | **Consistent** but under-tuned → 0.789 |

**Your intuition is right:** conditioning must be **consistent** (same information at train and test) and **semantically correct** (gray `he_proj`, not identity).

## Hypothesis A — `hemit_cond_fm_consistent` (run this first)

**Idea:** Keep **joint_perc** FM core; add:

1. **Seg concat** in UNet (rich cond, portable)
2. **Informed ODE init v2** (gray proj, σ=0.55) at **test**
3. **Train/test consistency**: light **ODE sample L1** from the **same** `z₀` as test (12-step Heun, prob 0.4) so `G` learns that sampling path

Still **not** GAN / CFG / bridge train.

```bash
git pull
bash scripts/verify_fm_cond_env.sh consistent   # add to verify script if needed
sbatch bash_scripts/train_hemit_cond_fm_consistent.sbatch
# test @ 130
sbatch bash_scripts/test_hemit_cond_fm_consistent.sbatch
```

## Phased ablations (parallel GPUs, 81→130 from joint_perc/80)

| Phase | `TRAIN_NAME` | What |
|-------|----------------|------|
| **A** | `hemit_cond_fm_seg_only` | Seg concat only (no init, no ODE-aux) |
| **B** | `hemit_cond_fm_init_only` | Informed z₀ + ODE-aux (no seg) — fixes init_cond v2 mismatch |
| **C** | `hemit_cond_fm_consistent` | Full stack (A+B) |

```bash
git pull
bash scripts/verify_fm_cond_env.sh init_only   # or seg_only
sbatch bash_scripts/train_hemit_cond_fm_init_only.sbatch   # Phase B (priority)
sbatch bash_scripts/train_hemit_cond_fm_seg_only.sbatch    # Phase A (optional)
# tests @ 130 when done
sbatch bash_scripts/test_hemit_cond_fm_init_only.sbatch
sbatch bash_scripts/test_hemit_cond_fm_seg_only.sbatch
```

Compare all to **0.831** (perc) and **~0.854** (pix2pix).

## Hypothesis B — better seg (if A is close but &lt; 0.854)

- Cellpose or stronger pseudo masks → regenerate `trainSeg/`
- Retrain `cond_consistent` from same `joint_perc/80`

## Hypothesis C — bridge v2 **only if** train/test both use bridge path

- `joint_bridge_v2` (gray proj + 35% noise mix at train)
- Score before combining with seg

## What we are **not** claiming

- Kitchen-sink `joint_opt` (perc+vel+everything) — dropped as main story
- GAN row as FM
- init_cond v1 / bridge v1 numbers

## Success

**avg SSIM ≥ 0.854** on 100 tiles, weights **1,1,1**, with paper text:

> Conditional flow matching with **structure-aware conditioning** and **consistent flow initialization** outperforms vanilla FM (0.831) and pix2pix (0.854) on HEMIT joint staining.
