# HEMIT vanilla FM+ phased experiments

**Baseline (do not overwrite):** `hemit_vanilla_fm_joint_perc` @ epoch **80** → **0.831** avg SSIM (100 tiles, 25-step Heun).

**Rule:** One phase at a time; `git pull`; never `sbatch --export=ALL`.

## Env locking (no overrides)

Phase sbatch scripts:

1. Source `bash_scripts/_vanilla_fm_sbatch_preamble.sh` (clears stale `FM_*` / `TRAIN_NAME`)
2. Call `vanilla_fm_apply_*_env` (hard-pins recipe)
3. `export VANILLA_FM_ENV_LOCKED=1`

With `LOCKED=1`, `hemit_model_profiles.sh` does **not** re-apply FM defaults (`64,128,192`, `FM_STEPS=50`, etc.). `vanilla_fm_verify_locked_env` exits if `TRAIN_NAME` or `FM_BACKBONE` drift.

**Log check** (first lines after train starts):

```text
vanilla_fm config: locked=1 TRAIN_NAME=hemit_vanilla_fm_joint_gan
  infer: steps=25  heun
```

If you see `locked=0`, `steps=50`, or wrong `TRAIN_NAME`, stop the job and fix env before trusting metrics.

| Phase | Goal | Train | Test |
|-------|------|-------|------|
| **0** | Baseline | (done) `joint_perc` | epoch 80 |
| **3** | CFG v2 (learned null, w≈1.1) — **prefer for new datasets** | `sbatch bash_scripts/train_hemit_vanilla_fm_joint_cfg_v2.sbatch` | `sbatch bash_scripts/test_hemit_vanilla_fm_joint_cfg_v2.sbatch` |
| **3b** | CFG `w` sweep | — | `bash scripts/sweep_fm_cfg_scale.sh` (after phase 3 train) |
| **4** | FiLM v2 (head, reg, 1,1,1) — **prefer for new datasets** | `sbatch bash_scripts/train_hemit_vanilla_fm_joint_film_v2.sbatch` | `sbatch bash_scripts/test_hemit_vanilla_fm_joint_film_v2.sbatch` |
| **1** | FM + PatchGAN (HEMIT sharpness; less portable) | `sbatch bash_scripts/train_hemit_vanilla_fm_joint_gan.sbatch` | `sbatch bash_scripts/test_hemit_vanilla_fm_joint_gan.sbatch` |
| ~~**2**~~ | ~~CD3-weighted L1~~ | **skipped** | — |

**Suggested order for cross-dataset work:** 0 → **cond** (seg / bridge / init) → **3** → **4** → (optional) **1** on HEMIT only.

See **`docs/HEMIT_FM_conditioning.md`** for the three conditioning runs (finetune from `joint_perc/80`).

## Phase details

### Phase 1 — `joint_gan`
- Finetune from `joint_perc/80` → epochs 81–100 (copies **G only**; **D** init random — expected)
- `FM_USE_GAN=1`, ODE sample steps=12, prob=0.35
- No CFG / FiLM

### Phase 3 — `joint_cfg_v2`
- `FM_NULL_MODE=learned` (init = **mean H&E**, not zeros), dropout **0.12**, test **w=1.1**
- Finetune LR **1e-4**, epochs 81–100 from `joint_perc/80`
- **Sanity:** `TEST_EPOCH=81` + `FM_CFG_SCALE=1.0` should be ~**0.831** (no guidance, FiLM/CFG barely moved G)

### Phase 4 — `joint_film_v2`
- `FM_FILM_WHERE=head`, `FM_FILM_REG=0.01`, `FM_CHANNEL_WEIGHTS=1,1,1`
- LR **5e-5**, 15+5 epochs → 100 total from 80

## Success criteria
Beat **0.831** avg SSIM on 100 tiles; **dapi_ssim** should stay **≥ ~0.75**.

## v1 vs v2 (why v2 might help)

| Issue (v1) | CFG v2 fix | FiLM v2 fix |
|------------|------------|-------------|
| Null = black H&E | Learned null, **mean H&E init** | N/A |
| w=1.5 too strong | **w=1.1** + sweep script | N/A |
| Decoder FiLM biased panCK | N/A | **Head-only** + **identity** at init (γ=1, β=0) |
| No reg → drift | N/A | **FM_FILM_REG=0.01** |
| 1,2,1 unfair weights | 1,1,1 | **1,1,1** |
| Full LR 20 ep | LR **1e-4**, dropout **0.12** | LR **5e-5**, 20 ep |

**Honest ceiling:** FiLM v2 is **more likely** to beat 0.831 (starts as joint_perc). CFG v2 may help slightly; concat H&E every step still limits CFG. Neither is guaranteed.

## v1 failures (reference)
- `joint_cfg` (zero null, w=1.5): ~0.726
- `joint_film` (decoder FiLM): ~0.755, dapi ~0.52
