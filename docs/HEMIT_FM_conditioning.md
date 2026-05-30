# HEMIT FM — rich conditioning (no manual seg)

Baseline: `hemit_vanilla_fm_joint_perc` epoch **80** → **~0.831** avg SSIM.

All three designs **finetune** from that checkpoint (`strict=False` for new stem channels / `he_proj`).

## 0. One-time: pseudo seg masks

```bash
cd ~/Pix2pix_DualBranch
python scripts/generate_hemit_seg_masks.py --dataroot ./datasets/hemit
# or from raw HEMIT:
python scripts/generate_hemit_seg_masks.py --src /path/to/hemit --dst ./datasets/hemit
```

Creates `trainSeg/`, `valSeg/`, `testSeg/` (1ch, Otsu on H&E nuclei-ish signal).

## 1. `joint_seg` — concat conditioning

- UNet input: `[x_t | H&E(3) | seg(1) | t]` (8ch stem vs 7ch baseline).
- `--dataset_mode aligned_cond` + `--fm_use_seg`
- Train: `sbatch bash_scripts/train_hemit_vanilla_fm_joint_seg.sbatch`
- Test: `sbatch bash_scripts/test_hemit_vanilla_fm_joint_seg.sbatch`

## 2. `joint_bridge` — image-to-image bridge path

- Train path: `x_t = (1-t)·proj(H&E) + t·x1` (not Gaussian `x0`).
- Test: ODE starts at `proj(H&E)` (learnable 1×1 conv, identity init).
- Train: `sbatch bash_scripts/train_hemit_vanilla_fm_joint_bridge.sbatch`
- Test: `sbatch bash_scripts/test_hemit_vanilla_fm_joint_bridge.sbatch`

## 3. `joint_init_cond` — informed noise start

- Still Gaussian flow training; ODE init: `z0 = σ·noise + (1-σ)·proj(H&E)` with **σ=0.3**.
- Train: `sbatch bash_scripts/train_hemit_vanilla_fm_joint_init_cond.sbatch`
- Test: `sbatch bash_scripts/test_hemit_vanilla_fm_joint_init_cond.sbatch`

## Sanity

After train epoch **81** (copy-only resume), test with `TEST_EPOCH=81` should be near **0.831** for bridge/init (minimal arch change). Seg may dip slightly until seg channel learns.

## Metrics

`results/<TRAIN_NAME>/test_<epoch>/score.csv` via `post_process.py`. Compare **avg SSIM** with channel weights **1,1,1**.

## Suggested order

1. **bridge** (no extra data)
2. **init_cond** (no extra data)
3. **seg** (run mask script first)

Run up to three GPU jobs in parallel. Do **not** use `sbatch --export=ALL`.

## Env safety (nothing overrides the recipe)

1. **Preamble** (`_vanilla_fm_sbatch_preamble.sh`) clears stale `FM_*`, `DATASET_MODE`, and `TRAIN_NAME` before each job.
2. **`VANILLA_FM_ENV_LOCKED=1`** skips `vanilla_fm_apply_train_env` and profile FM defaults (`FM_STEPS=50`, etc.).
3. **`vanilla_fm_verify_locked_env`** aborts if `TRAIN_NAME`, `FM_BACKBONE`, or conditioning flags disagree with `VANILLA_FM_COND_PROFILE`.
4. **Login-node check** (no GPU):

```bash
bash scripts/verify_fm_cond_env.sh bridge
bash scripts/verify_fm_cond_env.sh init_cond
bash scripts/verify_fm_cond_env.sh seg
```

5. **In job logs**, confirm these lines near train start:

```text
vanilla_fm config: locked=1 TRAIN_NAME=hemit_vanilla_fm_joint_bridge
  infer: steps=25  heun
  cond_profile=bridge (verified when locked=1)
    FM conditioning CLI: dataset_mode=aligned seg=0 flow=bridge init_cond=0
```

If you see `locked=0`, `steps=50`, `FM_USE_CFG=1`, or wrong `cond_profile`, **cancel the job** and resubmit without `--export=ALL`.
