# Fast FM / architecture track (avoid slow ODE-in-training)

## Speed ranking (per training step @ 1024²)

| Recipe | Relative cost | Why |
|--------|---------------|-----|
| **joint_perc** / **res3** / **perc_strong** | **1×** | One FM forward + perc; no ODE loop |
| **cond_light** | **~1.15×** | 20% of steps × 4 Heun steps |
| **cond_consistent_v2** | **~1.4×** | 35% × 8 steps |
| **cond_consistent (12-step)** | **~1.6×** | 40% × 12 steps |
| **beat_p2p_111** | **~2–3×** | 50% × 25 steps + heavy perc @ 512 |

**Rule:** For architecture experiments, stay on the **joint_perc training path** (noise FM + perc). Use **cond_light** if you want seg+init without v2 cost. Skip **beat_p2p** unless you have spare GPU weeks.

---

## What to run (priority)

### Already running

- `hemit_cond_fm_consistent_v2` — keep; it’s the full cond bet.

### Fast arch (pick 1–2 GPUs)

| Goal | Job | sbatch |
|------|-----|--------|
| Deeper conv, same speed class | **res3 scratch** | `train_hemit_vanilla_fm_joint_perc_res3_scratch.sbatch` |
| Stronger perc, no ODE | **perc_strong scratch** | `train_hemit_vanilla_fm_perc_strong_scratch.sbatch` |
| Cond but cheaper | **cond_light scratch** | `train_hemit_cond_fm_light_scratch.sbatch` |
| Re-baseline scratch | **joint_perc scratch** | `train_hemit_vanilla_fm_joint_perc_scratch.sbatch` |

All: **1→130**, **1,1,1**, **~24–30h** (similar to original joint_perc train).

### Finetune from perc/80 (only **20 epochs**, ~6h)

If you want quick arch nudges without full scratch:

```bash
sbatch bash_scripts/train_hemit_vanilla_fm_joint_perc_res3.sbatch
```

(Already in repo; loads `joint_perc/80`.)

### Non-FM (fast, best SSIM historically)

```bash
MODEL=dualbranch sbatch bash_scripts/run_hemit_all.sbatch   # SwinTResnet pix2pix
```

---

## Architecture knobs (no new code yet)

| Knob | Where | Note |
|------|--------|------|
| `FM_RESBLOCKS=3` | res3 | More depth, same step cost ~1.1× |
| `FM_LAMBDA_PERC=0.5` | perc_strong | SSIM-ish without ODE loop |
| `FM_CHANNELS=96,192,256` | default custom | Wider = OOM risk @ 1024 |
| `FM_BACKBONE=monai` + `FM_CROP_SIZE=512` | monai512 | **4× fewer pixels**/step; not 1024 fair compare |
| `FM_STEPS=12` at test only | env on test | Faster inference, may drop SSIM |

**Not wired:** NATTEN @ 1024 (docs/HEMIT_FM_mentor_track.md future item).

Tune params on login node:

```bash
python scripts/count_fm_params.py --fm_channels 96,192,256 --fm_num_res_blocks 3
```

---

## Decision tree

1. **Need results this week, fast** → `res3` finetune from perc/80 **or** `perc_strong_scratch`.
2. **Want cond + speed** → `cond_light_scratch` (not beat_p2p).
3. **Can wait 48h** → keep **cond_v2**; cancel **beat_p2p** if queued.
4. **Must beat 0.854 SSIM** → dualbranch/pix2pix row + FM ablations above.

---

## Advanced architecture (tri-head + cross-attn + seg)

**Flagship** (after `git pull`):

```bash
bash scripts/verify_fm_cond_env.sh advanced
sbatch bash_scripts/train_hemit_cond_fm_advanced_scratch.sbatch
```

`TRAIN_NAME=hemit_cond_fm_advanced_scratch` — tri-head, H&E cross-attn @ mid+decoder, seg, init, light ODE.

**Ablations** (isolate what helps):

```bash
sbatch bash_scripts/train_hemit_fm_tri_head_scratch.sbatch      # tri-head only
sbatch bash_scripts/train_hemit_fm_cross_attn_scratch.sbatch    # cross-attn only
```

Test: `export TEST_EPOCH=130 && sbatch bash_scripts/test_hemit_cond_fm_advanced_scratch.sbatch`

OOM: `FM_CROSS_ATTN_HEADS=4` or train without decoder cross-attn (edit env: unset `FM_CROSS_ATTN_DECODER`).

## Success

Beat **0.831** (`joint_perc`) with **≤1.2×** joint_perc step time (advanced ~1.25–1.4×).
