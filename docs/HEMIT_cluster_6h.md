# HEMIT training with 6h SLURM max

Partition caps jobs at **`#SBATCH --time=06:00:00`**. Plan **train → resume → resume** until epoch **130**.

## Epochs per 6h (rule of thumb)

| Recipe | ~epochs / 6h | Jobs for 1→130 |
|--------|----------------|----------------|
| **joint_perc** / tri_head / perc_strong | ~35–45 | **3–4** |
| **cond_v2** / advanced / light | ~22–28 | **5–6** |
| **beat_p2p** | ~15–20 | **7+** (avoid) |

Your cond finetune hit **~26 epochs in 6h** (81→106) — use that for planning.

## Workflow

### 1) First chunk (scratch)

```bash
git pull
sbatch bash_scripts/train_hemit_cond_fm_advanced_scratch.sbatch
# or: train_hemit_cond_fm_consistent_v2.sbatch, etc.
```

### 2) After timeout — resume (same `TRAIN_NAME`)

```bash
ls checkpoints/hemit_cond_fm_advanced_scratch/*_net_G.pth | tail -3
export RESUME_PROFILE=advanced
# optional: export RESUME_FROM_EPOCH=25
sbatch bash_scripts/resume_hemit_fm_scratch.sbatch
```

**Re-submit** until you have `130_net_G.pth` (or best val epoch).

### `RESUME_PROFILE` names

| Profile | `TRAIN_NAME` |
|---------|----------------|
| `advanced` | `hemit_cond_fm_advanced_scratch` |
| `v2` | `hemit_cond_fm_consistent_v2` |
| `consistent_scratch` | `hemit_cond_fm_consistent_scratch` |
| `light` | `hemit_cond_fm_light_scratch` |
| `tri_head` | `hemit_fm_tri_head_scratch` |
| `cross_attn` | `hemit_fm_cross_attn_scratch` |
| `perc_strong` | `hemit_vanilla_fm_perc_strong_scratch` |
| `res3` | `hemit_vanilla_fm_joint_perc_res3_scratch` |
| `consistent` | `hemit_cond_fm_consistent` (finetune from 80) |

### 3) Test

```bash
export TEST_EPOCH=130
sbatch bash_scripts/test_hemit_cond_fm_advanced_scratch.sbatch
```

## Do not

- Re-run the **train** sbatch after partial progress (re-seeds or copies perc/80).
- Use `sbatch --export=ALL`.

## cond_v2 already running

When the 6h job dies, note last epoch (e.g. 25), then:

```bash
export RESUME_PROFILE=v2
export RESUME_FROM_EPOCH=25
sbatch bash_scripts/resume_hemit_fm_scratch.sbatch
```

Repeat ~5× to reach 130.
