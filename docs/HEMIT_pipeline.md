# HEMIT models in Pix2pix_DualBranch

## Two backends (one command: `scripts/run_hemit_all.sh`)

### 1. This repo’s pix2pix (`train.py` → `test.py` → `post_process.py`)

Joint **3-channel** multiplex out. Same pipeline you used for dual-branch.

| `MODEL` | Generator |
|---------|-----------|
| `dualbranch` | `SwinTResnet` (paper) |
| `pix2pix` | `resnet_9blocks` (classic pix2pix baseline) |
| `resnet6`, `unet256`, … | other `--netG` in `networks.py` |

```bash
MODEL=dualbranch MODE=all bash scripts/run_hemit_all.sh
MODEL=pix2pix MODE=test|metrics TEST_EPOCH=80 bash scripts/run_hemit_all.sh
```

Outputs: `results/<name>/test_<epoch>/images/score.csv`

### 2. Comparison models in `hemit/` (our CUT / ASP / CycleGAN / FM code)

Per-marker GAN + flow matching. Native HEMIT paths (`HEMIT_SRC/.../input,label`).

| `MODEL` | Train script |
|---------|----------------|
| `cyclegan`, `cut`, `asp` | `hemit/training/train_baseline.py` |
| `fm` | `hemit/training/flow_matching_adapted.py` |
| `fm_plus` | `hemit/training/flow_matching_hemit_plus.py` |

```bash
MODEL=cut MARKER=CD3 MODE=train bash scripts/run_hemit_all.sh
MODEL=fm MARKER=DAPI MODE=train bash scripts/run_hemit_all.sh
MODEL=comparison MODE=metrics bash scripts/run_hemit_all.sh
```

Outputs: `eval/hemit/baselines/*.csv`, `eval/hemit/unified/by_marker.csv`

**Env:** separate `hemit_compare` (do not install into `pix2pix_cuda`):

```bash
sbatch bash_scripts/setup_hemit_compare_env.sbatch
# or: bash scripts/setup_hemit_compare_env.sh   # on GPU node
```

Default SLURM for GAN/FM arrays: `CONDA_ENV=hemit_compare`.

## SLURM

```bash
MODEL=dualbranch sbatch bash_scripts/run_hemit_all.sbatch
sbatch bash_scripts/train_hemit_gan_array.sbatch   # 9× cyclegan/cut/asp
sbatch bash_scripts/train_hemit_fm_array.sbatch    # 3× fm
MODEL=fm_plus MODE=train sbatch bash_scripts/run_hemit_all.sbatch
MODEL=comparison MODE=metrics sbatch bash_scripts/run_hemit_all.sbatch
```

## What we do *not* use

- vs_v2 per-marker **pix2pix** — replaced by this repo’s `MODEL=pix2pix` (`train.py`).
