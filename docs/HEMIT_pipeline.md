# HEMIT models in Pix2pix_DualBranch

## Two backends (one command: `scripts/run_hemit_all.sh`)

### 1. This repo’s pix2pix (`train.py` → `test.py` → `post_process.py`)

Joint **3-channel** multiplex out. Same pipeline you used for dual-branch.

| `MODEL` | Generator |
|---------|-----------|
| `dualbranch` | `SwinTResnet` (paper) |
| `pix2pix` | `resnet_9blocks` (classic pix2pix baseline) |
| `pix2pixhd` | `unet_1024` (high-res U-Net; not NVIDIA multi-scale HD) |
| `cut`, `asp`, `cyclegan` | same `train.py` / `test.py` / `post_process.py`, joint 3ch |
| `resnet6`, `unet256`, … | other `--netG` in `networks.py` |

```bash
MODEL=dualbranch MODE=all bash scripts/run_hemit_all.sh
MODEL=pix2pix MODE=test|metrics TEST_EPOCH=80 bash scripts/run_hemit_all.sh
MODEL=cut MODE=train|test|metrics bash scripts/run_hemit_all.sh
MODEL=asp MODE=train bash scripts/run_hemit_all.sh
MODEL=cyclegan MODE=train bash scripts/run_hemit_all.sh
```

Outputs: `results/<name>/test_<epoch>/images/score.csv` (+ optional `composite_rgb/`)

### 2. Flow matching in `hemit/` (optional)

| `MODEL` | Train script |
|---------|----------------|
| `fm` | `hemit/training/flow_matching_adapted.py` |
| `fm_plus` | `hemit/training/flow_matching_hemit_plus.py` (joint 3ch) |

```bash
MODEL=fm_plus MODE=train bash scripts/run_hemit_all.sh   # uses hemit_compare env
```

**Env:** `pix2pix_cuda` for native models; `hemit_compare` only for `fm` / `fm_plus`.

## SLURM

```bash
MODEL=dualbranch sbatch bash_scripts/run_hemit_all.sbatch
MODEL=cut MODE=train sbatch bash_scripts/run_hemit_all.sbatch
MODEL=fm_plus MODE=train CONDA_ENV=hemit_compare sbatch bash_scripts/run_hemit_all.sbatch
```

Legacy `hemit/training/train_baseline.py` (per-marker) is deprecated; use native `MODEL=cut|asp|cyclegan` instead.
