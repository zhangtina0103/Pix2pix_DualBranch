# Diffusion baselines on HEMIT (512²)

External diffusion methods integrated via `MODEL=diffvs|dvst` in `scripts/run_hemit_all.sh`.

## Important: parameter count

| Method | Trainable backbone | ~Params | Fair vs 11M GAN? |
|--------|-------------------|---------|------------------|
| pix2pix / CUT / ASP | ResNet9 G | **~11M** | Yes (internal sweep) |
| **DiffVS** | SD 2.1 UNet (finetuned) | **~860M** | No — pretrained diffusion |
| **D-VST** | PixArt-XL DiT (+ optional attn_adapter) | **~610M / ~730M** | No — pretrained diffusion |

Report diffusion rows in a **separate table** or with a footnote. They answer “can diffusion beat GAN/FM on HEMIT?” not “same-capacity generator.”

## Repos (clone beside Pix2pix_DualBranch)

```text
varanasi_group/
  Pix2pix_DualBranch/   ← this repo (orchestrator)
  DiffVS/               ← git clone hvcl/DiffVS fork with HEMIT scripts
  D-VST/                ← git clone https://github.com/yangshurong/D-VST
```

Set on cluster:

```bash
export DIFFVS_ROOT=/home/zhangtin/DiffVS
export DVST_ROOT=/home/zhangtin/D-VST
export HEMIT_SRC=/home/zhangtin/HEMIT
export IMAGE_SIZE=512
```

## DiffVS (recommended first)

Uses upstream `DiffVS/scripts/train_hemit_*.sh` unchanged. Two stages:

1. **Stage 1** — Marigold latent DDPM (100 epochs default)
2. **Stage 2** — Diffusion-FT @ t=999 (5 epochs)

```bash
# Train (24h walltime)
sbatch bash_scripts/train_hemit_diffvs.sbatch

# Eval test → score.csv
sbatch bash_scripts/eval_hemit_diffvs.sbatch
```

Scores: `results/diffusion/diffvs/score.csv` (same `post_process.py` as CUT/ASP).

**VRAM @ 512²:** start `DIFFVS_TRAIN_BS=4`; OOM → `2`.

**Pretrained:** `Manojb/stable-diffusion-2-1-base` (SD 2.1 mirror).

## D-VST

Uses upstream `D-VST/train.py` and `eval.py`. Finetune from `HE2mIHC.ckpt`:

```bash
# Separate env (Python 3.10, see D-VST/requirement.txt)
conda create -n D_VST python=3.10 -y
conda activate D_VST
pip install -r ${DVST_ROOT}/requirement.txt
# Download weights per D-VST README → weights/dvst_pretrained/

sbatch bash_scripts/train_hemit_dvst.sbatch   # 48h
sbatch bash_scripts/eval_hemit_dvst.sbatch
```

Configs: `configs/dvst/train1_HEMIT.yaml`, `train2_HEMIT.yaml` (512×512 native patches).

**D-VST inference note:** eval uses **blurred GT mIHC** as CLIP tone reference (`DVST_REF_MODE=paired_gt`, same as upstream `eval.py`). This is optimistic vs true blind HE→mIHC; document in paper. Set `DVST_REF_MODE=he_only` to ablate (weaker).

**512 patch fix:** applied in `D-VST/models/dvst_modules/dataset.py` (random crop when `img_resolution == resolution`).

## Data prep (both)

```bash
python scripts/prepare_hemit_for_diffusion.py --src ${HEMIT_SRC} --format both
```

Creates:

- `datasets/hemit_diffvs/{train,val,test}/{input,label}/`
- `datasets/hemit_dvst/HE/<slide>/`, `mIHC/<slide>/`

Or via pipeline:

```bash
MODEL=diffvs MODE=prepare bash scripts/run_hemit_all.sh
MODEL=dvst MODE=prepare bash scripts/run_hemit_all.sh
```

## Expected improvement?

From your 512² GAN sweep:

| Model | Avg SSIM | Avg Pearson |
|-------|----------|-------------|
| ASP / CUT | ~0.86 / **0.73** | Strong CD3 |
| pix2pix | 0.86 / 0.56 | Broken CD3 |
| CycleGAN | 0.76 / 0.66 | Weakest |

DiffVS/D-VST may improve **visual quality / PanCK** via pretrained priors but are **not guaranteed** to beat ASP on Pearson (especially CD3). Run DiffVS first — lighter integration, already has HEMIT loader.

## Quick commands

```bash
MODEL=diffvs MODE=prepare|train|test|metrics bash scripts/run_hemit_all.sh
MODEL=dvst  MODE=prepare|train|test|metrics bash scripts/run_hemit_all.sh
```

## Robustness eval (extended metrics + downstream)

Same pipeline as GAN/FM once TIFF pairs exist (`*_real_B.tif` / `*_fake_B.tif`):

| Model | TIFF export dir |
|-------|-----------------|
| D-VST | `results/diffusion/dvst/images/` |
| DiffVS | `results/diffusion/diffvs/images/` (symlinked from inference `pix2pix_metrics/`) |

```bash
# D-VST only (after MODE=test|metrics)
python scripts/build_diffusion_robustness_manifest.py --require-images
python scripts/run_hemit_robustness_eval.py \
  --manifest eval/diffusion/manifest.csv \
  --outdir eval/diffusion/robustness_comparison \
  --reference-model pix2pix

# Or sbatch (GPU for LPIPS)
sbatch bash_scripts/eval_diffusion_robustness.sbatch

# Merge GAN/FM + diffusion for one leaderboard
python scripts/concat_robustness_manifests.py \
  --manifest eval/hemit/manifest.csv \
  --manifest eval/diffusion/manifest.csv \
  --out eval/combined/manifest.csv
python scripts/run_hemit_robustness_eval.py \
  --manifest eval/combined/manifest.csv \
  --outdir eval/combined/robustness_comparison \
  --reference-model pix2pix
```

Add a second diffusion baseline tomorrow: export TIFFs to `results/diffusion/<name>/images/`, add a row to `eval/diffusion/paper_models.csv`, rebuild manifest, re-concat.

## YOLO CD3 downstream (mentor recommendation)

Train a **frozen** YOLO detector on **real** HEMIT CD3 stains only; at test, run the same weights on `real_B` (reference) vs each model's `fake_B` (prediction). Fair across generators: one detector, no per-model tuning, never train on generated images.

**Labels:** pseudo-boxes from CD3+ nuclei on train/val `label` TIFFs (same nucleus logic as downstream biology).

```bash
# 1) Export YOLO dataset + train (GPU)
export HEMIT_ROOT=/path/to/hemit   # or DATAROOT=./datasets/hemit
sbatch bash_scripts/train_hemit_yolo_cd3.sbatch

# 2) Eval all paper models on test TIFFs
export YOLO_WEIGHTS=weights/yolo_cd3_hemit.pt
export HEMIT_MANIFEST=eval/hemit/manifest.csv
sbatch bash_scripts/eval_hemit_yolo_downstream.sbatch
```

Outputs: `eval/hemit/yolo_downstream/yolo_cd3_leaderboard.csv` (count MAE, F1, precision, recall with bootstrap CI).

**Paper framing:** supplementary downstream — complements Pearson/per-cell metrics; cross-attn may not win on sparse detection counts.
