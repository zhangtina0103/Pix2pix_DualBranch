# HEMIT Multiplex Virtual Staining — Methods & Results

**Task:** Translate H&E histology to multiplex immunofluorescence (DAPI, CD3, panCK) on the HEMIT dataset.  
**Proposed model:** Flow matching (FM) with H&E cross-attention at the U-Net bottleneck.  
**Evaluation:** 945 held-out test tiles, 512×512 patches, epoch-80 checkpoints unless noted.

---

## Methods

### Dataset and preprocessing

We use the **HEMIT** paired H&E ↔ multiplex IF dataset with official splits: **3,717 training**, **630 validation**, and **945 test** tiles. Each tile is a **512×512** patch (center-cropped from 1024×1024 source imagery). Ground-truth IF is stored as a three-channel image with channel order **DAPI (nuclei), CD3 (T cells), panCK (epithelial cells)**. H&E input is RGB. All models are trained and evaluated on the **same splits and file list** so comparisons are paired at the tile level.

For visualization, multiplex composites follow standard IF coloring: **panCK = red, CD3 = green, DAPI = blue**.

### Unified experimental framework

All methods were implemented in an extended **Pix2pix_DualBranch** codebase that provides a **shared data loader**, **512×512 patch pipeline**, **80-epoch training schedule**, and **identical test-set evaluation script**. Each model predicts **all three IF channels jointly in one forward pass** (multi-channel output), rather than training separate single-marker models. This removes confounds from different preprocessing, augmentation, or evaluation code across baselines.

**Training schedule.** Models train for **80 epochs** with validation on the 630-tile validation set (monitoring only; hyperparameters fixed). Checkpoints at epoch 80 are used for all benchmark numbers on the 945-tile test set.

**Input/output normalization.** Images are mapped to a consistent floating range for training (typically [−1, 1] for GAN baselines; FM uses the same convention for source/target). At evaluation, predictions and ground truth are converted to comparable intensity scales before metric computation.

### Baseline methods

We compare against five established virtual-staining approaches trained under the unified pipeline:

1. **pix2pix** — Conditional GAN with U-Net generator and PatchGAN discriminator (ResNet-9 backbone, 512²).
2. **CUT** — Contrastive unpaired translation with patch-wise contrastive loss (joint multi-channel variant).
3. **CycleGAN** — Cycle-consistency GAN for unpaired image translation (joint multi-channel variant).
4. **ASP** — Adversarial staining prediction baseline adapted to HEMIT joint 3-channel output.
5. **DiffVS** — marker-wise conditioned diffusion virtual staining baseline (DiffVS, AAAI 2026; evaluated on HEMIT test tiles).

Each baseline uses the **same H&E → {DAPI, CD3, panCK}** pairing, patch size, epoch budget, and test harness as our method.

### Vanilla flow-matching baseline

Our generative backbone is **conditional flow matching (FM)** with a **U-Net velocity field** \(v_\theta(x_t, t \mid \text{H&E})\).

**Forward process.** For target IF image \(x_1\) and noise \(x_0 \sim \mathcal{N}(0, I)\), we sample timestep \(t \in (0,1)\) from a **logit-normal** distribution (emphasizing mid-timesteps). The linear interpolation path is \(x_t = (1-t)x_0 + t x_1\). The model predicts the **velocity** \(v = x_1 - x_0\) (or equivalent FM target) conditioned on H&E.

**Loss.** Training minimizes **L1 flow-matching loss** on the predicted velocity plus a **joint perceptual loss** (λ = 0.1) computed on all three output channels together, encouraging structural fidelity across the full multiplex panel.

**Inference.** We integrate the learned ODE from \(t=0\) to \(t=1\) using a **Heun (2nd-order) solver** with **25 steps**, starting from Gaussian noise and conditioning on the H&E input at every step.

**Architecture.** The vanilla FM baseline uses a standard **conditional U-Net**: H&E is encoded and injected via concatenation / feature conditioning through the decoder; no cross-attention.

This vanilla FM model is the **reference ablation** (not a row in the main benchmark table against GAN baselines).

### Proposed method: FM + H&E cross-attention

Our final model keeps the **same FM objective, perceptual weighting, timestep sampling, and Heun-25 inference** as vanilla FM, and adds **multi-head cross-attention at the U-Net bottleneck**:

- **Query:** bottleneck features of the FM U-Net at **64×64** spatial resolution.
- **Key/Value:** multi-scale features from a dedicated **H&E encoder** (morphology at multiple resolutions).
- **Mechanism:** 4-head cross-attention; attended features are **added residually** to the bottleneck tensor before decoding.
- **Rationale:** Each predicted IF channel can attend to H&E structures relevant to that marker (e.g., nuclear morphology for DAPI, membrane/cytoplasmic patterns for panCK, lymphocyte-rich regions for CD3) while still being generated jointly.

The model is **trained from scratch for 80 epochs** (no GAN pretraining). No tri-head decoupling is used (`FM_USE_TRI_HEAD=0`); a single joint decoder outputs all three channels.

### Ablations and conditioning screens

On top of vanilla FM and our cross-attention model, we ablate:

| Variant | Description |
|--------|-------------|
| **+ H&E cross-attn** | Proposed architecture (primary result). |
| **+ focal loss (γ=1)** | Focal modulation on FM loss to emphasize hard voxels. |
| **+ CD3 focal** | Focal loss applied only on CD3 channel. |
| **+ seg conditioning** | Auxiliary segmentation map concatenated to H&E input. |
| **+ consistency regularization** | Velocity/consistency regularizer on FM field (evaluated when training completes). |

**Conditioning screens (appendix).** Before full-scale training, we screened **classifier-free guidance (CFG)**, **FiLM**, **segmentation concatenation**, and **PatchNCE** on a **100-tile development subset** (not the first 100 rows of the test CSV—tiles matched by filename). All underperformed vanilla FM on SSIM; **cross-attention was the only variant that improved over vanilla FM** and was confirmed on the full 945-tile test set.

### Evaluation metrics

**Pixel-level (945 tiles).** For each tile and each channel, we compute **Pearson r**, **Spearman ρ**, **SSIM**, **PSNR**, **LPIPS**, **MAE**, and **RMSE**. We report **mean ± standard deviation** across tiles. **Average** columns pool metrics across DAPI, CD3, and panCK unless stated otherwise.

**Statistical testing.** For the main benchmark and ablations, we use **paired two-sided tests on per-tile means** (n = 945 paired tiles). Significance stars on **our model only** when it **significantly outperforms** the relevant comparator (typically **ASP** for the main table, **vanilla FM** for ablations): *** p < 0.001, ** p < 0.01, * p < 0.05. Where means differ slightly but p ≥ 0.05, we report as not significant (ns).

**Downstream biological validation.** On the same 945 tiles:

1. **Per-cell Pearson:** Otsu thresholding on DAPI → cell masks; mean IF intensity per cell compared between prediction and ground truth; Pearson computed per tile then aggregated.
2. **Co-expression error:** Measures whether **CD3⁺** and **panCK⁺** intensities maintain their **inverse spatial relationship** at the cell level (biologically, T cells and epithelial cells rarely co-express strongly in the same cell). Lower is better.

### Implementation and compute

Training uses PyTorch on GPU cluster nodes (H200-class GPUs for long FM runs). Environment profiles and launch scripts live in `scripts/vanilla_fm_env.sh` and `bash_scripts/` (e.g., `hemit_fm_cross_attn_scratch_512`, baseline `hemit_*_512` names). Evaluation: `test.py` → `post_process.py` → per-tile `score.csv` aggregated to paper tables.

---

## Results

### Main benchmark (945 test tiles, epoch 80)

| Model | Avg Pearson | Avg SSIM | Avg PSNR |
|-------|-------------|----------|----------|
| pix2pix | 0.563 ± 0.093 | 0.860 ± 0.062 | 29.02 ± 4.15 |
| CUT | 0.730 ± 0.128 | 0.855 ± 0.064 | 29.96 ± 3.52 |
| CycleGAN | 0.659 ± 0.135 | 0.763 ± 0.072 | 26.88 ± 3.70 |
| ASP | 0.739 ± 0.119 | 0.860 ± 0.064 | 29.95 ± 3.72 |
| DiffVS | 0.609 ± 0.129 | 0.719 ± 0.059 | 25.64 ± 3.88 |
| **Ours (FM + cross-attn)** | **0.750 ± 0.134*** | **0.862 ± 0.051** | **30.04 ± 5.10** |

\* Significantly higher average Pearson vs ASP (paired t-test, p < 0.001, n = 945). Average SSIM is highest for our model (+0.002 vs ASP) but **not statistically significant** vs ASP at α = 0.05; vs CUT and CycleGAN, SSIM improvements are significant. PSNR is significantly higher vs pix2pix, CycleGAN, and DiffVS; **ns vs CUT and ASP**.

**Summary.** Our model achieves the **best average Pearson correlation** and **best average SSIM** among all methods. GAN baselines ASP and CUT are strong competitors on SSIM/PSNR; diffusion baseline DiffVS lags on pixel metrics. Flow matching with H&E cross-attention provides the most consistent gains in **correlation-based** metrics, which align better with per-marker biological readouts.

### Per-marker highlights

| Model | DAPI Pearson | CD3 Pearson | panCK Pearson |
|-------|--------------|-------------|---------------|
| ASP | 0.695 ± 0.257 | 0.565 ± 0.121 | 0.967 ± 0.077 |
| CUT | 0.696 ± 0.253 | 0.541 ± 0.124 | 0.966 ± 0.082 |
| **Ours** | **0.724 ± 0.245** | 0.567 ± 0.140 | **0.976 ± 0.062** |

Our model reaches **best panCK Pearson (0.976 ± 0.062)**, strong **DAPI**, and **competitive CD3** vs the strongest GAN baselines. Full seven-metric per-channel tables are provided in supplementary material.

### Downstream biological validation

| Model | CD3 Pearson (per-cell) | panCK Pearson (per-cell) | Co-expression error |
|-------|------------------------|---------------------------|---------------------|
| CUT | 0.424 ± 0.426 | 0.981 ± 0.040 | 0.244 ± 0.256 |
| ASP | 0.432 ± 0.433 | 0.982 ± 0.069 | 0.248 ± 0.265 |
| **Ours** | 0.424 ± 0.430 | 0.982 ± 0.031 | **0.230 ± 0.255†** |

† Lowest mean co-expression error; **p < 0.05 vs ASP**. Per-cell CD3/panCK Pearson is **competitive with CUT/ASP** (differences not significant at α = 0.05).

Co-expression error captures preservation of **mutually exclusive cell-type staining** in the tumor microenvironment; our model reduces this error while matching top baselines on single-marker per-cell correlation.

### Ablation study (945 tiles, epoch 80)

| Model | Avg Pearson | Avg SSIM | CD3 Pearson | panCK Pearson |
|-------|-------------|----------|-------------|---------------|
| Vanilla FM | 0.736 ± 0.134 | 0.844 ± 0.058 | 0.559 ± 0.129 | 0.970 ± 0.082 |
| **+ H&E cross-attn (ours)** | **0.750 ± 0.134***** | **0.862 ± 0.051***** | **0.567 ± 0.140***** | **0.976 ± 0.062***** |
| + focal γ = 1 | 0.756 ± 0.133 | 0.853 ± 0.055 | 0.589 ± 0.132 | 0.961 ± 0.120 |
| + CD3 focal | 0.736 ± 0.135 | 0.845 ± 0.059 | 0.575 ± 0.137 | 0.956 ± 0.129 |
| + seg conditioning | 0.711 ± 0.126 | 0.825 ± 0.061 | 0.504 ± 0.129 | 0.953 ± 0.122 |
| + consistency reg. | — | — | — | — |

\*\*\* p < 0.001 vs vanilla FM (paired t-test on tile means, n = 945).

**Cross-attention is the primary architectural improvement**, with balanced gains across average and per-marker Pearson and SSIM. **Focal loss (γ = 1)** increases **CD3 Pearson (+0.030 vs vanilla FM)** but **degrades panCK Pearson and average SSIM**—reported as a **tradeoff ablation**, not the deployed model. Segmentation conditioning **hurts** all metrics. Consistency-regularization results pending completion of training/evaluation.

### Conditioning screens (100-tile dev subset, appendix)

| Variant | SSIM (avg) |
|---------|------------|
| Vanilla FM | 0.831 |
| Cross-attention | **0.869** |
| CFG | 0.726 |
| PatchNCE | 0.672 |

Only cross-attention improved over vanilla FM on the dev subset; full 945-tile evaluation confirmed cross-attention gains.

### Qualitative results

Representative test patches (Figure: qualitative comparison) show that **Ours** better preserves **nuclear DAPI**, **CD3⁺ infiltrate**, and **panCK⁺ epithelial** boundaries vs pix2pix and CycleGAN, with fewer structural artifacts than DiffVS. Compared to ASP and CUT, our predictions show **sharper cell-type contrast** in regions with overlapping morphology in H&E.

**Suggested figure tiles:** `[18778,52957]_patch_0_8`, `[19129,51780]_patch_2_8`, `[10382,50252]_patch_0_4`, `[19129,51780]_patch_4_7`.

Generate figures locally after copying cluster outputs:

```bash
# Unzip tiles from cluster, then:
python scripts/build_hemit_paper_figures.py --tiles-dir ~/new_tiles --out figures/hemit
```

Table figures are written even without tiles (`--tables-only`).

---

## Figure list for Jimin

| File | Content |
|------|---------|
| `figures/hemit/fig_table1_benchmark.png` | Main benchmark table |
| `figures/hemit/fig_table2_ablation.png` | Ablation table |
| `figures/hemit/fig_table_downstream.png` | Per-cell / co-expression |
| `figures/hemit/fig_table_per_marker.png` | DAPI / CD3 / panCK Pearson |
| `figures/hemit/fig_qualitative_comparison.png` | 4×8 model comparison grid |
| `figures/hemit/fig_qualitative_detail.png` | Single-tile GT vs Ours channels |

---

## One-paragraph takeaway (for slides/email)

We present a **flow-matching virtual staining model** with **H&E cross-attention** for HEMIT multiplex IF (DAPI, CD3, panCK). Trained in a **unified 80-epoch, 512² framework** against pix2pix, CUT, CycleGAN, ASP, and DiffVS, our model achieves **best average Pearson (0.750)** and **best average SSIM (0.862)** on **945 test tiles**, with **largest gains in panCK correlation** and **lowest co-expression error** in downstream per-cell analysis. Ablations show **cross-attention** as the key architectural addition over vanilla FM; alternative conditioning strategies (CFG, PatchNCE, seg concat) failed on dev screening.
