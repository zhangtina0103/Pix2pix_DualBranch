"""
flow_matching_adapted.py
Kanghyun's flow matching code adapted to work with datasets.py.

Key changes from flow_matching.py:
  - Data pipeline replaced with get_dataloader / get_combined_dataloader
  - Batch keys: batch["input"] / batch["targets"] instead of DeepLIIF-specific keys
  - FlowNet in/out channels are now configurable (handles variable markers per dataset)
  - SaveValImageCallback updated to work without LightningDataModule
  - Metrics computed per-channel then averaged (handles multi-channel targets)
  - LPIPS added alongside PSNR / SSIM / PCC (optional, skipped if unavailable)

Bugs fixed vs first draft:
  - ODE last step (t=1) Heun explosion: switched to Euler at final step
  - ModelCheckpoint given explicit dirpath so --eval_only can find checkpoints
  - ModelCheckpoint save_last=True as fallback when val_ssim not yet logged
  - LPIPS made optional with try/except (requires VGG weights download)
  - collate_fn always returns targets as list — _unpack handles this correctly

New components (full model):
  - FFL (Focal Frequency Loss): penalises high-frequency errors in frequency domain
  - CFG (Classifier-Free Guidance): null-cond dropout at train time, guided sampling
  - FiLM (Feature-wise Linear Modulation): scale/shift conditioning in every decoder block
"""

import hemit._bootstrap  # noqa: F401

from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, Callback
from monai.networks.nets import DiffusionModelUNet
from monai.losses import PerceptualLoss
from monai.inferers import sliding_window_inference
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics import PearsonCorrCoef
import torchvision.utils as vutils

# LPIPS is optional — requires VGG weights download on first use
try:
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("Warning: LPIPS not available. Install torchvision to enable.")

# Your dataset utilities
from datasets import get_dataloader, get_combined_dataloader

ROOT = path_setup.repo_root()


# ---------------------------------------------------------------------------
# FFL — Focal Frequency Loss
# Penalises errors in the frequency domain, weighted toward high frequencies
# which flow matching tends to underfit. Pure PyTorch, no extra deps.
# Reference: Jiang et al. "Focal Frequency Loss for Image Reconstruction
# and Synthesis" (ICCV 2021).
# ---------------------------------------------------------------------------

class FocalFrequencyLoss(nn.Module):
    """
    Computes a spatially-weighted L2 loss in the 2D frequency domain.

    For each (u,v) frequency bin the weight is the squared magnitude of the
    *average* spectrum across the batch, so persistently wrong frequencies
    (typically high-freq) are penalised more.  alpha controls the overall
    strength of the frequency weighting (alpha=1 → standard focal, alpha=0
    → plain spectral L2).
    """
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred / target: [B, C, H, W] in [-1, 1]
        # FFT doesn't support bfloat16 — cast to float32
        pred   = pred.float()
        target = target.float()

        # Compute 2D FFT and shift DC to centre
        pred_f   = torch.fft.fftshift(torch.fft.fft2(pred,   norm="ortho"))
        target_f = torch.fft.fftshift(torch.fft.fft2(target, norm="ortho"))

        # Spectral error: real + imaginary parts separately
        diff_real = pred_f.real - target_f.real
        diff_imag = pred_f.imag - target_f.imag

        # Focal weight: mean magnitude of target spectrum across batch
        # [B, C, H, W] -> [1, C, H, W], then broadcast
        weight = (target_f.real ** 2 + target_f.imag ** 2).mean(dim=0, keepdim=True)
        weight = weight ** self.alpha                           # raise to alpha
        weight = weight / (weight.mean() + 1e-8)               # normalise

        loss = weight * (diff_real ** 2 + diff_imag ** 2)
        return loss.mean()


# ---------------------------------------------------------------------------
# FiLM — Feature-wise Linear Modulation (decoder injection only)
# A small MLP maps a pooled embedding of the conditioning image to per-channel
# scale (gamma) and shift (beta) vectors, applied after each decoder block.
# Decoder-only injection: encoder extracts structure, decoder steers generation.
# Reference: Perez et al. "FiLM: Visual Reasoning with a General Conditioning
# Layer" (AAAI 2018).
# ---------------------------------------------------------------------------

class FiLMGenerator(nn.Module):
    """
    Maps a spatially-pooled conditioning embedding to (gamma, beta) pairs
    for each decoder feature map channel.

    Args:
        in_ch:        channels of the conditioning image (e.g. 3 for RGB)
        decoder_dims: tuple of channel counts at each decoder resolution level
                      (must match the UNet decoder order, coarse -> fine)
        hidden_dim:   width of the shared MLP trunk
    """
    def __init__(self, in_ch: int, decoder_dims: tuple, hidden_dim: int = 128):
        super().__init__()
        # Shared encoder: pool cond image -> embedding
        self.pool = nn.AdaptiveAvgPool2d(1)          # spatial -> scalar per ch
        self.trunk = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_ch, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        # Per-level heads: one (gamma, beta) pair per decoder level
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, 2 * dim) for dim in decoder_dims
        ])
        # Initialise gamma branch to 1 and beta to 0 (identity at start)
        for head in self.heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
            # Set gamma portion to 1
            head.bias.data[:head.out_features // 2] = 1.0

    def forward(self, cond: torch.Tensor):
        """
        Args:
            cond: [B, in_ch, H, W]
        Returns:
            list of (gamma, beta) tuples, one per decoder level
            gamma/beta each [B, dim, 1, 1] ready for broadcasting
        """
        h = self.trunk(self.pool(cond).squeeze(-1).squeeze(-1))
        out = []
        for head in self.heads:
            gb = head(h)                             # [B, 2*dim]
            dim = gb.shape[1] // 2
            gamma = gb[:, :dim].unsqueeze(-1).unsqueeze(-1)
            beta  = gb[:, dim:].unsqueeze(-1).unsqueeze(-1)
            out.append((gamma, beta))
        return out


class FiLMedFlowNet(nn.Module):
    """
    Wraps DiffusionModelUNet with FiLM conditioning injected at every
    decoder block output.  The UNet still receives the concat input
    (x_t || cond); FiLM provides an *additional* affine modulation of
    each decoder feature map using a global embedding of cond.

    This is additive to the existing concat conditioning — FiLM helps
    the decoder adapt generation style/intensity without changing the
    encoder's structural feature extraction.
    """
    def __init__(self, in_ch: int = 3, out_ch: int = 1,
                 channels: tuple = (64, 128, 192),
                 attention_levels: tuple = (False, False, True),
                 num_res_blocks: int = 2,
                 num_head_channels: int = 32,
                 film_hidden_dim: int = 128,
                 use_film: bool = True):
        super().__init__()
        self.out_ch  = out_ch
        self.use_film = use_film

        self.unet = DiffusionModelUNet(
            spatial_dims      = 2,
            in_channels       = in_ch + out_ch,
            out_channels      = out_ch,
            channels          = channels,
            attention_levels  = attention_levels,
            num_res_blocks    = num_res_blocks,
            num_head_channels = num_head_channels,
        )

        if use_film:
            # Decoder levels run fine->coarse in reverse channel order
            # MONAI UNet decoder mirrors encoder: same channel dims reversed
            decoder_dims = tuple(reversed(channels))
            self.film = FiLMGenerator(in_ch, decoder_dims, film_hidden_dim)
        else:
            self.film = None

        # Register decoder block hooks so FiLM can intercept their outputs
        self._film_params  = None   # set per forward call
        self._hook_handles = []

    def _register_decoder_hooks(self):
        """
        Attach forward hooks to MONAI UNet decoder ResBlocks so we can apply
        FiLM modulation after each block.  Hooks are registered once and
        reused; params are updated each forward call via self._film_params.
        """
        if self._hook_handles:
            return   # already registered

        n_film_levels = len(self.film.heads)   # number of FiLM heads available

        # MONAI DiffusionModelUNet stores decoder blocks in self.unet.up_blocks
        # Each up_block is a list of ResBlocks; we hook the *last* ResBlock in
        # each spatial level (the one whose output feeds the skip connection).
        for level_idx, up_block in enumerate(self.unet.up_blocks):
            if level_idx >= n_film_levels:
                break   # no FiLM head for this level — skip

            resblocks = [m for m in up_block.modules()
                         if m.__class__.__name__ == "DiffusionUNetResnetBlock"]
            if not resblocks:
                continue
            last_rb = resblocks[-1]

            def make_hook(lvl):
                def hook(module, input, output):
                    if self._film_params is None:
                        return output
                    gamma, beta = self._film_params[lvl]
                    # output may be a tuple (MONAI sometimes returns (out, emb))
                    if isinstance(output, tuple):
                        return (gamma * output[0] + beta,) + output[1:]
                    return gamma * output + beta
                return hook

            handle = last_rb.register_forward_hook(make_hook(level_idx))
            self._hook_handles.append(handle)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor,
                cond: torch.Tensor) -> torch.Tensor:
        """
        x_t:  [B, out_ch, H, W]
        t:    [B]
        cond: [B, in_ch,  H, W]
        """
        if self.use_film and self.film is not None:
            self._register_decoder_hooks()
            self._film_params = self.film(cond)
        else:
            self._film_params = None

        x   = torch.cat([x_t, cond], dim=1)
        out = self.unet(x, timesteps=t)
        self._film_params = None   # clear after forward
        return torch.tanh(out)


# ---------------------------------------------------------------------------
# Convenience alias — FlowNet now points to FiLMedFlowNet so the rest of
# the file (FlowModule, argparse) can remain unchanged.  Pass use_film=False
# to reproduce the vanilla behaviour exactly.
# ---------------------------------------------------------------------------

class FlowNet(FiLMedFlowNet):
    """Thin alias kept for backward compatibility with existing checkpoints."""
    def __init__(self, in_ch=3, out_ch=1,
                 channels=(64, 128, 192),
                 attention_levels=(False, False, True),
                 num_res_blocks=2,
                 num_head_channels=32,
                 film_hidden_dim=128,
                 use_film=True,
                 **kwargs):
        super().__init__(
            in_ch=in_ch, out_ch=out_ch,
            channels=channels,
            attention_levels=attention_levels,
            num_res_blocks=num_res_blocks,
            num_head_channels=num_head_channels,
            film_hidden_dim=film_hidden_dim,
            use_film=use_film,
        )


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

class SaveValImageCallback(Callback):
    """
    Saves a grid of [Input | Prediction | GT] every val epoch.
    First 3 channels are shown for multi-channel tensors.
    """
    def __init__(self, val_dataset, num_samples=4, output_dir="val_images"):
        self.val_dataset = val_dataset
        self.num_samples = num_samples
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def on_validation_epoch_end(self, trainer, pl_module):
        n = min(self.num_samples, len(self.val_dataset))
        indices = torch.randperm(len(self.val_dataset))[:n]

        def to3(t):
            """Normalise to [0,1] and ensure 3 channels for saving."""
            if t.shape[1] == 1:
                t = t.repeat(1, 3, 1, 1)
            elif t.shape[1] > 3:
                t = t[:, :3]
            return t.add(1).div(2).clamp(0, 1)

        rows = []
        for idx in indices:
            sample = self.val_dataset[idx.item()]
            inp = sample["input"].unsqueeze(0).to(pl_module.device)      # [1, C_in, H, W]
            # targets may be list (from collate_fn) or tensor
            tgt = sample["targets"]
            if isinstance(tgt, list):
                tgt = tgt[0]
            gt = tgt.unsqueeze(0).to(pl_module.device)                   # [1, C_out, H, W]

            with torch.no_grad():
                pred = pl_module.sample(inp)                             # [1, C_out, H, W]

            # 3 columns: Input | Pred | GT
            rows.append(torch.cat([to3(inp), to3(pred), to3(gt)], dim=0))

        grid = vutils.make_grid(torch.cat(rows), nrow=3, padding=2)
        vutils.save_image(grid, self.output_dir / f"epoch_{trainer.current_epoch:03d}.png")


# ---------------------------------------------------------------------------
# Lightning Module
# ---------------------------------------------------------------------------

class FlowModule(L.LightningModule):
    def __init__(self, model: FlowNet, lr=3e-4, lambda_perc=0.1, P_mean=-0.8, P_std=0.8,
                 save_images=False, image_dir="inference_images",
                 # FFL
                 use_ffl=True, lambda_ffl=0.1, ffl_alpha=1.0,
                 # CFG
                 use_cfg=True, cfg_dropout=0.1, cfg_scale=3.0):
        super().__init__()
        self.model = model
        self.lr = lr
        self.lambda_perc = lambda_perc
        self.P_mean = P_mean
        self.P_std = P_std
        self.save_images = save_images
        self.image_dir = Path(image_dir)
        if save_images:
            self.image_dir.mkdir(parents=True, exist_ok=True)

        # FFL
        self.use_ffl    = use_ffl
        self.lambda_ffl = lambda_ffl
        self.ffl_fn     = FocalFrequencyLoss(alpha=ffl_alpha) if use_ffl else None

        # CFG
        self.use_cfg     = use_cfg
        self.cfg_dropout = cfg_dropout   # prob of zeroing cond during training
        self.cfg_scale   = cfg_scale     # guidance weight w at inference

        self.perceptual_loss_fn = PerceptualLoss(
            spatial_dims=2, network_type="vgg", is_fake_3d=False,
        )
        self.psnr  = PeakSignalNoiseRatio(data_range=2.0)
        self.ssim  = StructuralSimilarityIndexMeasure(data_range=2.0)
        self.pcc   = PearsonCorrCoef()
        if LPIPS_AVAILABLE:
            # Keep LPIPS on CPU — don't register as submodule so Lightning
            # won't move it to GPU. We'll move it temporarily during eval.
            lpips_metric = LearnedPerceptualImagePatchSimilarity(
                net_type="vgg", normalize=False
            )
            # Store as plain attribute (not nn.Module) to avoid GPU placement
            object.__setattr__(self, "_lpips_cpu", lpips_metric)
        else:
            object.__setattr__(self, "_lpips_cpu", None)

    # ------------------------------------------------------------------
    # Flow matching forward pass
    # Logit-normal time sampling (Karras et al.), x1 prediction formulation
    # ------------------------------------------------------------------

    def forward_cfm(self, x1, cond):
        B = x1.shape[0]
        x0 = torch.randn_like(x1)
        # Logit-normal time: concentrates samples away from t=0 and t=1
        t    = torch.sigmoid(self.P_mean + self.P_std * torch.randn(B, device=x1.device))
        t_bc = t.view(B, 1, 1, 1)

        # Linear interpolation: x_t = (1-t)*x0 + t*x1
        x_t = (1 - t_bc) * x0 + t_bc * x1

        # CFG: randomly zero out conditioning so model learns unconditional path
        if self.use_cfg and self.training:
            mask = (torch.rand(B, device=x1.device) > self.cfg_dropout)
            # mask: [B] bool — False means drop this sample's conditioning
            cond_in = cond * mask.view(B, 1, 1, 1).float()
        else:
            cond_in = cond

        x1_pred = self.model(x_t, t, cond_in)

        loss_l1   = F.l1_loss(x1_pred, x1)
        loss_perc = self.perceptual_loss_fn(self._to3ch(x1_pred), self._to3ch(x1))

        loss = loss_l1 + self.lambda_perc * loss_perc

        # FFL: focal frequency loss on top of spatial losses
        if self.use_ffl and self.ffl_fn is not None:
            loss_ffl = self.ffl_fn(x1_pred, x1)
            loss = loss + self.lambda_ffl * loss_ffl
        else:
            loss_ffl = torch.tensor(0.0, device=x1.device)

        return loss, loss_l1, loss_perc, loss_ffl

    # ------------------------------------------------------------------
    # ODE sampler (Heun solver, x1-prediction parameterisation)
    # FIX: use Euler at the final step (t->1) to avoid v = (x1-x_t)/(1-t)
    #      blowing up when 1-t -> 0.
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(self, cond, steps=25, method="heun", cfg_scale=None):
        """
        ODE sampler with optional CFG guidance.

        cfg_scale: override self.cfg_scale at call time (useful for sweeps).
                   Set to 1.0 or use_cfg=False to disable guidance.
        """
        B      = cond.shape[0]
        out_ch = self.model.out_ch
        x_t    = torch.randn(B, out_ch, *cond.shape[2:], device=cond.device)
        ts     = torch.linspace(0, 1, steps + 1, device=cond.device)

        w = cfg_scale if cfg_scale is not None else self.cfg_scale
        do_cfg = self.use_cfg and (w != 1.0)

        # Null conditioning for CFG (zeros, same shape as cond)
        null_cond = torch.zeros_like(cond) if do_cfg else None

        for i in range(steps):
            t0, t1 = ts[i], ts[i + 1]
            dt     = t1 - t0
            is_last = (i == steps - 1)

            def pred_v(xt, t_scalar):
                x1_cond = self.model(xt, t_scalar.expand(B), cond)
                if do_cfg:
                    x1_uncond = self.model(xt, t_scalar.expand(B), null_cond)
                    # Guided prediction: interpolate between uncond and cond
                    x1_guided = x1_uncond + w * (x1_cond - x1_uncond)
                else:
                    x1_guided = x1_cond
                # Convert x1 prediction to velocity
                return (x1_guided - xt) / (1 - t_scalar).clamp(min=1e-5)

            v0 = pred_v(x_t, t0)

            if method == "euler" or is_last:
                x_t = x_t + dt * v0
            elif method == "heun":
                x_t_euler = x_t + dt * v0
                v1 = pred_v(x_t_euler, t1)
                x_t = x_t + dt * (v0 + v1) / 2

        return x_t

    # ------------------------------------------------------------------
    # Train / val / test steps
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        inp, x1 = self._unpack(batch)
        loss, loss_l1, loss_perc, loss_ffl = self.forward_cfm(x1, inp)
        self.log_dict({
            "train_loss": loss, "train_l1": loss_l1,
            "train_perceptual": loss_perc, "train_ffl": loss_ffl,
        }, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def _eval_step(self, batch, stage):
        inp, x1 = self._unpack(batch)

        # During validation use fewer steps and no CFG to save memory.
        # CFG is only applied at test time (final eval).
        is_test = (stage == "test")
        eval_steps  = getattr(self, "_ode_steps", 25) if is_test else 10
        eval_method = getattr(self, "_ode_method", "heun") if is_test else "euler"
        eval_cfg    = self.cfg_scale if is_test else 1.0   # 1.0 = no guidance

        # Slide over full-resolution input; predictor receives inp patches
        ps = getattr(self, "_patch_size", 128)
        x1_pred = sliding_window_inference(
            inp,
            roi_size=(ps, ps),
            sw_batch_size=1,
            predictor=lambda patch: self.sample(
                patch,
                steps=eval_steps,
                method=eval_method,
                cfg_scale=eval_cfg,
            ),
        ).clamp(-1, 1)

        # Compute metrics per output channel, then average
        psnr_vals, ssim_vals, pcc_vals, lpips_vals = [], [], [], []
        C = x1.shape[1]

        # Move LPIPS to GPU temporarily for eval, then back to CPU
        lpips_metric = self._lpips_cpu
        if lpips_metric is not None:
            lpips_metric = lpips_metric.to(x1.device)

        for c in range(C):
            pred_c = x1_pred[:, c:c+1]   # [B, 1, H, W]
            gt_c   = x1[:, c:c+1]

            psnr_vals.append(self.psnr(pred_c, gt_c))
            ssim_vals.append(self.ssim(pred_c, gt_c))
            pcc_vals.append(self.pcc(pred_c.reshape(-1), gt_c.reshape(-1)))

            if lpips_metric is not None:
                # LPIPS needs 3ch input in [-1, 1]
                lpips_vals.append(
                    lpips_metric(pred_c.repeat(1, 3, 1, 1), gt_c.repeat(1, 3, 1, 1))
                )

        # Move LPIPS back to CPU to free GPU memory
        if lpips_metric is not None:
            self._lpips_cpu = lpips_metric.cpu()

        metrics = {
            f"{stage}_psnr": torch.stack(psnr_vals).mean(),
            f"{stage}_ssim": torch.stack(ssim_vals).mean(),
            f"{stage}_pcc":  torch.stack(pcc_vals).mean(),
        }
        if lpips_vals:
            metrics[f"{stage}_lpips"] = torch.stack(lpips_vals).mean()

        if self.save_images:
            # Save 3-panel grid: Input | Pred | GT
            def to_rgb(t):
                """[1,C,H,W] -> [H,W,3] uint8, first 3ch, [-1,1] -> [0,255]"""
                t = t[0].detach().cpu().float().clamp(-1, 1)
                if t.shape[0] == 1:
                    t = t.repeat(3, 1, 1)
                elif t.shape[0] > 3:
                    t = t[:3]
                arr = ((t + 1) / 2 * 255).numpy().astype(np.uint8)
                return arr.transpose(1, 2, 0)  # [H,W,3]

            inp_np  = to_rgb(inp)
            pred_np = to_rgb(x1_pred)
            gt_np   = to_rgb(x1)

            H = inp_np.shape[0]
            border = np.ones((H, 4, 3), dtype=np.uint8) * 128
            grid = np.concatenate([inp_np, border, pred_np, border, gt_np], axis=1)
            patch_id = getattr(self, "_test_batch_idx", 0)
            Image.fromarray(grid).save(self.image_dir / f"{patch_id:04d}.png")
            self._test_batch_idx = patch_id + 1

        self.log_dict(metrics, prog_bar=True)

    def validation_step(self, batch, batch_idx): self._eval_step(batch, "val")
    def test_step(self, batch, batch_idx):       self._eval_step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.lr)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _unpack(self, batch):
        """
        Unpack batch produced by datasets.py collate_fn.
        collate_fn always returns targets as a list of [C_out, H, W] tensors
        (variable channel counts across datasets). We stack them, padding or
        truncating to model.out_ch so the batch is a uniform tensor.
        """
        inp     = batch["input"]       # [B, C_in, H, W]
        targets = batch["targets"]     # list of [C_out, H, W]  (always a list from collate_fn)

        out_ch = self.model.out_ch
        stacked = []
        for t in targets:
            if t.shape[0] == out_ch:
                stacked.append(t)                           # already right shape
            elif out_ch == 1 and t.shape[0] == 3:
                # Convert RGB to grayscale — avoids picking wrong color channel
                # (e.g. Lap2 is green, Ki67 is brown — mean captures all signal)
                gray = t.mean(dim=0, keepdim=True)          # [1, H, W]
                stacked.append(gray)
            elif t.shape[0] > out_ch:
                stacked.append(t[:out_ch])
            else:
                stacked.append(F.pad(t, (0, 0, 0, 0, 0, out_ch - t.shape[0])))  # zero-pad
        x1 = torch.stack(stacked).to(inp.device)  # [B, out_ch, H, W]

        return inp, x1

    @staticmethod
    def _to3ch(t):
        """Ensure tensor has 3 channels for perceptual loss."""
        if t.shape[1] == 3:
            return t
        if t.shape[1] == 1:
            return t.repeat(1, 3, 1, 1)
        return t[:, :3]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",     type=str, default="deepliif",
                        help="deepliif | bci | hnscc | hemit | combined")
    parser.add_argument("--data_root",   type=str, required=True,
                        help="Path to dataset root. For combined: comma-sep list of 4 roots "
                             "in order deepliif,bci,hnscc,hemit")
    parser.add_argument("--in_ch",       type=int, default=3,
                        help="Input channels (3 for any RGB input modality)")
    parser.add_argument("--out_ch",      type=int, default=1,
                        help="Output channels per marker (1 for grayscale, 3 for RGB)")
    parser.add_argument("--marker",      type=str, default=None,
                        help="Single marker to predict e.g. DAPI, Hematoxylin, HER2_IHC. "
                             "If None, predicts all markers.")
    parser.add_argument("--batch_size",  type=int, default=8)
    parser.add_argument("--max_epochs",  type=int, default=300)
    parser.add_argument("--lr",          type=float, default=3e-4)
    parser.add_argument("--lambda_perc", type=float, default=0.1)
    parser.add_argument("--patch_size",  type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--devices",     type=str, default="0",
                        help="GPU id(s), comma-sep e.g. '0' or '0,1'")
    parser.add_argument("--ckpt",        type=str, default=None,
                        help="Path to checkpoint to resume training or run eval from")
    parser.add_argument("--test_only",   action="store_true",
                        help="Skip training entirely, just run test on --ckpt")
    parser.add_argument("--save_images", action="store_true",
                        help="Save 4-panel grid images during test (for compute_metrics.py)")
    parser.add_argument("--image_dir",   type=str, default="inference_images",
                        help="Directory to save inference grid images")

    # --- FFL ---
    parser.add_argument("--use_ffl",      action="store_true", default=True,
                        help="Enable Focal Frequency Loss (default: on)")
    parser.add_argument("--no_ffl",       dest="use_ffl", action="store_false",
                        help="Disable Focal Frequency Loss (ablation)")
    parser.add_argument("--lambda_ffl",   type=float, default=0.1,
                        help="Weight of FFL term in total loss")
    parser.add_argument("--ffl_alpha",    type=float, default=1.0,
                        help="Focal exponent: 0=flat spectral L2, 1=focal (default)")

    # --- CFG ---
    parser.add_argument("--use_cfg",      action="store_true", default=True,
                        help="Enable Classifier-Free Guidance (default: on)")
    parser.add_argument("--no_cfg",       dest="use_cfg", action="store_false",
                        help="Disable CFG (ablation)")
    parser.add_argument("--cfg_dropout",  type=float, default=0.1,
                        help="Prob of dropping conditioning during training")
    parser.add_argument("--cfg_scale",    type=float, default=3.0,
                        help="Guidance scale w at inference (1.0 = no guidance)")

    # --- FiLM ---
    parser.add_argument("--use_film",     action="store_true", default=True,
                        help="Enable FiLM decoder conditioning (default: on)")
    parser.add_argument("--no_film",      dest="use_film", action="store_false",
                        help="Disable FiLM (ablation)")
    parser.add_argument("--film_hidden",  type=int, default=128,
                        help="Hidden dim of FiLM MLP generator")

    # --- Backbone ---
    parser.add_argument("--channels",    type=str, default="64,128,192",
                        help="UNet channel sizes, comma-sep e.g. '64,128,192'")
    parser.add_argument("--attn_levels", type=str, default="0,0,1",
                        help="Attention levels per UNet stage, comma-sep 0/1 e.g. '0,0,1'")
    parser.add_argument("--num_res_blocks", type=int, default=2,
                        help="ResBlocks per UNet level")
    parser.add_argument("--ode_steps",   type=int, default=25,
                        help="Number of ODE integration steps at inference")
    parser.add_argument("--ode_method",  type=str, default="heun",
                        choices=["heun", "euler"],
                        help="ODE solver: heun (2nd order) or euler (1st order)")
    parser.add_argument("--variant",     type=str, default=None,
                        help="Variant name for organizing checkpoints e.g. vanilla, ffl, film, full")
    args = parser.parse_args()

    if args.test_only:
        assert args.ckpt is not None, "--test_only requires --ckpt"

    devices = [int(d) for d in args.devices.split(",")]

    # ------------------------------------------------------------------
    # Data loaders
    # ------------------------------------------------------------------
    if args.dataset == "combined":
        roots = args.data_root.split(",")
        names = ["deepliif", "bci", "hnscc", "hemit"]
        assert len(roots) == 4, "For combined, provide exactly 4 comma-separated roots"
        dataset_configs = [{"name": n, "root": r.strip()} for n, r in zip(names, roots)]
        train_loader = get_combined_dataloader(dataset_configs, split="train",
                                               batch_size=args.batch_size,
                                               num_workers=args.num_workers)
        val_loader   = get_combined_dataloader(dataset_configs, split="val",
                                               batch_size=1, num_workers=args.num_workers)
        test_loader  = get_combined_dataloader(dataset_configs, split="test",
                                               batch_size=1, num_workers=args.num_workers)
    else:
        marker_list = [args.marker] if args.marker else None
        # BCI only has one marker so target_markers not supported
        tm_kwargs = {} if args.dataset == "bci" else {"target_markers": marker_list}
        train_loader = get_dataloader(args.dataset, args.data_root, split="train",
                                      patch_size=args.patch_size,
                                      batch_size=args.batch_size,
                                      num_workers=args.num_workers,
                                      **tm_kwargs)
        val_loader   = get_dataloader(args.dataset, args.data_root, split="val",
                                      patch_size=args.patch_size,
                                      batch_size=1, num_workers=args.num_workers,
                                      **tm_kwargs)
        test_loader  = get_dataloader(args.dataset, args.data_root, split="test",
                                      patch_size=args.patch_size,
                                      batch_size=1, num_workers=args.num_workers,
                                      **tm_kwargs)

    val_dataset = val_loader.dataset

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    channels     = tuple(int(c) for c in args.channels.split(","))
    attn_levels  = tuple(bool(int(a)) for a in args.attn_levels.split(","))

    model     = FlowNet(
        in_ch             = args.in_ch,
        out_ch            = args.out_ch,
        channels          = channels,
        attention_levels  = attn_levels,
        num_res_blocks    = args.num_res_blocks,
        film_hidden_dim   = args.film_hidden,
        use_film          = args.use_film,
    )
    lit_model = FlowModule(
        model        = model,
        lr           = args.lr,
        lambda_perc  = args.lambda_perc,
        save_images  = args.save_images,
        image_dir    = args.image_dir,
        use_ffl      = args.use_ffl,
        lambda_ffl   = args.lambda_ffl,
        ffl_alpha    = args.ffl_alpha,
        use_cfg      = args.use_cfg,
        cfg_dropout  = args.cfg_dropout,
        cfg_scale    = args.cfg_scale,
    )

    lit_model._ode_steps   = args.ode_steps
    lit_model._ode_method  = args.ode_method
    lit_model._patch_size  = args.patch_size

    marker_tag = f"_{args.marker}" if args.marker else ""
    variant_tag = f"{args.variant}/" if args.variant else ""
    run_name   = f"flow_{args.dataset}_in{args.in_ch}_out{args.out_ch}{marker_tag}"
    ckpt_dir   = Path("checkpoints") / f"{variant_tag}{run_name}"

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices=devices,
        precision="bf16-mixed",
        logger=CSVLogger("logs", name=f"{variant_tag}{run_name}"),
        accumulate_grad_batches=4,
        callbacks=[
            ModelCheckpoint(
                dirpath=ckpt_dir,
                monitor="val_ssim",
                mode="max",
                save_top_k=1,
                save_last=True,
                filename="best",
            ),
            SaveValImageCallback(
                val_dataset=val_dataset,
                num_samples=4,
                output_dir=f"val_images/{variant_tag}{run_name}",
            ),
        ],
        check_val_every_n_epoch=10,
        limit_val_batches=10,
        log_every_n_steps=10,
    )

    if args.test_only:
        # Skip training, run inference only
        # strict=False: tolerates LPIPS key mismatch between old/new checkpoints
        lit_model = FlowModule.load_from_checkpoint(
            args.ckpt, model=model, lr=args.lr, lambda_perc=args.lambda_perc,
            save_images=args.save_images, image_dir=args.image_dir,
            use_ffl=args.use_ffl, lambda_ffl=args.lambda_ffl, ffl_alpha=args.ffl_alpha,
            use_cfg=args.use_cfg, cfg_dropout=args.cfg_dropout, cfg_scale=args.cfg_scale,
            strict=False,
        )
        lit_model._ode_steps  = args.ode_steps
        lit_model._ode_method = args.ode_method
        trainer.test(lit_model, test_loader, verbose=False)
    else:
        trainer.fit(lit_model, train_loader, val_loader, ckpt_path=args.ckpt)
        print(f"\nTraining complete. Best checkpoint saved to {ckpt_dir}/best.ckpt")
