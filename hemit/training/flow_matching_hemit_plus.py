#!/usr/bin/env python3
"""
flow_matching_hemit_plus.py

HEMIT-focused Flow Matching training entrypoint that keeps the vanilla FM code intact.

Key differences vs scripts/training/flow_matching_adapted.py (vanilla HEMIT train parity:
 512², batch 2, 64/128/192, attn 0,0,1, lr/perc/no_ffl — see train_hemit_vanilla_fm.sbatch):
  - Joint out_ch=3 (one model for DAPI+panCK+CD3) + FiLM + extra H&E aug on input.
  - Gradient checkpointing ON by default (offsets joint/FiLM VRAM, not a quality downgrade).
  - Val sliding-window ROI may be < train patch (monitoring only; paper eval uses infer scripts).

Run (from repo root):
  python scripts/training/flow_matching_hemit_plus.py --data_root data/hemit
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import hemit._bootstrap  # noqa: F401

from datasets import get_dataloader  # noqa: E402


class HemitAugWrapper(Dataset):
    """
    Wrap an existing dataset returning:
      {"input": [3,H,W] in [-1,1], "targets": [3,H,W] in [-1,1], ...}
    and apply additional stochastic augmentations to input only.
    """

    def __init__(
        self,
        base: Dataset,
        p: float = 0.8,
        rgb_scale: float = 0.15,
        rgb_bias: float = 0.05,
        noise_std: float = 0.02,
        blur_p: float = 0.15,
    ):
        self.base = base
        self.p = float(p)
        self.rgb_scale = float(rgb_scale)
        self.rgb_bias = float(rgb_bias)
        self.noise_std = float(noise_std)
        self.blur_p = float(blur_p)

    def __len__(self):
        return len(self.base)

    @staticmethod
    def _to01(x: torch.Tensor) -> torch.Tensor:
        return (x + 1) / 2

    @staticmethod
    def _to11(x: torch.Tensor) -> torch.Tensor:
        return x * 2 - 1

    def _rgb_affine(self, x01: torch.Tensor) -> torch.Tensor:
        # Per-channel affine jitter (simulates stain/intensity variation)
        # x01: [3,H,W] in [0,1]
        scale = torch.empty(3, 1, 1, device=x01.device).uniform_(
            1.0 - self.rgb_scale, 1.0 + self.rgb_scale
        )
        bias = torch.empty(3, 1, 1, device=x01.device).uniform_(
            -self.rgb_bias, self.rgb_bias
        )
        return (x01 * scale + bias).clamp(0, 1)

    def _gaussian_blur(self, x01: torch.Tensor) -> torch.Tensor:
        # Lightweight separable blur via conv (no torchvision dependency)
        # Kernel size 5, sigma ~1.0
        k = torch.tensor([1, 4, 6, 4, 1], device=x01.device, dtype=x01.dtype)
        k = (k / k.sum()).view(1, 1, -1)  # [1,1,5]
        x = x01.unsqueeze(0)  # [1,3,H,W]
        # horizontal
        x = torch.nn.functional.pad(x, (2, 2, 0, 0), mode="reflect")
        x = torch.nn.functional.conv2d(x, k.expand(3, 1, 5), groups=3)
        # vertical
        x = torch.nn.functional.pad(x, (0, 0, 2, 2), mode="reflect")
        x = torch.nn.functional.conv2d(x, k.transpose(1, 2).expand(3, 1, 5), groups=3)
        return x.squeeze(0).clamp(0, 1)

    def __getitem__(self, idx):
        sample = self.base[idx]
        inp = sample["input"]
        if (not torch.is_tensor(inp)) or inp.ndim != 3 or inp.shape[0] != 3:
            return sample

        if random.random() < self.p:
            x01 = self._to01(inp)
            x01 = self._rgb_affine(x01)

            # Additive noise (sensor / scanner noise)
            if self.noise_std > 0:
                x01 = (x01 + torch.randn_like(x01) * self.noise_std).clamp(0, 1)

            # Occasional slight blur (defocus / scanning softness)
            if random.random() < self.blur_p:
                x01 = self._gaussian_blur(x01)

            sample = dict(sample)
            sample["input"] = self._to11(x01)
        return sample


class HemitPlusFlowModule:
    """
    Mixin on FlowModule.

    - forward_cfm: same as vanilla unless aux_loss_max_side > 0 and patch is larger
      (optional escape hatch for OOM; default 0 = full-res perceptual like vanilla).
    - validation_step: smaller sliding-window ROI if val_patch_size < train patch
      (Lightning monitoring only — not paper eval).
    """

    aux_loss_max_side: int = 0

    def _downsample_aux(self, t: torch.Tensor) -> torch.Tensor:
        m = getattr(self, "aux_loss_max_side", 0)
        h, w = t.shape[-2:]
        if m <= 0 or max(h, w) <= m:
            return t
        scale = m / max(h, w)
        return F.interpolate(
            t, size=(max(1, int(h * scale)), max(1, int(w * scale))), mode="area",
        )

    def forward_cfm(self, x1, cond):
        m = getattr(self, "aux_loss_max_side", 0)
        if m <= 0 or max(x1.shape[-2:]) <= m:
            return super().forward_cfm(x1, cond)

        B = x1.shape[0]
        x0 = torch.randn_like(x1)
        t = torch.sigmoid(self.P_mean + self.P_std * torch.randn(B, device=x1.device))
        t_bc = t.view(B, 1, 1, 1)
        x_t = (1 - t_bc) * x0 + t_bc * x1

        if self.use_cfg and self.training:
            mask = torch.rand(B, device=x1.device) > self.cfg_dropout
            cond_in = cond * mask.view(B, 1, 1, 1).float()
        else:
            cond_in = cond

        x1_pred = self.model(x_t, t, cond_in)
        loss_l1 = F.l1_loss(x1_pred, x1)

        xp = self._downsample_aux(x1_pred)
        xg = self._downsample_aux(x1)
        loss_perc = self.perceptual_loss_fn(self._to3ch(xp), self._to3ch(xg))
        loss = loss_l1 + self.lambda_perc * loss_perc

        if self.use_ffl and self.ffl_fn is not None:
            loss_ffl = self.ffl_fn(xp, xg)
            loss = loss + self.lambda_ffl * loss_ffl
        else:
            loss_ffl = torch.tensor(0.0, device=x1.device)

        return loss, loss_l1, loss_perc, loss_ffl

    def validation_step(self, batch, batch_idx):
        train_ps = getattr(self, "_patch_size", 128)
        val_ps = getattr(self, "_val_patch_size", min(256, train_ps))
        self._patch_size = val_ps
        try:
            super().validation_step(batch, batch_idx)
        finally:
            self._patch_size = train_ps

    def test_step(self, batch, batch_idx):
        train_ps = getattr(self, "_patch_size", 128)
        val_ps = getattr(self, "_val_patch_size", min(256, train_ps))
        self._patch_size = val_ps
        try:
            super().test_step(batch, batch_idx)
        finally:
            self._patch_size = train_ps


def parse_args():
    p = argparse.ArgumentParser(description="HEMIT+ Flow Matching (multi-marker) training")
    p.add_argument("--data_root", default="data/hemit", help="Path to HEMIT root (contains train/val/test)")
    p.add_argument("--batch_size", type=int, default=2,
                   help="Per-GPU batch (vanilla HEMIT FM uses 2)")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument(
        "--patch_size",
        type=int,
        default=512,
        help="Train patch size (vanilla HEMIT FM default: 512)",
    )
    p.add_argument(
        "--val_patch_size",
        type=int,
        default=256,
        help="Val ODE sliding-window ROI; train patch unchanged (monitoring only)",
    )
    p.add_argument(
        "--aux_loss_max_side",
        type=int,
        default=0,
        help="If >0, downsample perceptual/FFL to this side (0 = full-res, vanilla parity)",
    )
    p.add_argument("--max_epochs", type=int, default=300)

    # Model
    p.add_argument("--channels", type=str, default="64,128,192")
    p.add_argument("--attn_levels", type=str, default="0,0,1")
    p.add_argument("--num_res_blocks", type=int, default=2)

    # FM
    p.add_argument("--ode_steps", type=int, default=25)
    p.add_argument("--ode_method", type=str, default="heun", choices=["heun", "euler"])

    # Loss / tricks (defaults lean toward quality)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lambda_perc", type=float, default=0.1)
    p.add_argument("--use_ffl", action="store_true", default=False)
    p.add_argument("--no_ffl", dest="use_ffl", action="store_false")
    p.add_argument("--lambda_ffl", type=float, default=0.5)
    p.add_argument("--ffl_alpha", type=float, default=1.0)
    p.add_argument("--use_cfg", action="store_true", default=False)
    p.add_argument("--cfg_dropout", type=float, default=0.1)
    p.add_argument("--cfg_scale", type=float, default=1.0)
    p.add_argument("--use_film", action="store_true", default=True)
    p.add_argument("--no_film", dest="use_film", action="store_false")
    p.add_argument("--film_hidden", type=int, default=128)

    # Aug wrapper
    p.add_argument("--aug_p", type=float, default=0.8)
    p.add_argument("--aug_rgb_scale", type=float, default=0.15)
    p.add_argument("--aug_rgb_bias", type=float, default=0.05)
    p.add_argument("--aug_noise_std", type=float, default=0.02)
    p.add_argument("--aug_blur_p", type=float, default=0.15)

    # Logging / ckpt
    p.add_argument("--variant", type=str, default="hemit_plus", help="Subfolder under checkpoints/")
    p.add_argument("--devices", type=str, default="0",
                   help="GPU id(s), comma-separated (e.g. 0,1 for 2-GPU DDP)")
    p.add_argument("--accumulate_grad_batches", type=int, default=4,
                   help="Gradient accumulation steps (effective batch = batch_size * this * num_gpus)")
    p.add_argument("--gradient_checkpointing", action="store_true", default=True,
                   help="Checkpoint UNet activations (default on for joint+FiLM vs vanilla)")
    p.add_argument("--no_gradient_checkpointing", dest="gradient_checkpointing",
                   action="store_false")
    p.add_argument("--no_val_images", action="store_true", default=True,
                   help="Skip val image grids (default on; saves VRAM during val ODE)")
    p.add_argument("--save_val_images", dest="no_val_images", action="store_false")
    return p.parse_args()


def main():
    args = parse_args()

    # Import training components lazily so `--help` works without Lightning installed.
    from scripts.training.flow_matching_adapted import FlowNet, FlowModule  # noqa: E402

    class CheckpointedFlowNet(FlowNet):
        """Activation checkpointing on UNet (lower VRAM, slower backward)."""

        def __init__(self, *a, gradient_checkpointing: bool = False, **kw):
            super().__init__(*a, **kw)
            self.gradient_checkpointing = gradient_checkpointing

        def forward(self, x_t, t, cond):
            if self.use_film and self.film is not None:
                self._register_decoder_hooks()
                self._film_params = self.film(cond)
            else:
                self._film_params = None
            x = torch.cat([x_t, cond], dim=1)
            if self.gradient_checkpointing and self.training:
                from torch.utils.checkpoint import checkpoint

                def _unet(inp, timesteps):
                    return self.unet(inp, timesteps=timesteps)

                out = checkpoint(_unet, x, t, use_reentrant=False)
            else:
                out = self.unet(x, timesteps=t)
            self._film_params = None
            return torch.tanh(out)

    class HemitPlusLit(HemitPlusFlowModule, FlowModule):
        pass

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("medium")

    channels = tuple(int(c) for c in args.channels.split(","))
    attn_levels = tuple(bool(int(a)) for a in args.attn_levels.split(","))
    devices = [int(d) for d in args.devices.split(",")]

    # HEMIT joint markers (multi-output)
    train_loader = get_dataloader(
        "hemit",
        str(args.data_root),
        split="train",
        patch_size=args.patch_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        target_markers=["DAPI", "panCK", "CD3"],
    )
    val_loader = get_dataloader(
        "hemit",
        str(args.data_root),
        split="val",
        patch_size=args.patch_size,
        batch_size=1,
        num_workers=0,
        shuffle=False,
        target_markers=["DAPI", "panCK", "CD3"],
    )

    # Wrap training dataset only (extra H&E aug)
    train_loader = DataLoader(
        HemitAugWrapper(
            train_loader.dataset,
            p=args.aug_p,
            rgb_scale=args.aug_rgb_scale,
            rgb_bias=args.aug_rgb_bias,
            noise_std=args.aug_noise_std,
            blur_p=args.aug_blur_p,
        ),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True,
    )

    model = CheckpointedFlowNet(
        in_ch=3,
        out_ch=3,
        channels=channels,
        attention_levels=attn_levels,
        num_res_blocks=args.num_res_blocks,
        film_hidden_dim=args.film_hidden,
        use_film=args.use_film,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    val_ps = min(args.val_patch_size, args.patch_size)
    print(
        f"Train patch={args.patch_size}  val_roi={val_ps}  "
        f"channels={channels}  attn={attn_levels}  "
        f"aux_loss_max_side={args.aux_loss_max_side}  ffl={args.use_ffl}  "
        f"grad_ckpt={args.gradient_checkpointing}  gpus={len(devices)}",
        flush=True,
    )

    lit = HemitPlusLit(
        model=model,
        lr=args.lr,
        lambda_perc=args.lambda_perc,
        save_images=not args.no_val_images,
        image_dir="val_images",
        use_ffl=args.use_ffl,
        lambda_ffl=args.lambda_ffl,
        ffl_alpha=args.ffl_alpha,
        use_cfg=args.use_cfg,
        cfg_dropout=args.cfg_dropout,
        cfg_scale=args.cfg_scale,
    )
    lit._ode_steps = args.ode_steps
    lit._ode_method = args.ode_method
    lit._patch_size = args.patch_size
    lit._val_patch_size = val_ps
    lit.aux_loss_max_side = args.aux_loss_max_side

    # Defer Trainer construction to the vanilla script's defaults by importing lightning here.
    import lightning as L
    from lightning.pytorch.callbacks import ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger

    run_name = f"flow_hemit_in3_out3"
    ckpt_dir = Path("checkpoints") / args.variant / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    n_gpu = len(devices)
    strategy = "ddp" if n_gpu > 1 else "auto"
    eff_batch = args.batch_size * args.accumulate_grad_batches * n_gpu
    print(
        f"Trainer : strategy={strategy}  accumulate={args.accumulate_grad_batches}  "
        f"effective_batch≈{eff_batch} (batch/GPU={args.batch_size})",
        flush=True,
    )

    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices=devices,
        strategy=strategy,
        precision="bf16-mixed",
        logger=CSVLogger("logs", name=f"{args.variant}/{run_name}"),
        accumulate_grad_batches=args.accumulate_grad_batches,
        num_sanity_val_steps=1,
        callbacks=[
            ModelCheckpoint(
                dirpath=ckpt_dir,
                monitor="val_ssim",
                mode="max",
                save_top_k=1,
                save_last=True,
                filename="best",
            ),
        ],
        check_val_every_n_epoch=10,
        log_every_n_steps=20,
    )

    trainer.fit(lit, train_loader, val_loader)
    print(f"Done. Best ckpt in: {ckpt_dir}")


if __name__ == "__main__":
    main()

