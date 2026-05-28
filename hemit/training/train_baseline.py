"""
hemit baselines training
"""
import hemit._bootstrap  # noqa: F401

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from datasets import get_dataloader


HEMIT_MARKERS = ["DAPI", "panCK", "CD3"]


def parse_args():
    p = argparse.ArgumentParser(description="Train GAN baselines on HEMIT")
    p.add_argument("--model",      required=True, choices=["pix2pix", "cyclegan", "cut", "asp"])
    p.add_argument("--marker",     required=True, choices=HEMIT_MARKERS)
    p.add_argument("--data-root",  default="data/hemit",
                   help="Path to HEMIT root")
    p.add_argument("--patch-size", type=int,   default=256)
    p.add_argument("--epochs",     type=int,   default=100)
    p.add_argument("--batch-size", type=int,   default=8)
    p.add_argument("--lr",         type=float, default=2e-4)
    p.add_argument("--lambda-l1",  type=float, default=100.0,
                   help="L1 weight for pix2pix generator loss")
    p.add_argument("--lambda-nce", type=float, default=1.0,
                   help="NCE weight for CUT loss")
    p.add_argument("--lambda-asp", type=float, default=1.0,
                   help="ASP weight for adaptive NCE loss (ASP only)")
    p.add_argument("--num-workers",type=int,   default=4)
    p.add_argument("--checkpoint-dir", default="checkpoints/baselines")
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--resume",     default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data helpers — extract single marker from your existing dataloader
# ---------------------------------------------------------------------------

def get_single_marker_batch(batch, marker: str, device: str):
    """
    Your dataloader returns multi-marker batches. This extracts just one marker.
    Returns (input_img, target_img) both [B, 3, H, W] in [-1, 1].
    """
    image = batch["input"].to(device)          # [B, 3, H, W]
    B = image.shape[0]

    target_imgs = []
    valid_idxs  = []

    for j in range(B):
        names = list(batch["target_names"][j])  # e.g. ["Hematoxylin", "DAPI", ...]
        if marker not in names:
            continue
        idx = names.index(marker)

        # targets[j] is [C_out, H, W] where each marker occupies 1 or 3 channels
        gt = batch["targets"][j].to(device)     # [C_out, H, W]
        n_markers = len(names)
        c_per = gt.shape[0] // n_markers        # channels per marker (1 or 3)
        ch_start = idx * c_per

        marker_gt = gt[ch_start : ch_start + c_per]  # [c_per, H, W] already in [-1, 1]

        # Convert to 3-channel RGB if needed (e.g. HEMIT returns 1-channel markers)
        if marker_gt.shape[0] == 1:
            marker_gt = marker_gt.expand(3, -1, -1)

        target_imgs.append(marker_gt)
        valid_idxs.append(j)

    if not valid_idxs:
        return None, None

    image_valid  = image[valid_idxs]
    target_stack = torch.stack(target_imgs)    # [B', 3, H, W]
    return image_valid, target_stack


# ---------------------------------------------------------------------------
# Network building blocks (lightweight, no external deps beyond torch)
# ---------------------------------------------------------------------------

def conv_block(in_ch, out_ch, stride=2, norm=True, activation=True):
    layers = [nn.Conv2d(in_ch, out_ch, 4, stride=stride, padding=1, bias=not norm)]
    if norm:
        layers.append(nn.InstanceNorm2d(out_ch))
    if activation:
        layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(ch, ch, 3),
            nn.InstanceNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(ch, ch, 3),
            nn.InstanceNorm2d(ch),
        )

    def forward(self, x):
        return x + self.block(x)


# ---------------------------------------------------------------------------
# Generator — ResNet-based (standard for pix2pix/CycleGAN/CUT)
# ---------------------------------------------------------------------------

class ResNetGenerator(nn.Module):
    """
    ResNet generator: encode → 9 residual blocks → decode.
    Standard for 256x256 input (pix2pix uses 6 blocks for 128x128).
    """

    def __init__(self, in_ch=3, out_ch=3, ngf=64, n_res=9):
        super().__init__()
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_ch, ngf, 7),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True),
        ]
        # Downsample
        for i in range(2):
            m = 2 ** i
            layers += [
                nn.Conv2d(ngf * m, ngf * m * 2, 4, stride=2, padding=1),
                nn.InstanceNorm2d(ngf * m * 2),
                nn.ReLU(inplace=True),
            ]
        # Residual blocks
        for _ in range(n_res):
            layers.append(ResBlock(ngf * 4))
        # Upsample
        for i in range(2):
            m = 2 ** (1 - i)
            layers += [
                nn.ConvTranspose2d(ngf * m * 2, ngf * m, 4, stride=2, padding=1),
                nn.InstanceNorm2d(ngf * m),
                nn.ReLU(inplace=True),
            ]
        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, out_ch, 7),
            nn.Tanh(),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# ---------------------------------------------------------------------------
# Discriminator — PatchGAN 70×70
# ---------------------------------------------------------------------------

class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_ch=6, ndf=64, n_layers=3):
        super().__init__()
        layers = [nn.Conv2d(in_ch, ndf, 4, stride=2, padding=1),
                  nn.LeakyReLU(0.2, inplace=True)]
        nf = ndf
        for i in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            layers += [nn.Conv2d(nf_prev, nf, 4, stride=2, padding=1, bias=False),
                       nn.InstanceNorm2d(nf),
                       nn.LeakyReLU(0.2, inplace=True)]
        nf_prev = nf
        nf = min(nf * 2, 512)
        layers += [nn.Conv2d(nf_prev, nf, 4, stride=1, padding=1, bias=False),
                   nn.InstanceNorm2d(nf),
                   nn.LeakyReLU(0.2, inplace=True),
                   nn.Conv2d(nf, 1, 4, stride=1, padding=1)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# ---------------------------------------------------------------------------
# CUT — Contrastive Unpaired Translation
# Adds a patch-NCE loss on top of pix2pix GAN. We use a lightweight MLP head
# on intermediate encoder features to compute InfoNCE.
# ---------------------------------------------------------------------------

class PatchNCELoss(nn.Module):
    """
    InfoNCE loss for CUT. Positives = (query feat, same-location fake feat).
    Negatives = all other spatial locations in the batch.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.T   = temperature
        self.ce  = nn.CrossEntropyLoss()

    def forward(self, feat_q: torch.Tensor, feat_k: torch.Tensor) -> torch.Tensor:
        """
        feat_q, feat_k: [B*n_patches, C] L2-normalised feature vectors
        """
        feat_q = F.normalize(feat_q, dim=1)
        feat_k = F.normalize(feat_k, dim=1)
        B      = feat_q.shape[0]

        # Positive: dot product of matched pairs
        pos = (feat_q * feat_k).sum(dim=1, keepdim=True)  # [B, 1]

        # Negatives: all other keys
        neg = feat_q @ feat_k.T                            # [B, B]

        logits = torch.cat([pos, neg], dim=1) / self.T    # [B, B+1]
        labels = torch.zeros(B, dtype=torch.long, device=feat_q.device)
        return self.ce(logits, labels)


class EncoderFeatureExtractor(nn.Module):
    """
    Hooks into the ResNet generator encoder to extract intermediate features
    for the NCE loss (layers 0, 4, 8 of the generator sequential).
    """

    HOOK_LAYERS = [3, 6, 9]   # ReLU after first conv (64ch), after downsample1 (128ch), after downsample2 (256ch)

    def __init__(self, generator: ResNetGenerator, feat_dim: int = 256):
        super().__init__()
        self.generator = generator
        # Channels at each hook layer for ngf=64
        in_dims = [64, 128, 256]
        self.projectors = nn.ModuleList([
            nn.Sequential(nn.Linear(d, feat_dim), nn.ReLU(), nn.Linear(feat_dim, feat_dim))
            for d in in_dims
        ])

    def get_features(self, x: torch.Tensor, n_patches: int = 256):
        """Run encoder, sample random spatial patches, project to feat_dim.
        Returns (feats, patch_indices) so callers can reuse the same locations."""
        feats   = []
        indices = []
        h = x
        for i, layer in enumerate(self.generator.model):
            h = layer(h)
            if i in self.HOOK_LAYERS:
                B, C, H, W = h.shape
                n = min(n_patches, H * W)
                idx = torch.randperm(H * W, device=x.device)[:n]
                patch = h.flatten(2)[:, :, idx]       # [B, C, n]
                patch = patch.permute(0, 2, 1)        # [B, n, C]
                patch = patch.reshape(B * n, C)
                proj_idx = self.HOOK_LAYERS.index(i)
                feats.append(self.projectors[proj_idx](patch))
                indices.append((idx, H, W))
        return feats, indices   # list of [B*n, feat_dim], list of (idx, H, W)


# ---------------------------------------------------------------------------
# GAN Loss (LSGAN)
# ---------------------------------------------------------------------------

class LSGANLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def real_loss(self, pred):
        return self.mse(pred, torch.ones_like(pred))

    def fake_loss(self, pred):
        return self.mse(pred, torch.zeros_like(pred))


# ---------------------------------------------------------------------------
# Trainers
# ---------------------------------------------------------------------------

def train_pix2pix(args, train_loader, val_loader, device, ckpt_dir):
    G = ResNetGenerator(in_ch=3, out_ch=3).to(device)
    D = PatchGANDiscriminator(in_ch=6).to(device)  # input + target concatenated

    opt_G  = Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_D  = Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))
    sched_G = CosineAnnealingLR(opt_G, T_max=args.epochs, eta_min=1e-6)
    sched_D = CosineAnnealingLR(opt_D, T_max=args.epochs, eta_min=1e-6)
    gan_loss = LSGANLoss()
    l1_loss  = nn.L1Loss()

    best_val = float("inf")

    for epoch in range(args.epochs):
        G.train(); D.train()
        t0 = time.time()
        total_g = total_d = n_batches = 0

        for batch in train_loader:
            real_A, real_B = get_single_marker_batch(batch, args.marker, device)
            if real_A is None:
                continue

            fake_B = G(real_A)

            # --- Discriminator ---
            opt_D.zero_grad()
            real_pair = torch.cat([real_A, real_B], dim=1)
            fake_pair = torch.cat([real_A, fake_B.detach()], dim=1)
            d_loss = 0.5 * (gan_loss.real_loss(D(real_pair)) +
                            gan_loss.fake_loss(D(fake_pair)))
            d_loss.backward(); opt_D.step()

            # --- Generator ---
            opt_G.zero_grad()
            fake_pair_g = torch.cat([real_A, fake_B], dim=1)
            g_loss = (gan_loss.real_loss(D(fake_pair_g)) +
                      args.lambda_l1 * l1_loss(fake_B, real_B))
            g_loss.backward(); opt_G.step()

            total_g += g_loss.item(); total_d += d_loss.item(); n_batches += 1

        # Validation (L1 proxy)
        G.eval()
        val_l1 = 0.0; val_n = 0
        with torch.no_grad():
            for batch in val_loader:
                real_A, real_B = get_single_marker_batch(batch, args.marker, device)
                if real_A is None:
                    continue
                fake_B = G(real_A)
                val_l1 += l1_loss(fake_B, real_B).item(); val_n += 1
        val_loss = val_l1 / max(val_n, 1)

        sched_G.step(); sched_D.step()
        print(f"Epoch {epoch:3d} | G: {total_g/max(n_batches,1):.4f} | "
              f"D: {total_d/max(n_batches,1):.4f} | Val L1: {val_loss:.4f} | "
              f"Time: {time.time()-t0:.1f}s")

        state = {"epoch": epoch, "model_state": G.state_dict(), "val_loss": val_loss}
        if val_loss < best_val:
            best_val = val_loss
            torch.save(state, ckpt_dir / "best.pt")
            print(f"  Saved best (val={val_loss:.4f})")


def train_cyclegan(args, train_loader, val_loader, device, ckpt_dir):
    """
    CycleGAN: G_AB (input→stain), G_BA (stain→input), D_A, D_B.
    Cycle consistency loss: A→B→A and B→A→B.
    Since we have paired data we also add L1 (supervised signal).
    """
    G_AB = ResNetGenerator(3, 3).to(device)
    G_BA = ResNetGenerator(3, 3).to(device)
    D_A  = PatchGANDiscriminator(in_ch=3).to(device)   # unpaired discriminators
    D_B  = PatchGANDiscriminator(in_ch=3).to(device)

    opt_G = Adam(list(G_AB.parameters()) + list(G_BA.parameters()),
                 lr=args.lr, betas=(0.5, 0.999))
    opt_D = Adam(list(D_A.parameters()) + list(D_B.parameters()),
                 lr=args.lr, betas=(0.5, 0.999))
    sched_G = CosineAnnealingLR(opt_G, T_max=args.epochs, eta_min=1e-6)
    sched_D = CosineAnnealingLR(opt_D, T_max=args.epochs, eta_min=1e-6)

    gan_loss   = LSGANLoss()
    l1_loss    = nn.L1Loss()
    lambda_cyc = 10.0
    lambda_sup = args.lambda_l1

    best_val = float("inf")

    for epoch in range(args.epochs):
        G_AB.train(); G_BA.train(); D_A.train(); D_B.train()
        t0 = time.time()
        total_g = total_d = n_batches = 0

        for batch in train_loader:
            real_A, real_B = get_single_marker_batch(batch, args.marker, device)
            if real_A is None:
                continue

            fake_B  = G_AB(real_A)
            fake_A  = G_BA(real_B)
            rec_A   = G_BA(fake_B)
            rec_B   = G_AB(fake_A)

            # --- Generators ---
            opt_G.zero_grad()
            g_gan  = (gan_loss.real_loss(D_B(fake_B)) +
                      gan_loss.real_loss(D_A(fake_A)))
            g_cyc  = (l1_loss(rec_A, real_A) + l1_loss(rec_B, real_B)) * lambda_cyc
            g_sup  = l1_loss(fake_B, real_B) * lambda_sup   # supervised bonus
            g_loss = g_gan + g_cyc + g_sup
            g_loss.backward(); opt_G.step()

            # --- Discriminators ---
            opt_D.zero_grad()
            d_loss = 0.5 * (
                gan_loss.real_loss(D_B(real_B)) + gan_loss.fake_loss(D_B(fake_B.detach())) +
                gan_loss.real_loss(D_A(real_A)) + gan_loss.fake_loss(D_A(fake_A.detach()))
            )
            d_loss.backward(); opt_D.step()

            total_g += g_loss.item(); total_d += d_loss.item(); n_batches += 1

        G_AB.eval()
        val_l1 = 0.0; val_n = 0
        with torch.no_grad():
            for batch in val_loader:
                real_A, real_B = get_single_marker_batch(batch, args.marker, device)
                if real_A is None:
                    continue
                fake_B = G_AB(real_A)
                val_l1 += l1_loss(fake_B, real_B).item(); val_n += 1
        val_loss = val_l1 / max(val_n, 1)

        sched_G.step(); sched_D.step()
        print(f"Epoch {epoch:3d} | G: {total_g/max(n_batches,1):.4f} | "
              f"D: {total_d/max(n_batches,1):.4f} | Val L1: {val_loss:.4f} | "
              f"Time: {time.time()-t0:.1f}s")

        state = {"epoch": epoch, "model_state": G_AB.state_dict(), "val_loss": val_loss}
        if val_loss < best_val:
            best_val = val_loss
            torch.save(state, ckpt_dir / "best.pt")
            print(f"  Saved best (val={val_loss:.4f})")


def train_cut(args, train_loader, val_loader, device, ckpt_dir):
    """
    CUT: pix2pix GAN + patch-NCE contrastive loss on encoder features.
    No cycle or G_BA — more efficient than CycleGAN.
    """
    G = ResNetGenerator(3, 3).to(device)
    D = PatchGANDiscriminator(in_ch=3).to(device)   # CUT uses unpaired D
    F = EncoderFeatureExtractor(G, feat_dim=256).to(device)

    opt_G = Adam(list(G.parameters()) + list(F.projectors.parameters()),
                 lr=args.lr, betas=(0.5, 0.999))
    opt_D = Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))
    sched_G = CosineAnnealingLR(opt_G, T_max=args.epochs, eta_min=1e-6)
    sched_D = CosineAnnealingLR(opt_D, T_max=args.epochs, eta_min=1e-6)

    gan_loss = LSGANLoss()
    nce_loss = PatchNCELoss(temperature=0.07)
    l1_loss  = nn.L1Loss()

    best_val = float("inf")

    for epoch in range(args.epochs):
        G.train(); D.train()
        t0 = time.time()
        total_g = total_d = n_batches = 0

        for batch in train_loader:
            real_A, real_B = get_single_marker_batch(batch, args.marker, device)
            if real_A is None:
                continue

            fake_B = G(real_A)

            # --- Discriminator (on fake_B vs real_B, unpaired) ---
            opt_D.zero_grad()
            d_loss = 0.5 * (gan_loss.real_loss(D(real_B)) +
                            gan_loss.fake_loss(D(fake_B.detach())))
            d_loss.backward(); opt_D.step()

            # --- Generator + NCE ---
            opt_G.zero_grad()
            g_gan = gan_loss.real_loss(D(fake_B))

            # NCE: features of real_A vs features of fake_B at same locations
            feats_real, _ = F.get_features(real_A)
            feats_fake, _ = F.get_features(fake_B)
            nce = sum(nce_loss(fq, fk) for fq, fk in zip(feats_fake, feats_real))
            nce = nce / len(feats_real)

            # Optional supervised L1 (CUT authors note this helps in paired setting)
            g_sup  = l1_loss(fake_B, real_B) * args.lambda_l1
            g_loss = g_gan + args.lambda_nce * nce + g_sup
            g_loss.backward(); opt_G.step()

            total_g += g_loss.item(); total_d += d_loss.item(); n_batches += 1

        G.eval()
        val_l1 = 0.0; val_n = 0
        with torch.no_grad():
            for batch in val_loader:
                real_A, real_B = get_single_marker_batch(batch, args.marker, device)
                if real_A is None:
                    continue
                fake_B = G(real_A)
                val_l1 += l1_loss(fake_B, real_B).item(); val_n += 1
        val_loss = val_l1 / max(val_n, 1)

        sched_G.step(); sched_D.step()
        print(f"Epoch {epoch:3d} | G: {total_g/max(n_batches,1):.4f} | "
              f"D: {total_d/max(n_batches,1):.4f} | Val L1: {val_loss:.4f} | "
              f"Time: {time.time()-t0:.1f}s")

        state = {"epoch": epoch, "model_state": G.state_dict(), "val_loss": val_loss}
        if val_loss < best_val:
            best_val = val_loss
            torch.save(state, ckpt_dir / "best.pt")
            print(f"  Saved best (val={val_loss:.4f})")


# ---------------------------------------------------------------------------
# ASP — Adaptive Supervised PatchNCE Loss (Li et al. 2023)
#
# Key difference from CUT:
#   CUT weights all patches equally in the NCE loss.
#   ASP computes a per-patch difficulty weight w_i based on the L1 error
#   between fake and real at that patch location. Hard patches (high L1)
#   get UP-weighted so the model focuses where it struggles most.
#
#   weight_i = softmax(l1_per_patch / temperature)  × N
#
# Everything else (generator, discriminator, GAN loss) is identical to CUT.
# ---------------------------------------------------------------------------

class AdaptivePatchNCELoss(nn.Module):
    """
    ASP: InfoNCE loss with per-patch adaptive weights derived from L1 error.

    feat_q, feat_k : [B*N, C] L2-normalised patch features (fake, real)
    l1_weights     : [B*N]    per-patch L1 errors (higher = harder patch)
    """

    def __init__(self, temperature: float = 0.07, weight_temp: float = 0.1):
        super().__init__()
        self.T        = temperature      # NCE temperature
        self.wT       = weight_temp      # softmax temperature for weights
        self.ce       = nn.CrossEntropyLoss(reduction="none")

    def forward(self, feat_q: torch.Tensor, feat_k: torch.Tensor,
                l1_weights: torch.Tensor) -> torch.Tensor:
        feat_q = F.normalize(feat_q, dim=1)
        feat_k = F.normalize(feat_k, dim=1)
        B      = feat_q.shape[0]

        pos    = (feat_q * feat_k).sum(dim=1, keepdim=True)   # [B, 1]
        neg    = feat_q @ feat_k.T                             # [B, B]
        logits = torch.cat([pos, neg], dim=1) / self.T        # [B, B+1]
        labels = torch.zeros(B, dtype=torch.long, device=feat_q.device)

        # Per-patch CE loss
        loss_per_patch = self.ce(logits, labels)               # [B]

        # Adaptive weights: harder patches get higher weight
        # Normalise so weights sum to B (keeps loss scale stable)
        weights = F.softmax(l1_weights / self.wT, dim=0) * B
        return (loss_per_patch * weights.detach()).mean()


def get_patch_l1_weights(real_A: torch.Tensor, fake_B: torch.Tensor,
                         patch_indices: list) -> list:
    """
    Compute per-patch L1 error at the SAME spatial locations used by
    EncoderFeatureExtractor.get_features, so weights align with features.

    patch_indices: list of (idx, H, W) returned by get_features
    Returns list of [B*n] tensors, one per hook layer.
    """
    weights_per_layer = []
    B = real_A.shape[0]

    for (idx, H, W) in patch_indices:
        n = idx.shape[0]
        # Downsample real_A and fake_B to the feature map's spatial size
        r_down = F.interpolate(real_A, size=(H, W), mode="bilinear", align_corners=False)
        f_down = F.interpolate(fake_B, size=(H, W), mode="bilinear", align_corners=False)

        l1_map  = (r_down - f_down).abs().mean(dim=1)  # [B, H, W]
        l1_flat = l1_map.flatten(1)                    # [B, H*W]

        # Sample the exact same patch locations
        l1_patches = l1_flat[:, idx]                   # [B, n]
        weights_per_layer.append(l1_patches.reshape(B * n))

    return weights_per_layer


def train_asp(args, train_loader, val_loader, device, ckpt_dir):
    """
    ASP: same architecture as CUT, but replaces uniform PatchNCE with
    Adaptive Supervised PatchNCE that up-weights hard patches.
    """
    G = ResNetGenerator(3, 3).to(device)
    D = PatchGANDiscriminator(in_ch=3).to(device)
    F_net = EncoderFeatureExtractor(G, feat_dim=256).to(device)

    opt_G = Adam(list(G.parameters()) + list(F_net.projectors.parameters()),
                 lr=args.lr, betas=(0.5, 0.999))
    opt_D = Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))
    sched_G = CosineAnnealingLR(opt_G, T_max=args.epochs, eta_min=1e-6)
    sched_D = CosineAnnealingLR(opt_D, T_max=args.epochs, eta_min=1e-6)

    gan_loss  = LSGANLoss()
    asp_loss  = AdaptivePatchNCELoss(temperature=0.07, weight_temp=0.1)
    l1_loss   = nn.L1Loss()

    best_val = float("inf")

    for epoch in range(args.epochs):
        G.train(); D.train()
        t0 = time.time()
        total_g = total_d = n_batches = 0

        for batch in train_loader:
            real_A, real_B = get_single_marker_batch(batch, args.marker, device)
            if real_A is None:
                continue

            fake_B = G(real_A)

            # --- Discriminator ---
            opt_D.zero_grad()
            d_loss = 0.5 * (gan_loss.real_loss(D(real_B)) +
                            gan_loss.fake_loss(D(fake_B.detach())))
            d_loss.backward(); opt_D.step()

            # --- Generator + Adaptive NCE ---
            opt_G.zero_grad()
            g_gan = gan_loss.real_loss(D(fake_B))

            # Encoder features + patch indices (shared random locations)
            feats_fake, patch_indices = F_net.get_features(fake_B)
            feats_real, _             = F_net.get_features(real_A)

            # Per-patch L1 weights at the SAME locations as the features
            l1_weights = get_patch_l1_weights(real_A, fake_B, patch_indices)

            # Adaptive NCE loss across all hook layers
            nce = sum(
                asp_loss(fq, fk, w)
                for fq, fk, w in zip(feats_fake, feats_real, l1_weights)
            ) / len(feats_real)

            # Supervised L1 (same as CUT)
            g_sup  = l1_loss(fake_B, real_B) * args.lambda_l1
            g_loss = g_gan + args.lambda_asp * nce + g_sup
            g_loss.backward(); opt_G.step()

            total_g += g_loss.item(); total_d += d_loss.item(); n_batches += 1

        G.eval()
        val_l1 = 0.0; val_n = 0
        with torch.no_grad():
            for batch in val_loader:
                real_A, real_B = get_single_marker_batch(batch, args.marker, device)
                if real_A is None:
                    continue
                fake_B = G(real_A)
                val_l1 += l1_loss(fake_B, real_B).item(); val_n += 1
        val_loss = val_l1 / max(val_n, 1)

        sched_G.step(); sched_D.step()
        print(f"Epoch {epoch:3d} | G: {total_g/n_batches:.4f} | "
              f"D: {total_d/n_batches:.4f} | Val L1: {val_loss:.4f} | "
              f"Time: {time.time()-t0:.1f}s")

        state = {"epoch": epoch, "model_state": G.state_dict(), "val_loss": val_loss}
        if val_loss < best_val:
            best_val = val_loss
            torch.save(state, ckpt_dir / "best.pt")
            print(f"  Saved best (val={val_loss:.4f})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | Model: {args.model} | HEMIT | Marker: {args.marker}")

    run_name = f"{args.model}_hemit_{args.marker}"
    ckpt_dir = Path(args.checkpoint_dir) / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    data_root = str(args.data_root)
    train_loader = get_dataloader(
        "hemit", data_root, split="train",
        patch_size=args.patch_size, batch_size=args.batch_size,
        num_workers=args.num_workers, shuffle=True,
    )
    val_loader = get_dataloader(
        "hemit", data_root, split="val",
        patch_size=args.patch_size, batch_size=args.batch_size,
        num_workers=args.num_workers, shuffle=False,
    )

    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    if args.model == "pix2pix":
        train_pix2pix(args, train_loader, val_loader, device, ckpt_dir)
    elif args.model == "cyclegan":
        train_cyclegan(args, train_loader, val_loader, device, ckpt_dir)
    elif args.model == "cut":
        train_cut(args, train_loader, val_loader, device, ckpt_dir)
    elif args.model == "asp":
        train_asp(args, train_loader, val_loader, device, ckpt_dir)


if __name__ == "__main__":
    main()
