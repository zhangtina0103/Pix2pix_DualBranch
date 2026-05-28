"""Patch NCE losses and encoder hooks for CUT / ASP (joint 3-channel HEMIT)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchNCELoss(nn.Module):
    """InfoNCE: matched patch = positive, other batch locations = negatives."""

    def __init__(self, temperature=0.07):
        super().__init__()
        self.T = temperature
        self.ce = nn.CrossEntropyLoss()

    def forward(self, feat_q, feat_k):
        feat_q = F.normalize(feat_q, dim=1)
        feat_k = F.normalize(feat_k, dim=1)
        b = feat_q.shape[0]
        pos = (feat_q * feat_k).sum(dim=1, keepdim=True)
        neg = feat_q @ feat_k.T
        logits = torch.cat([pos, neg], dim=1) / self.T
        labels = torch.zeros(b, dtype=torch.long, device=feat_q.device)
        return self.ce(logits, labels)


class AdaptivePatchNCELoss(nn.Module):
    """ASP: per-patch NCE weighted by L1 difficulty (fake vs real target)."""

    def __init__(self, temperature=0.07, weight_temp=0.1):
        super().__init__()
        self.T = temperature
        self.wT = weight_temp
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, feat_q, feat_k, l1_weights):
        feat_q = F.normalize(feat_q, dim=1)
        feat_k = F.normalize(feat_k, dim=1)
        b = feat_q.shape[0]
        pos = (feat_q * feat_k).sum(dim=1, keepdim=True)
        neg = feat_q @ feat_k.T
        logits = torch.cat([pos, neg], dim=1) / self.T
        labels = torch.zeros(b, dtype=torch.long, device=feat_q.device)
        loss_per_patch = self.ce(logits, labels)
        weights = F.softmax(l1_weights / self.wT, dim=0) * b
        return (loss_per_patch * weights.detach()).mean()


class EncoderFeatureExtractor(nn.Module):
    """Hooks ResnetGenerator.model at layers 3, 6, 9 (64 / 128 / 256 channels)."""

    HOOK_LAYERS = [3, 6, 9]

    def __init__(self, generator, feat_dim=256):
        super().__init__()
        if not hasattr(generator, 'model'):
            raise ValueError('EncoderFeatureExtractor expects a generator with .model Sequential')
        self.generator = generator
        in_dims = [64, 128, 256]
        self.projectors = nn.ModuleList([
            nn.Sequential(nn.Linear(d, feat_dim), nn.ReLU(), nn.Linear(feat_dim, feat_dim))
            for d in in_dims
        ])

    def get_features(self, x, n_patches=256):
        feats = []
        indices = []
        h = x
        for i, layer in enumerate(self.generator.model):
            h = layer(h)
            if i in self.HOOK_LAYERS:
                b, c, height, width = h.shape
                n = min(n_patches, height * width)
                idx = torch.randperm(height * width, device=x.device)[:n]
                patch = h.flatten(2)[:, :, idx].permute(0, 2, 1).reshape(b * n, c)
                proj_idx = self.HOOK_LAYERS.index(i)
                feats.append(self.projectors[proj_idx](patch))
                indices.append((idx, height, width))
        return feats, indices


def get_patch_l1_weights(real, fake, patch_indices):
    """Per-patch mean L1 between real and fake target at NCE patch locations."""
    weights_per_layer = []
    b = real.shape[0]
    for idx, height, width in patch_indices:
        n = idx.shape[0]
        r_down = F.interpolate(real, size=(height, width), mode='bilinear', align_corners=False)
        f_down = F.interpolate(fake, size=(height, width), mode='bilinear', align_corners=False)
        l1_map = (r_down - f_down).abs().mean(dim=1)
        l1_flat = l1_map.flatten(1)
        l1_patches = l1_flat[:, idx]
        weights_per_layer.append(l1_patches.reshape(b * n))
    return weights_per_layer


def generator_module(net):
    """Unwrap DataParallel for feature hooks."""
    return net.module if isinstance(net, torch.nn.DataParallel) else net
