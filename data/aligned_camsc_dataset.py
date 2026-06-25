"""CaMSC brightfield → fluorescence with synced spatial aug + BF jitter (train only)."""
import random

import torch
import torchvision.transforms as transforms
from PIL import Image, ImageFilter

from data.base_dataset import BaseDataset, get_params, get_transformA, get_transformB


def _center_crop_pair(a_img: Image.Image, b_img: Image.Image, size: int):
    w, h = a_img.size
    if w < size or h < size:
        a_img = a_img.resize((size, size), Image.Resampling.BILINEAR)
        b_img = b_img.resize((size, size), Image.Resampling.BILINEAR)
        return a_img, b_img
    x = max(0, (w - size) // 2)
    y = max(0, (h - size) // 2)
    box = (x, y, x + size, y + size)
    return a_img.crop(box), b_img.crop(box)


def _apply_rotation_pair(a_img: Image.Image, b_img: Image.Image, k: int):
    if k <= 0:
        return a_img, b_img
    return a_img.rotate(90 * k, expand=True), b_img.rotate(90 * k, expand=True)


def _scale_jitter_pair(a_img: Image.Image, b_img: Image.Image, crop_size: int, jitter: float):
    """Random zoom before crop (synced). jitter=0.12 → scale in [0.88, 1.12]."""
    if jitter <= 0:
        return a_img, b_img
    w, h = a_img.size
    if w < crop_size or h < crop_size:
        return a_img, b_img
    scale = random.uniform(1.0 - jitter, 1.0 + jitter)
    new_w = max(crop_size, int(w * scale))
    new_h = max(crop_size, int(h * scale))
    if new_w == w and new_h == h:
        return a_img, b_img
    resample = Image.Resampling.BILINEAR
    return a_img.resize((new_w, new_h), resample), b_img.resize((new_w, new_h), resample)


def _jitter_bf(tensor: torch.Tensor) -> torch.Tensor:
    """Brightness / contrast / gamma on input only. tensor in [-1, 1]."""
    t = (tensor + 1.0) / 2.0
    t = (t + random.uniform(-0.12, 0.12)).clamp(0, 1)
    mean = t.mean(dim=(-2, -1), keepdim=True)
    contrast = random.uniform(0.85, 1.15)
    t = ((t - mean) * contrast + mean).clamp(0, 1)
    gamma = random.uniform(0.85, 1.15)
    t = t.clamp(1e-8, 1).pow(gamma)
    return t * 2.0 - 1.0


def _bf_noise(tensor: torch.Tensor, std: float) -> torch.Tensor:
    if std <= 0:
        return tensor
    return (tensor + torch.randn_like(tensor) * std).clamp(-1.0, 1.0)


def _maybe_blur_bf(a_img: Image.Image, p: float = 0.15) -> Image.Image:
    """Mild defocus on BF only (synced aug would blur labels too)."""
    if random.random() >= p:
        return a_img
    radius = random.uniform(0.3, 0.8)
    return a_img.filter(ImageFilter.GaussianBlur(radius=radius))


class AlignedCamscDataset(BaseDataset):
    """
    Paired trainA/trainB with synced aug for tiny CaMSC sets.

    Train: scale jitter → rot90 → random crop → h-flip → v-flip → BF blur/jitter/noise.
    Val/test: center crop 512.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(
            dataset_mode="aligned_camsc",
            preprocess="crop",
            crop_size=512,
            camsc_repeats=8,
            camsc_scale_jitter=0.12,
            camsc_bf_noise=0.02,
        )
        return parser

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        from data.image_folder import make_dataset

        self.dir_A = f"{opt.dataroot}/{opt.phase}A"
        self.dir_B = f"{opt.dataroot}/{opt.phase}B"
        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))
        assert len(self.A_paths) == len(self.B_paths), "trainA/trainB size mismatch"
        self.is_train = opt.phase == "train"
        self.repeats = max(1, int(getattr(opt, "camsc_repeats", 1) or 1))
        self.scale_jitter = float(getattr(opt, "camsc_scale_jitter", 0.12) or 0.0)
        self.bf_noise = float(getattr(opt, "camsc_bf_noise", 0.02) or 0.0)
        if self.is_train:
            print(
                f"AlignedCamscDataset train: {len(self.A_paths)} fields × "
                f"{self.repeats} repeats = {len(self)} samples/epoch, "
                f"scale_jitter={self.scale_jitter}, bf_noise={self.bf_noise}"
            )

    def __len__(self):
        if self.is_train:
            return len(self.A_paths) * self.repeats
        return len(self.A_paths)

    def __getitem__(self, index):
        image_index = index % len(self.A_paths)
        a_path = self.A_paths[image_index]
        b_path = self.B_paths[image_index]
        a_img = Image.open(a_path).convert("RGB")
        b_img = Image.open(b_path).convert("RGB")

        if self.is_train and not self.opt.no_flip:
            a_img, b_img = _scale_jitter_pair(
                a_img, b_img, self.opt.crop_size, self.scale_jitter,
            )
            if random.random() > 0.5:
                k = random.randint(1, 3)
                a_img, b_img = _apply_rotation_pair(a_img, b_img, k)
            a_img = _maybe_blur_bf(a_img)
            params = get_params(self.opt, a_img.size)
            transform_a = get_transformA(self.opt, params=params, grayscale=0)
            transform_b = get_transformB(self.opt, params=params, grayscale=0)
            a = transform_a(a_img)
            b = transform_b(b_img)
            if random.random() > 0.5:
                a = torch.flip(a, dims=[-2])
                b = torch.flip(b, dims=[-2])
            a = _jitter_bf(a)
            a = _bf_noise(a, self.bf_noise)
        else:
            a_img, b_img = _center_crop_pair(a_img, b_img, self.opt.crop_size)
            to_tensor = transforms.Compose([
                transforms.Resize([self.opt.crop_size, self.opt.crop_size], Image.Resampling.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            a = to_tensor(a_img)
            b = to_tensor(b_img)

        return {"A": a, "B": b, "A_paths": a_path, "B_paths": b_path}
