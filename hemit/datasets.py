"""
datasets.py — Unified data loader for variable-input virtual staining.

Supports:
  - DeepLIIF   : IHC → Hematoxylin, DAPI, Lap2, Ki67, Segmentation
  - BCI         : H&E → HER2 IHC
  - HNSCC       : mIHC (Hematoxylin+AEC) → CD3, CD8, FoxP3, PanCK mIF
  - HEMIT       : H&E → DAPI, panCK, CD3 mIHC

Each dataset returns a dict:
  {
    "input":       FloatTensor [C_in, H, W],   # normalized to [-1, 1]
    "targets":     FloatTensor [C_out, H, W],  # normalized to [-1, 1]
    "input_type":  str,                        # e.g. "IHC", "HE"
    "target_names":list[str],                  # e.g. ["DAPI", "Ki67"]
    "dataset":     str,                        # e.g. "deepliif"
    "patch_id":    str,                        # for traceability
  }

Usage:
  from datasets import get_dataloader
  loader = get_dataloader(
      dataset_name="deepliif",
      root="/data/deepliif",
      split="train",
      patch_size=512,
      batch_size=8,
  )
"""

import os
import re
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import transforms


# ---------------------------------------------------------------------------
# Normalization / augmentation helpers
# ---------------------------------------------------------------------------

def _to_tensor_norm(img: Image.Image) -> torch.Tensor:
    """Convert PIL image to float tensor in [-1, 1]."""
    arr = np.array(img).astype(np.float32) / 127.5 - 1.0
    if arr.ndim == 2:
        arr = arr[:, :, None]
    return torch.from_numpy(arr.transpose(2, 0, 1))  # [C, H, W]


def _resize(img: Image.Image, size: int) -> Image.Image:
    return img.resize((size, size), Image.Resampling.BILINEAR)  # fixed deprecation


def _augment(inp: torch.Tensor, target: torch.Tensor):
    """
    Random flips + 90-degree rotations applied identically to input and target.
    Intensity jitter applied to input only (brightness/contrast/gamma).
    Safe for histology — tissue has no canonical orientation.
    Only used during training (datasets pass augment=True for train split).
    """
    # Spatial augmentations — applied to both input and target
    if random.random() > 0.5:
        inp    = torch.flip(inp,    dims=[-1])
        target = torch.flip(target, dims=[-1])
    if random.random() > 0.5:
        inp    = torch.flip(inp,    dims=[-2])
        target = torch.flip(target, dims=[-2])
    k = random.randint(0, 3)
    if k > 0:
        inp    = torch.rot90(inp,    k, dims=[-2, -1])
        target = torch.rot90(target, k, dims=[-2, -1])

    # Intensity jitter — input only (HE stain variation across scanners/labs)
    # inp is in [-1, 1]; convert to [0,1], jitter, convert back
    if random.random() > 0.5:
        inp01 = (inp + 1) / 2  # [-1,1] -> [0,1]

        # Brightness: uniform shift in [-0.1, 0.1]
        brightness = random.uniform(-0.1, 0.1)
        inp01 = (inp01 + brightness).clamp(0, 1)

        # Contrast: scale around mean in [0.9, 1.1]
        contrast = random.uniform(0.9, 1.1)
        mean = inp01.mean(dim=(-2, -1), keepdim=True)
        inp01 = ((inp01 - mean) * contrast + mean).clamp(0, 1)

        # Gamma: random gamma in [0.9, 1.1]
        gamma = random.uniform(0.9, 1.1)
        inp01 = inp01.clamp(1e-8, 1).pow(gamma)

        inp = inp01 * 2 - 1  # [0,1] -> [-1,1]

    return inp, target


# ---------------------------------------------------------------------------
# DeepLIIF
# ---------------------------------------------------------------------------

# Panel order in the side-by-side image
DEEPLIIF_PANELS = ["IHC", "Hematoxylin", "DAPI", "Lap2", "Ki67", "Segmentation"]
DEEPLIIF_INPUT_IDX = 0
DEEPLIIF_OUTPUT_IDXS = [1, 2, 3, 4, 5]


class DeepLIIFDataset(Dataset):
    """
    DeepLIIF: each PNG is a 3072×512 strip of 6 side-by-side 512×512 panels.
    Input:  panel 0 (IHC)
    Outputs: panels 1-5 (Hematoxylin, DAPI, Lap2, Ki67, Segmentation)

    Augmented files (_neg, _empty, _X.XX_X.XX suffixes) are filtered out
    so only base images are used unless include_augmented=True.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        patch_size: int = 512,
        target_markers: Optional[list] = None,
        include_augmented: bool = False,
    ):
        split_name = {"train": "training", "val": "validation", "test": "testing"}[split]
        self.root = Path(root) / split_name
        self.patch_size = patch_size
        self.target_markers = target_markers or DEEPLIIF_PANELS[1:]
        self.augment = (split == "train")

        # Validate target markers
        for m in self.target_markers:
            assert m in DEEPLIIF_PANELS[1:], f"Unknown DeepLIIF marker: {m}"

        # Filter to base images only (no augmentation suffixes)
        aug_pattern = re.compile(r"(_\d+\.\d+_\d+\.\d+|_neg|_empty)\.png$")
        self.files = sorted([
            f for f in self.root.glob("*.png")
            if include_augmented or not aug_pattern.search(f.name)
        ])
        assert len(self.files) > 0, f"No images found in {self.root}"

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        img = Image.open(path).convert("RGB")
        w, h = img.size
        panel_w = w // 6

        # Slice panels
        panels = [
            img.crop((i * panel_w, 0, (i + 1) * panel_w, h))
            for i in range(6)
        ]

        # Resize if needed
        if self.patch_size != panel_w:
            panels = [_resize(p, self.patch_size) for p in panels]

        inp = _to_tensor_norm(panels[DEEPLIIF_INPUT_IDX])  # [3, H, W]

        # Stack requested output markers
        target_tensors = []
        target_names = []
        for marker in self.target_markers:
            panel_idx = DEEPLIIF_PANELS.index(marker)
            target_tensors.append(_to_tensor_norm(panels[panel_idx]))
            target_names.append(marker)

        targets = torch.cat(target_tensors, dim=0)  # [C_out, H, W]

        if self.augment:
            inp, targets = _augment(inp, targets)

        return {
            "input": inp,
            "targets": targets,
            "input_type": "IHC",
            "target_names": target_names,
            "dataset": "deepliif",
            "patch_id": path.stem,
        }


# ---------------------------------------------------------------------------
# BCI Challenge 2022
# ---------------------------------------------------------------------------

class BCIDataset(Dataset):
    """
    BCI: H&E → HER2 IHC.
    Structure:
      root/
        HE/train/*.png   (input)  e.g. 00003_train_1+.png
        IHC/train/*.png  (output) e.g. 00003_train_2+.png

    HE and IHC filenames differ in the HER2 score suffix (1+/2+/3+),
    so we match by the numeric prefix (first 5 chars, e.g. "00003").
    """

    def __init__(self, root: str, split: str = "train", patch_size: int = 512):
        # BCI only has train/test, no val — use test as val
        actual_split = "test" if split == "val" else split
        self.he_dir = Path(root) / "HE" / actual_split
        self.ihc_dir = Path(root) / "IHC" / actual_split
        self.patch_size = patch_size
        self.augment = (split == "train")

        # Match HE to IHC by exact filename
        self.pairs = []
        for he_file in sorted(self.he_dir.glob("*.png")):
            ihc_file = self.ihc_dir / he_file.name
            if ihc_file.exists():
                self.pairs.append((he_file, ihc_file))

        assert len(self.pairs) > 0, f"No matched HE/IHC pairs found in {root}"
        print(f"  [BCI] Matched {len(self.pairs)} pairs for split={split}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        he_path, ihc_path = self.pairs[idx]

        he  = Image.open(he_path).convert("RGB")
        ihc = Image.open(ihc_path).convert("RGB")

        if self.patch_size != he.size[0]:
            he  = _resize(he,  self.patch_size)
            ihc = _resize(ihc, self.patch_size)

        inp     = _to_tensor_norm(he)   # [3, H, W]
        targets = _to_tensor_norm(ihc)  # [3, H, W] — RGB IHC image as single "marker"

        if self.augment:
            inp, targets = _augment(inp, targets)

        return {
            "input": inp,
            "targets": targets,
            "input_type": "HE",
            "target_names": ["HER2_IHC"],
            "dataset": "bci",
            "patch_id": he_path.stem,
        }


# ---------------------------------------------------------------------------
# HNSCC-mIF-mIHC
# ---------------------------------------------------------------------------

HNSCC_INPUT_MARKER = "hematoxylin"  # lowercase in actual files
HNSCC_OUTPUT_MARKERS = ["CD3", "CD8", "FoxP3", "PanCK"]


class HNSCCDataset(Dataset):
    """
    HNSCC: mIHC (Hematoxylin+AEC) → CD3, CD8, FoxP3, PanCK mIF.
    Structure:
      root/
        Case1/
          mIHC_Data_T1_0_0_Hematoxylin.png  (input)
          mIHC_Data_T1_0_0_CD3.png           (output)
          mIHC_Data_T1_0_0_CD8.png
          mIHC_Data_T1_0_0_FoxP3.png
          mIHC_Data_T1_0_0_PanCK.png
          Segmentation-T1_0_0_CD3.png        (skip)
          ...
        Case2/ ...

    Pairs are grouped by (case, location, patch_index).
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        patch_size: int = 512,
        target_markers: Optional[list] = None,
        train_cases: Optional[list] = None,
        val_cases: Optional[list] = None,
        test_cases: Optional[list] = None,
    ):
        self.root = Path(root)
        self.patch_size = patch_size
        self.target_markers = target_markers or HNSCC_OUTPUT_MARKERS
        self.augment = (split == "train")

        # Default case splits (8 patients total)
        # Adjust these based on your desired split
        all_cases = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        n = len(all_cases)
        default_train = all_cases[:int(n * 0.7)]
        default_val = all_cases[int(n * 0.7):int(n * 0.85)]
        default_test = all_cases[int(n * 0.85):]

        split_map = {
            "train": train_cases or default_train,
            "val": val_cases or default_val,
            "test": test_cases or default_test,
        }
        self.cases = split_map[split]

        # Build list of valid patch groups
        self.samples = []
        for case in self.cases:
            case_dir = self.root / case
            if not case_dir.exists():
                continue

            # Three valid prefixes exist with slightly different naming conventions:
            # Segmentation- : CD3, CD8, FoxP3, PanCK, hematoxylin (lowercase)
            # mIF_Data-     : CD3, CD8, FoxP3, PanCK, hematoxylin (lowercase)
            # mIHC_Data-    : CD3, CD8, FoxP3, PCK,   Hematoxylin (capitalized)
            PREFIX_CONFIG = [
                ("Segmentation-", "hematoxylin", {"PanCK": "PanCK", "CD3": "CD3", "CD8": "CD8", "FoxP3": "FoxP3"}),
                ("mIF_Data-",     "hematoxylin", {"PanCK": "PanCK", "CD3": "CD3", "CD8": "CD8", "FoxP3": "FoxP3"}),
                ("mIHC_Data-",    "Hematoxylin", {"PanCK": "PCK",   "CD3": "CD3", "CD8": "CD8", "FoxP3": "FoxP3"}),
            ]

            for prefix, he_suffix, marker_map in PREFIX_CONFIG:
                he_files = list(case_dir.glob(f"{prefix}*_{he_suffix}.png"))
                for he_file in he_files:
                    stem = he_file.stem
                    patch_key = stem.replace(prefix, "").replace(f"_{he_suffix}", "")

                    marker_paths = {}
                    valid = True
                    for marker in self.target_markers:
                        file_marker = marker_map.get(marker, marker)
                        marker_file = case_dir / f"{prefix}{patch_key}_{file_marker}.png"
                        if marker_file.exists():
                            marker_paths[marker] = marker_file
                        else:
                            valid = False
                            break

                    if valid:
                        self.samples.append({
                            "input_path": he_file,
                            "marker_paths": marker_paths,
                            "patch_id": f"{case}_{prefix}{patch_key}",
                        })

        assert len(self.samples) > 0, f"No valid HNSCC samples found for split={split}"

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        inp_img = Image.open(sample["input_path"]).convert("RGB")
        if self.patch_size != inp_img.size[0]:
            inp_img = _resize(inp_img, self.patch_size)

        target_tensors = []
        target_names = []
        for marker in self.target_markers:
            m_img = Image.open(sample["marker_paths"][marker]).convert("RGB")
            if self.patch_size != m_img.size[0]:
                m_img = _resize(m_img, self.patch_size)
            target_tensors.append(_to_tensor_norm(m_img))
            target_names.append(marker)

        inp     = _to_tensor_norm(inp_img)
        targets = torch.cat(target_tensors, dim=0)

        if self.augment:
            inp, targets = _augment(inp, targets)

        return {
            "input": inp,
            "targets": targets,
            "input_type": "mIHC_Hematoxylin",
            "target_names": target_names,
            "dataset": "hnscc",
            "patch_id": sample["patch_id"],
        }


# ---------------------------------------------------------------------------
# HEMIT
# ---------------------------------------------------------------------------

HEMIT_OUTPUT_MARKERS = ["DAPI", "panCK", "CD3"]  # channel order in label tif


class HEMITDataset(Dataset):
    """
    HEMIT: H&E → DAPI, panCK, CD3 mIHC.
    Structure:
      root/
        train/
          input/  *.tif  (H&E, RGB)
          label/  *.tif  (mIHC, RGB — 3 channels = DAPI, panCK, CD3)
        val/ ...
        test/ ...

    Label TIF stores mIHC as a 3-channel image (not separate files per marker).
    Channel order matches official HEMIT post_process.py:
      ch0 = DAPI, ch1 = CD3, ch2 = panCK
    Paper composites: DAPI=blue, CD3=green, panCK=red.
    """

    CHANNEL_MAP = {"DAPI": 0, "CD3": 1, "panCK": 2}  # HEMIT post_process.py

    def __init__(
        self,
        root: str,
        split: str = "train",
        patch_size: int = 512,
        target_markers: Optional[list] = None,
    ):
        self.root = Path(root) / split
        self.patch_size = patch_size
        self.target_markers = target_markers or HEMIT_OUTPUT_MARKERS
        self.augment = (split == "train")

        self.input_dir = self.root / "input"
        self.label_dir = self.root / "label"

        self.files = sorted(self.input_dir.glob("*.tif"))
        if not self.files:
            self.files = sorted(self.input_dir.glob("*.tiff"))
        assert len(self.files) > 0, f"No input TIFs found in {self.input_dir}"

        missing = [f for f in self.files if not (self.label_dir / f.name).exists()]
        if missing:
            print(f"  [HEMIT] Warning: {len(missing)} inputs have no label")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        inp_path = self.files[idx]
        label_path = self.label_dir / inp_path.name

        inp_img = Image.open(inp_path).convert("RGB")
        label_img = Image.open(label_path).convert("RGB")

        if self.patch_size != inp_img.size[0]:
            inp_img = _resize(inp_img, self.patch_size)
            label_img = _resize(label_img, self.patch_size)

        inp_tensor = _to_tensor_norm(inp_img)  # [3, H, W]

        # Extract requested marker channels from label
        label_arr = np.array(label_img).astype(np.float32) / 127.5 - 1.0
        target_tensors = []
        target_names = []
        for marker in self.target_markers:
            ch = self.CHANNEL_MAP[marker]
            channel = torch.from_numpy(label_arr[:, :, ch]).unsqueeze(0)  # [1, H, W]
            target_tensors.append(channel)
            target_names.append(marker)

        targets = torch.cat(target_tensors, dim=0)  # [C_out, H, W]

        if self.augment:
            inp_tensor, targets = _augment(inp_tensor, targets)

        return {
            "input": inp_tensor,
            "targets": targets,
            "input_type": "HE",
            "target_names": target_names,
            "dataset": "hemit",
            "patch_id": inp_path.stem,
        }


# ---------------------------------------------------------------------------
# Combined loader
# ---------------------------------------------------------------------------

DATASET_REGISTRY = {
    "deepliif": DeepLIIFDataset,
    "bci": BCIDataset,
    "hnscc": HNSCCDataset,
    "hemit": HEMITDataset,
}


def get_dataset(dataset_name: str, root: str, split: str = "train", **kwargs) -> Dataset:
    """Get a single dataset by name."""
    assert dataset_name in DATASET_REGISTRY, \
        f"Unknown dataset: {dataset_name}. Choose from {list(DATASET_REGISTRY)}"
    return DATASET_REGISTRY[dataset_name](root=root, split=split, **kwargs)


def get_combined_dataset(dataset_configs: list, split: str = "train") -> ConcatDataset:
    """
    Combine multiple datasets into one.

    Args:
        dataset_configs: list of dicts, each with keys:
            - name: dataset name
            - root: path to dataset root
            - kwargs: (optional) extra args for the dataset class
        split: "train", "val", or "test"

    Example:
        configs = [
            {"name": "deepliif", "root": "/data/deepliif"},
            {"name": "bci",      "root": "/data/bci"},
            {"name": "hemit",    "root": "/data/hemit"},
            {"name": "hnscc",    "root": "/data/hnscc"},
        ]
        ds = get_combined_dataset(configs, split="train")
    """
    datasets = []
    for cfg in dataset_configs:
        name = cfg["name"]
        root = cfg["root"]
        kwargs = cfg.get("kwargs", {})
        ds = get_dataset(name, root, split=split, **kwargs)
        print(f"  [{name}] {split}: {len(ds)} samples")
        datasets.append(ds)
    return ConcatDataset(datasets)


def collate_fn(batch):
    """
    Custom collate that handles variable target channels across datasets.
    Targets are returned as a list since different datasets have different
    numbers of output markers.
    """
    return {
        "input": torch.stack([b["input"] for b in batch]),
        "targets": [b["targets"] for b in batch],           # list of [C_out, H, W]
        "input_type": [b["input_type"] for b in batch],
        "target_names": [b["target_names"] for b in batch], # list of lists
        "dataset": [b["dataset"] for b in batch],
        "patch_id": [b["patch_id"] for b in batch],
    }


def get_dataloader(
    dataset_name: str,
    root: str,
    split: str = "train",
    patch_size: int = 512,
    batch_size: int = 8,
    num_workers: int = 4,
    shuffle: Optional[bool] = None,
    **dataset_kwargs,
) -> DataLoader:
    """
    Convenience function: get a DataLoader for a single dataset.

    Example:
        loader = get_dataloader("deepliif", "/data/deepliif", split="train", batch_size=8)
        for batch in loader:
            x = batch["input"]       # [B, 3, 512, 512]
            y = batch["targets"]     # [B, C_out, 512, 512]
            print(batch["target_names"])
    """
    ds = get_dataset(dataset_name, root, split=split, patch_size=patch_size, **dataset_kwargs)
    if shuffle is None:
        shuffle = (split == "train")
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )


def get_combined_dataloader(
    dataset_configs: list,
    split: str = "train",
    batch_size: int = 8,
    num_workers: int = 4,
    shuffle: Optional[bool] = None,
) -> DataLoader:
    """Get a DataLoader combining multiple datasets."""
    ds = get_combined_dataset(dataset_configs, split=split)
    if shuffle is None:
        shuffle = (split == "train")
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    DATA_ROOTS = {
        "deepliif": "/home/zhangtin/data/deepliif",
        "bci":      "/home/zhangtin/data/bci",
        "hnscc":    "/home/zhangtin/data/hnscc",
        "hemit":    "/home/zhangtin/data/hemit",
    }

    print("=" * 60)
    print("Dataset sanity check")
    print("=" * 60)

    for name, root in DATA_ROOTS.items():
        if not Path(root).exists():
            print(f"\n[{name}] SKIP — root not found: {root}")
            continue
        try:
            loader = get_dataloader(name, root, split="train", batch_size=2, num_workers=0)
            batch = next(iter(loader))
            print(f"\n[{name}]")
            print(f"  input shape:    {batch['input'].shape}")
            print(f"  targets shape:  {batch['targets'].shape}")
            print(f"  input_type:     {batch['input_type']}")
            print(f"  target_names:   {batch['target_names']}")
            print(f"  input range:    [{batch['input'].min():.2f}, {batch['input'].max():.2f}]")
            print(f"  targets range:  [{batch['targets'].min():.2f}, {batch['targets'].max():.2f}]")
        except Exception as e:
            print(f"\n[{name}] ERROR: {e}")

    print("\n" + "=" * 60)
    print("Done.")
