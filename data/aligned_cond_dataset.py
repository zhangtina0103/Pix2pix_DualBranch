"""Aligned H&E + multiplex + pseudo seg (trainSeg/ from generate_hemit_seg_masks.py)."""
import os
from data.aligned_dataset import AlignedDataset
from data.base_dataset import get_transformA
from PIL import Image
import torchvision.transforms as transforms


class AlignedCondDataset(AlignedDataset):
    """AlignedDataset + 1ch seg mask."""

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(dataset_mode="aligned_cond")
        return parser

    def __init__(self, opt):
        AlignedDataset.__init__(self, opt)
        self.dir_Seg = os.path.join(opt.dataroot, opt.phase + "Seg")
        self.transform_Seg = transforms.Compose([
            transforms.Resize([opt.load_size, opt.load_size], interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])

    def __getitem__(self, index):
        item = AlignedDataset.__getitem__(self, index)
        seg_path = os.path.join(self.dir_Seg, os.path.basename(self.A_paths[index]))
        if os.path.isfile(seg_path):
            seg = Image.open(seg_path).convert("L")
            item["seg"] = self.transform_Seg(seg) * 2.0 - 1.0
        else:
            item["seg"] = item["A"].new_zeros(1, item["A"].shape[1], item["A"].shape[2])
        return item
