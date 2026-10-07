"""Image dataset for the 512x384 cache (full image with the bar by default; E3 uses `region`)."""
import random

import numpy as np
import torch
from PIL import Image

from .bar import region_rows

W, H = 512, 384


class SemDataset(torch.utils.data.Dataset):
    """Returns (uint8 tensor 3xHxW, label). Augmentation (train only): horizontal flip and a light
    random crop resized back to the (cropped) size. No vertical flip: it would move the info bar.
    `region` (E3, see semcls.bar): the rows kept after resizing to hw and before augmentation."""

    def __init__(self, paths, labels, train=False, crop_scale=(0.85, 1.0), hw=(H, W), region="full"):
        assert len(paths) == len(labels)
        self.paths, self.labels, self.train = list(paths), np.asarray(labels), train
        self.crop_scale, self.hw, self.region = crop_scale, hw, region
        region_rows(region, hw[0])  # validates the name

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        im = Image.open(self.paths[i]).convert("RGB")
        h, w = self.hw
        if im.size != (w, h):
            im = im.resize((w, h), Image.BILINEAR)
        if self.region != "full":
            y0, y1 = region_rows(self.region, h)
            im = im.crop((0, y0, w, y1))
            h = y1 - y0
        if self.train:
            if random.random() < 0.5:
                im = im.transpose(Image.FLIP_LEFT_RIGHT)
            s = random.uniform(*self.crop_scale)
            if s < 0.999:
                cw, ch = int(round(w * s ** 0.5)), int(round(h * s ** 0.5))
                x0, y0 = random.randint(0, w - cw), random.randint(0, h - ch)
                im = im.crop((x0, y0, x0 + cw, y0 + ch)).resize((w, h), Image.BILINEAR)
        a = np.array(im, dtype=np.uint8)
        return torch.from_numpy(a).permute(2, 0, 1), int(self.labels[i])
