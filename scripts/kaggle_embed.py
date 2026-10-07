"""Embed the 512x384 image cache with a frozen ImageNet-pretrained ResNet-50 (no labels used).

Run in a Kaggle notebook (GPU on, dataset attached, internet on for the weights download).
Output: /kaggle/working/emb_resnet50.npz
  names : cache_name of every kept image (as in the manifest), sorted
  emb   : float16 array (N, 2048), raw global-average-pooled features (not normalised)
and /kaggle/working/emb_resnet50.meta.json (library versions, weights, counts).
"""
import glob
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image

IN_ROOT = "/kaggle/input"
OUT_NPZ = "/kaggle/working/emb_resnet50.npz"
OUT_META = "/kaggle/working/emb_resnet50.meta.json"
BATCH = 64
EXPECTED_HW = (384, 512)

# --- manifest: kept images only -------------------------------------------------
mf = sorted(glob.glob(f"{IN_ROOT}/**/*.manifest.csv", recursive=True))
assert mf, "no manifest files found under /kaggle/input - is the dataset attached?"
man = pd.concat([pd.read_csv(p) for p in mf], ignore_index=True)
man = man[man["note"].str.startswith("kept")].sort_values("cache_name").reset_index(drop=True)
assert man["cache_name"].is_unique
print("manifest files:", len(mf), "| kept images:", len(man))

# --- locate image files (Kaggle unpacks the tars; layout may differ) -------------
by_base = {}
for dp, _, fns in os.walk(IN_ROOT):
    for fn in fns:
        if fn.lower().endswith(".jpg"):
            assert fn not in by_base, f"duplicate basename {fn}"
            by_base[fn] = os.path.join(dp, fn)
paths = []
missing = []
for cn in man["cache_name"]:
    p = by_base.get(os.path.basename(cn))
    (paths if p else missing).append(p or cn)
assert not missing, f"{len(missing)} images missing, e.g. {missing[:3]}"
print("images found:", len(paths), "| extra files not in manifest:", len(by_base) - len(paths))


class Images(torch.utils.data.Dataset):
    def __len__(self):
        return len(paths)

    def __getitem__(self, i):
        a = np.asarray(Image.open(paths[i]).convert("RGB"), dtype=np.uint8)
        assert a.shape[:2] == EXPECTED_HW, (paths[i], a.shape)
        return torch.from_numpy(a).permute(2, 0, 1)


dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", dev)
weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1
model = torchvision.models.resnet50(weights=weights)
model.fc = torch.nn.Identity()
model = model.eval().to(dev)
mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1) * 255
std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1) * 255

dl = torch.utils.data.DataLoader(Images(), batch_size=BATCH, shuffle=False, num_workers=4)
out = []
t0 = time.time()
with torch.no_grad(), torch.autocast(device_type=dev, dtype=torch.float16, enabled=(dev == "cuda")):
    for k, x in enumerate(dl):
        x = (x.to(dev).float() - mean) / std
        out.append(model(x).float().cpu().numpy().astype(np.float16))
        if k % 50 == 0:
            print(f"{k * BATCH}/{len(paths)}  {time.time() - t0:.0f}s", flush=True)
emb = np.concatenate(out)
assert emb.shape == (len(paths), 2048) and np.isfinite(emb.astype(np.float32)).all()

np.savez(OUT_NPZ, names=man["cache_name"].to_numpy().astype(str), emb=emb)
json.dump(
    {
        "model": "torchvision resnet50",
        "weights": "IMAGENET1K_V1",
        "input": "full 384x512 image, ImageNet mean/std, no crop",
        "feature": "global average pool, 2048-d, float16",
        "n": int(len(emb)),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "device": dev,
    },
    open(OUT_META, "w"),
    indent=2,
)
print("saved", OUT_NPZ, emb.shape, f"{time.time() - t0:.0f}s")
