"""Final test evaluation. Run once per checkpoint, after the configuration is frozen."""
import json
import time
from pathlib import Path

import numpy as np
import torch

from .data import SemDataset
from .models import build
from .splits import get_split, index_images, load_splits, resolve_paths, sha256_file
from .train import make_loader, predict


def evaluate_test(run_dir, splits_path, image_root, device="cuda", workers=4):
    run_dir = Path(run_dir)
    lock = run_dir / "test.lock"
    if lock.exists():
        raise RuntimeError(f"test split already evaluated for {run_dir.name}; refusing to run again")
    if not (run_dir / "done.flag").exists():
        raise RuntimeError(f"{run_dir.name} is not a finished run")
    info = json.loads((run_dir / "config.json").read_text())
    cfg = info["cfg"]
    sp = load_splits(splits_path)
    if sha256_file(splits_path) != info["splits_sha256"]:
        raise RuntimeError("splits.json differs from the one used for training")
    files, y = get_split(sp, info["variant"], ("test",), allow_test=True)["test"]
    paths = resolve_paths(files, index_images(image_root))
    ds = SemDataset(paths, y, train=False, hw=tuple(cfg["hw"]), region=cfg.get("region", "full"))
    micro = cfg["batch"] // cfg["accum"]
    loader = make_loader(ds, micro * 2, False, workers, 0, False, device)
    model, _ = build(info["backbone"], len(info["classes"]), pretrained=False)
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location="cpu"))
    model = model.to(device).to(memory_format=torch.channels_last)
    logits = predict(model, loader, device, str(device).startswith("cuda"))
    np.savez(run_dir / "test_logits.npz", names=np.array(files), y=y, logits=logits)
    lock.write_text(json.dumps(dict(checkpoint_sha256=sha256_file(run_dir / "best.pt"),
                                    time=time.strftime("%Y-%m-%d %H:%M:%S"), n=len(files))))
    return run_dir / "test_logits.npz"
