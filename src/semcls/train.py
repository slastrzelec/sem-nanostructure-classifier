"""Training of one run: (split variant, backbone, seed). Sees only the train and val parts."""
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import metrics as M
from .data import SemDataset
from .models import build
from .splits import canon_variant, class_names, get_split, index_images, load_splits, resolve_paths, sha256_file

DEFAULTS = dict(
    epochs=12, batch=32, accum=1, lr_backbone=1e-4, lr_head=1e-3, wd=0.01, warmup_epochs=1,
    label_smoothing=0.0, crop_scale=(0.85, 1.0), hw=(384, 512), workers=4, patience=5,
    clip=1.0, pretrained=True, limit=None, region="full",
)
PER_BACKBONE = {"resnet50": {}, "convnext_tiny": {"batch": 16, "accum": 2}}
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def code_sha():
    h = hashlib.sha256()
    for p in sorted(Path(__file__).parent.glob("*.py")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def run_name(variant, backbone, seed, region="full"):
    tag = "" if region == "full" else f"__{region}"
    return f"{variant.replace(':', '')}__{backbone}{tag}__s{seed}"


def prep(x, device):
    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1) * 255
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1) * 255
    x = (x.to(device, non_blocking=True).float() - mean) / std
    return x.contiguous(memory_format=torch.channels_last)


def make_loader(ds, batch, shuffle, workers, seed, drop_last, device):
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.utils.data.DataLoader(
        ds, batch_size=batch, shuffle=shuffle, num_workers=workers, drop_last=drop_last,
        generator=g, pin_memory=str(device).startswith("cuda"), persistent_workers=workers > 0)


@torch.no_grad()
def predict(model, loader, device, amp):
    model.eval()
    out = []
    for x, _ in loader:
        with torch.autocast(device_type=torch.device(device).type, dtype=torch.float16, enabled=amp):
            out.append(model(prep(x, device)).float().cpu())
    return torch.cat(out).numpy()


def run(variant, backbone, seed, splits_path, image_root, out_root, overrides=None, device="cuda"):
    variant = canon_variant(variant)
    cfg = dict(DEFAULTS)
    cfg.update(PER_BACKBONE.get(backbone, {}))
    cfg.update(overrides or {})
    out = Path(out_root) / run_name(variant, backbone, seed, cfg["region"])
    if (out / "done.flag").exists():
        print("skip (done):", out.name)
        return out
    out.mkdir(parents=True, exist_ok=True)
    seed_all(seed)
    torch.backends.cudnn.benchmark = True

    sp = load_splits(splits_path)
    k = len(class_names(sp))
    data = get_split(sp, variant, ("train", "val"))  # no access to the test part
    by_base = index_images(image_root)
    parts = {}
    for p in ("train", "val"):
        files, y = data[p]
        if cfg["limit"]:
            files, y = files[: cfg["limit"]], y[: cfg["limit"]]
        parts[p] = (files, resolve_paths(files, by_base), y)
    tr = SemDataset(parts["train"][1], parts["train"][2], train=True, crop_scale=cfg["crop_scale"], hw=cfg["hw"], region=cfg["region"])
    va = SemDataset(parts["val"][1], parts["val"][2], train=False, hw=cfg["hw"], region=cfg["region"])
    micro = cfg["batch"] // cfg["accum"]
    tl = make_loader(tr, micro, True, cfg["workers"], seed, True, device)
    vl = make_loader(va, micro * 2, False, cfg["workers"], seed, False, device)

    model, head = build(backbone, k, pretrained=cfg["pretrained"])
    model = model.to(device).to(memory_format=torch.channels_last)
    head_ids = {id(p) for p in head}
    body = [p for p in model.parameters() if id(p) not in head_ids]
    opt = torch.optim.AdamW(
        [{"params": body, "lr": cfg["lr_backbone"]}, {"params": head, "lr": cfg["lr_head"]}],
        weight_decay=cfg["wd"])
    steps_per_epoch = len(tl) // cfg["accum"]
    total, warm = steps_per_epoch * cfg["epochs"], steps_per_epoch * cfg["warmup_epochs"]

    def lr_lambda(s):
        if s < warm:
            return (s + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    amp = str(device).startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    history, best, bad = [], -1.0, 0
    for ep in range(cfg["epochs"]):
        model.train()
        t0 = time.time()
        opt.zero_grad(set_to_none=True)
        tot, n = 0.0, 0
        for it, (x, y) in enumerate(tl):
            y = y.to(device)
            with torch.autocast(device_type=torch.device(device).type, dtype=torch.float16, enabled=amp):
                logits = model(prep(x, device))
            loss = F.cross_entropy(logits.float(), y, label_smoothing=cfg["label_smoothing"])
            scaler.scale(loss / cfg["accum"]).backward()
            if (it + 1) % cfg["accum"] == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"])
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
            tot += loss.item() * len(y)
            n += len(y)
        t_train = time.time() - t0
        logits = predict(model, vl, device, amp)
        yv = parts["val"][2]
        pred = logits.argmax(1)
        f1 = M.macro_f1(yv, pred, k)
        vloss = float(F.cross_entropy(torch.from_numpy(logits), torch.from_numpy(yv)).item())
        history.append(dict(epoch=ep + 1, train_loss=tot / n, val_loss=vloss, val_macro_f1=f1,
                            val_acc=M.accuracy(yv, pred), seconds_train=round(t_train, 1),
                            seconds_total=round(time.time() - t0, 1)))
        print(run_name(variant, backbone, seed, cfg["region"]), history[-1], flush=True)
        (out / "history.json").write_text(json.dumps(history, indent=1))
        if f1 > best:
            best, bad = f1, 0
            torch.save(model.state_dict(), out / "best.pt")
            np.savez(out / "val_logits.npz", names=np.array(parts["val"][0]), y=yv, logits=logits)
        else:
            bad += 1
            if cfg["patience"] and bad >= cfg["patience"]:
                break

    info = dict(cfg=cfg, variant=variant, backbone=backbone, seed=seed, classes=class_names(sp),
                best_val_macro_f1=best, epochs_run=len(history), code_sha256=code_sha(),
                splits_sha256=sha256_file(splits_path), torch=torch.__version__,
                device=torch.cuda.get_device_name(0) if amp else "cpu")
    (out / "config.json").write_text(json.dumps(info, indent=1, default=list))
    (out / "done.flag").write_text("ok")
    return out
