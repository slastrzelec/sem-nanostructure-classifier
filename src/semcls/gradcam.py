"""Grad-CAM for ConvNeXt-Tiny (torch only). Illustrative: it shows where the gradient of the predicted logit is
largest in the last feature map, not what the model "uses" (SPEC section 11, E3 and demo entries)."""
import numpy as np
import torch
import torch.nn.functional as F


def grad_cam(model, x, target=None, layer=None):
    """x: normalised batch of one (1, 3, H, W). Returns (logits as a 1-D CPU tensor, cam as H x W float array in [0, 1]).
    target: class index (default: the predicted class). layer: module whose output is used (default: model.features[-1])."""
    layer = layer if layer is not None else model.features[-1]
    store = {}

    def fwd(_m, _i, out):
        store["act"] = out
        out.register_hook(lambda g: store.__setitem__("grad", g))

    h = layer.register_forward_hook(fwd)
    was_training = model.training
    model.eval()
    try:
        with torch.enable_grad():
            x = x.detach().clone().requires_grad_(True)
            logits = model(x)
            k = int(logits.argmax(1)) if target is None else int(target)
            model.zero_grad(set_to_none=True)
            logits[0, k].backward()
    finally:
        h.remove()
        model.train(was_training)
    act, grad = store["act"].detach(), store["grad"].detach()
    w = grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * act).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)[0, 0]
    cam = cam.cpu().numpy().astype(np.float64)
    m = cam.max()
    cam = cam / m if m > 0 else np.zeros_like(cam)
    return logits.detach().float().cpu()[0], cam
