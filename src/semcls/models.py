"""Backbones: ImageNet-pretrained, new linear head."""
import torch.nn as nn
import torchvision as tv


def build(name, num_classes, pretrained=True):
    """Returns (model, head_parameters)."""
    if name == "resnet50":
        w = tv.models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        m = tv.models.resnet50(weights=w)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        head = m.fc
    elif name == "convnext_tiny":
        w = tv.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        m = tv.models.convnext_tiny(weights=w)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, num_classes)
        head = m.classifier[2]
    else:
        raise ValueError(f"unknown backbone {name!r}")
    return m, list(head.parameters())
