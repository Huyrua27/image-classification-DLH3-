from __future__ import annotations

from typing import Any

import timm
import torch.nn as nn
from torchvision.models import (
    EfficientNet_B0_Weights,
    EfficientNet_B2_Weights,
    ResNet50_Weights,
    ViT_B_16_Weights,
    efficientnet_b0,
    efficientnet_b2,
    resnet50,
    vit_b_16,
)


SUPPORTED_MODELS = {
    "resnet50",
    "efficientnet_b0",
    "efficientnet_b2",
    "vit_b_16",
    "deit_small_patch16_224",
}


def _replace_classifier(model: nn.Module, model_name: str, num_classes: int) -> nn.Module:
    if model_name == "resnet50":
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if model_name in {"efficientnet_b0", "efficientnet_b2"}:
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model
    if model_name == "vit_b_16":
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)
        return model
    if model_name == "deit_small_patch16_224":
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, num_classes)
        return model
    raise ValueError(f"Unsupported model name: {model_name}")


def _freeze_backbone(model: nn.Module, model_name: str) -> None:
    for param in model.parameters():
        param.requires_grad = False

    if model_name == "resnet50":
        for param in model.fc.parameters():
            param.requires_grad = True
        return
    if model_name in {"efficientnet_b0", "efficientnet_b2"}:
        for param in model.classifier.parameters():
            param.requires_grad = True
        return
    if model_name == "vit_b_16":
        for param in model.heads.parameters():
            param.requires_grad = True
        return
    if model_name == "deit_small_patch16_224":
        for param in model.head.parameters():
            param.requires_grad = True
        return
    raise ValueError(f"Unsupported model name: {model_name}")


def create_model(config: dict[str, Any], num_classes: int) -> nn.Module:
    model_name = config["name"]
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model. Choose one of: {sorted(SUPPORTED_MODELS)}")

    pretrained = bool(config.get("pretrained", True))
    strict_pretrained = bool(config.get("strict_pretrained", False))

    try:
        if model_name == "resnet50":
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            model = resnet50(weights=weights)
        elif model_name == "efficientnet_b0":
            weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
            model = efficientnet_b0(weights=weights)
        elif model_name == "efficientnet_b2":
            weights = EfficientNet_B2_Weights.DEFAULT if pretrained else None
            model = efficientnet_b2(weights=weights)
        elif model_name == "vit_b_16":
            weights = ViT_B_16_Weights.DEFAULT if pretrained else None
            model = vit_b_16(weights=weights)
        else:
            model = timm.create_model(model_name, pretrained=pretrained)
    except Exception:
        if strict_pretrained or not pretrained:
            raise
        if model_name == "resnet50":
            model = resnet50(weights=None)
        elif model_name == "efficientnet_b0":
            model = efficientnet_b0(weights=None)
        elif model_name == "efficientnet_b2":
            model = efficientnet_b2(weights=None)
        elif model_name == "vit_b_16":
            model = vit_b_16(weights=None)
        else:
            model = timm.create_model(model_name, pretrained=False)

    model = _replace_classifier(model, model_name, num_classes)
    if config.get("freeze_backbone", False):
        _freeze_backbone(model, model_name)
    return model
