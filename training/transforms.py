from __future__ import annotations

from typing import Any

from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_train_transform(config: dict[str, Any]) -> transforms.Compose:
    image_size = int(config["data"]["image_size"])
    aug_cfg = config["data"].get("train_augmentation", {})
    use_aug = bool(aug_cfg.get("enabled", False))

    ops: list[Any] = [transforms.Resize((image_size, image_size))]
    if use_aug:
        ops.extend(
            [
                transforms.RandomHorizontalFlip(p=float(aug_cfg.get("horizontal_flip_prob", 0.5))),
                transforms.RandomRotation(degrees=float(aug_cfg.get("rotation_degrees", 15))),
                transforms.ColorJitter(
                    brightness=float(aug_cfg.get("brightness", 0.2)),
                    contrast=float(aug_cfg.get("contrast", 0.2)),
                    saturation=float(aug_cfg.get("saturation", 0.2)),
                    hue=float(aug_cfg.get("hue", 0.05)),
                ),
                transforms.RandomResizedCrop(
                    size=image_size,
                    scale=tuple(aug_cfg.get("resized_crop_scale", [0.8, 1.0])),
                    ratio=tuple(aug_cfg.get("resized_crop_ratio", [0.9, 1.1])),
                ),
            ]
        )
    ops.extend([transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    return transforms.Compose(ops)


def build_eval_transform(config: dict[str, Any]) -> transforms.Compose:
    image_size = int(config["data"]["image_size"])
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
