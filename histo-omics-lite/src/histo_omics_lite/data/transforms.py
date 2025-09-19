"""Image transforms used by the project."""
from __future__ import annotations

from typing import Callable

import torch
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_simclr_transform(size: int = 224) -> Callable[[object], torch.Tensor]:
    color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
    gaussian_blur = transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))

    return transforms.Compose(
        [
            transforms.RandomResizedCrop(size=size, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            gaussian_blur,
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def get_eval_transform(size: int = 224) -> Callable[[object], torch.Tensor]:
    return transforms.Compose(
        [
            transforms.Resize(size + 32),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
