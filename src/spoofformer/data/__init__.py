"""
SpoofFormer data handling components.

This module contains:
- FASDataset: PyTorch Dataset for face anti-spoofing data
- Augmentation pipelines for training
- Data transforms and preprocessing utilities
"""

from spoofformer.data.dataset import FASDataset, IMAGENET_MEAN, IMAGENET_STD
from spoofformer.data.augmentation import FASAugmentation, apply_fas_augmentation
from spoofformer.data.transforms import get_transforms

__all__ = [
    "FASDataset",
    "FASAugmentation",
    "apply_fas_augmentation",
    "get_transforms",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
]
