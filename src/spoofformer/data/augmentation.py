"""
FAS-specific data augmentation pipeline.

Augmentations designed for face anti-spoofing to improve
robustness against various attack types.
"""

import random
from typing import Optional

import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as TF


class FASAugmentation(nn.Module):
    """Face Anti-Spoofing augmentation pipeline.
    
    Applies FAS-specific augmentations:
    - Color jittering
    - Gaussian blur
    - Random noise
    - Random cutout/erasing
    
    Args:
        p_color: Probability of color jittering
        p_blur: Probability of Gaussian blur
        p_noise: Probability of random noise
        p_cutout: Probability of random cutout
    """
    
    def __init__(
        self,
        p_color: float = 0.5,
        p_blur: float = 0.3,
        p_noise: float = 0.3,
        p_cutout: float = 0.2
    ) -> None:
        super().__init__()
        
        self.p_color = p_color
        self.p_blur = p_blur
        self.p_noise = p_noise
        self.p_cutout = p_cutout
        
        # Color jitter transform
        self.color_jitter = T.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.1
        )
        
        # Gaussian blur
        self.blur = T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))
        
        # Random erasing (cutout)
        self.erasing = T.RandomErasing(
            p=1.0,  # We control probability externally
            scale=(0.02, 0.2),
            ratio=(0.3, 3.3)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply augmentations.
        
        Args:
            x: Input tensor [C, H, W] or [B, C, H, W] with values in [0, 1]
        
        Returns:
            Augmented tensor with same shape, values clamped to [0, 1]
        """
        # Handle batch dimension
        if x.dim() == 4:
            return torch.stack([self._augment_single(img) for img in x])
        return self._augment_single(x)
    
    def _augment_single(self, x: torch.Tensor) -> torch.Tensor:
        """Augment a single image.
        
        Args:
            x: Input tensor [C, H, W]
        
        Returns:
            Augmented tensor [C, H, W]
        """
        # Color jittering
        if random.random() < self.p_color:
            x = self.color_jitter(x)
        
        # Gaussian blur
        if random.random() < self.p_blur:
            x = self.blur(x)
        
        # Random noise
        if random.random() < self.p_noise:
            noise = torch.randn_like(x) * 0.05
            x = x + noise
        
        # Random cutout
        if random.random() < self.p_cutout:
            x = self.erasing(x)
        
        # Clamp to valid range
        x = torch.clamp(x, 0.0, 1.0)
        
        return x


def apply_fas_augmentation(
    image: torch.Tensor,
    p_color: float = 0.5,
    p_blur: float = 0.3,
    p_noise: float = 0.3,
    p_cutout: float = 0.2
) -> torch.Tensor:
    """Apply FAS-specific augmentations to image.
    
    Args:
        image: Input tensor [C, H, W] or [B, C, H, W]
        p_color: Probability of color jittering
        p_blur: Probability of Gaussian blur
        p_noise: Probability of random noise
        p_cutout: Probability of random cutout
    
    Returns:
        Augmented tensor with same shape
    """
    aug = FASAugmentation(
        p_color=p_color,
        p_blur=p_blur,
        p_noise=p_noise,
        p_cutout=p_cutout
    )
    return aug(image)
