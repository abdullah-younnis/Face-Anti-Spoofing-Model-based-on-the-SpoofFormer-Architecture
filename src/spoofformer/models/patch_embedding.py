"""
Patch Embedding module for Vision Transformer.

Converts input images into a sequence of patch embeddings with
CLS token and positional embeddings.
"""

import torch
import torch.nn as nn
from typing import Optional


class PatchEmbedding(nn.Module):
    """Convert images to patch embeddings for transformer processing.
    
    This module:
    1. Partitions input image into non-overlapping patches
    2. Projects patches to embedding dimension via Conv2d
    3. Prepends learnable CLS token
    4. Adds learnable positional embeddings
    
    Args:
        img_size: Input image size (must be divisible by patch_size)
        patch_size: Size of each patch
        in_channels: Number of input channels (3 for RGB)
        embed_dim: Embedding dimension
    
    Raises:
        ValueError: If img_size is not divisible by patch_size
    """
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768
    ) -> None:
        super().__init__()
        
        # Validate img_size divisibility
        if img_size % patch_size != 0:
            raise ValueError(
                f"img_size ({img_size}) must be divisible by "
                f"patch_size ({patch_size})"
            )
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2
        
        # Patch projection using Conv2d (equivalent to linear projection)
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        
        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Learnable positional embeddings for num_patches + 1 (CLS token)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embed_dim)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize CLS token and positional embeddings."""
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convert images to patch embeddings.
        
        Args:
            x: Input images of shape [B, C, H, W]
        
        Returns:
            Patch embeddings of shape [B, num_patches + 1, embed_dim]
        """
        B = x.shape[0]
        
        # Project patches: [B, C, H, W] -> [B, embed_dim, H/P, W/P]
        x = self.proj(x)
        
        # Flatten spatial dimensions: [B, embed_dim, H/P, W/P] -> [B, embed_dim, num_patches]
        x = x.flatten(2)
        
        # Transpose: [B, embed_dim, num_patches] -> [B, num_patches, embed_dim]
        x = x.transpose(1, 2)
        
        # Expand CLS token for batch: [1, 1, embed_dim] -> [B, 1, embed_dim]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        
        # Prepend CLS token: [B, num_patches + 1, embed_dim]
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional embeddings
        x = x + self.pos_embed
        
        return x
