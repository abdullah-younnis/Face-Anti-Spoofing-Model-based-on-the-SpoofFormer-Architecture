"""
Configuration dataclasses for SpoofFormer.

This module contains:
- ModelConfig: Architecture hyperparameters
- TrainingConfig: Training hyperparameters
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class ModelConfig:
    """Configuration for SpoofFormer model architecture.
    
    Attributes:
        img_size: Input image size (must be divisible by patch_size)
        patch_size: Size of each patch
        in_channels: Number of input channels (3 for RGB)
        embed_dim: Embedding dimension (must be divisible by num_heads)
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        mlp_ratio: MLP hidden dimension ratio
        dropout: Dropout rate for FFN
        attention_dropout: Dropout rate for attention weights
        extract_layers: Layers to extract intermediate features from
    """
    img_size: int = 224
    patch_size: int = 16
    in_channels: int = 3
    embed_dim: int = 768
    num_heads: int = 12
    num_layers: int = 12
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    attention_dropout: float = 0.0
    extract_layers: List[int] = field(default_factory=lambda: [8, 11])
    
    def __post_init__(self):
        """Validate configuration parameters."""
        # Validate img_size divisible by patch_size
        if self.img_size % self.patch_size != 0:
            raise ValueError(
                f"img_size ({self.img_size}) must be divisible by "
                f"patch_size ({self.patch_size})"
            )
        
        # Validate embed_dim divisible by num_heads
        if self.embed_dim % self.num_heads != 0:
            raise ValueError(
                f"embed_dim ({self.embed_dim}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )
        
        # Validate extract_layers < num_layers
        for layer_idx in self.extract_layers:
            if layer_idx >= self.num_layers:
                raise ValueError(
                    f"extract_layers value ({layer_idx}) must be less than "
                    f"num_layers ({self.num_layers})"
                )
    
    @classmethod
    def vit_tiny(cls) -> "ModelConfig":
        """Create ViT-Tiny configuration."""
        return cls(embed_dim=192, num_heads=3, num_layers=12)
    
    @classmethod
    def vit_small(cls) -> "ModelConfig":
        """Create ViT-Small configuration."""
        return cls(embed_dim=384, num_heads=6, num_layers=12)
    
    @classmethod
    def vit_base(cls) -> "ModelConfig":
        """Create ViT-Base configuration."""
        return cls(embed_dim=768, num_heads=12, num_layers=12)
    
    @classmethod
    def vit_large(cls) -> "ModelConfig":
        """Create ViT-Large configuration."""
        return cls(embed_dim=1024, num_heads=16, num_layers=24, extract_layers=[16, 23])
    
    @classmethod
    def mobile(cls) -> "ModelConfig":
        """Create lightweight mobile configuration."""
        return cls(embed_dim=192, num_heads=3, num_layers=6, dropout=0.0, extract_layers=[3, 5])
    
    @property
    def num_patches(self) -> int:
        """Calculate number of patches."""
        return (self.img_size // self.patch_size) ** 2
    
    @property
    def head_dim(self) -> int:
        """Calculate dimension per attention head."""
        return self.embed_dim // self.num_heads


@dataclass
class TrainingConfig:
    """Configuration for training pipeline.
    
    Attributes:
        data_root: Root directory for dataset
        batch_size: Training batch size
        num_workers: Number of data loading workers
        learning_rate: Initial learning rate
        weight_decay: Weight decay for optimizer
        epochs: Total training epochs
        warmup_epochs: Number of warmup epochs
        bce_weight: Weight for BCE loss
        intermediate_weight: Weight for intermediate supervision loss
        checkpoint_dir: Directory to save checkpoints
        save_every: Save checkpoint every N epochs
        seed: Random seed for reproducibility
    """
    data_root: str = "dataset"
    batch_size: int = 32
    num_workers: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.05
    epochs: int = 100
    warmup_epochs: int = 5
    bce_weight: float = 1.0
    intermediate_weight: float = 0.5
    checkpoint_dir: str = "checkpoints"
    save_every: int = 5
    seed: int = 42
    
    def __post_init__(self):
        """Validate configuration parameters."""
        # Validate learning_rate is positive
        if self.learning_rate <= 0:
            raise ValueError(
                f"learning_rate ({self.learning_rate}) must be positive"
            )
        
        # Validate batch_size is positive integer
        if not isinstance(self.batch_size, int) or self.batch_size <= 0:
            raise ValueError(
                f"batch_size ({self.batch_size}) must be a positive integer"
            )
