"""
SpoofFormer: Vision Transformer for Face Anti-Spoofing.

Main model integrating PatchEmbedding, TransformerEncoder, and ClassificationHead.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union

from spoofformer.models.patch_embedding import PatchEmbedding
from spoofformer.models.transformer import TransformerEncoder
from spoofformer.models.classification_head import ClassificationHead


class SpoofFormer(nn.Module):
    """Vision Transformer for face anti-spoofing.
    
    Integrates patch embedding, transformer encoder, and classification head
    for end-to-end real/spoof detection.
    
    Args:
        img_size: Input image size
        patch_size: Patch size for embedding
        in_channels: Number of input channels
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        mlp_ratio: MLP hidden dimension ratio
        dropout: Dropout rate
        attention_dropout: Attention dropout rate
        extract_layers: Layers for intermediate feature extraction
        hidden_dim: Classification head hidden dimension
    """
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attention_dropout: float = 0.0,
        extract_layers: Optional[List[int]] = None,
        hidden_dim: int = 256
    ) -> None:
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.extract_layers = extract_layers or [8, 11]
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )
        
        # Transformer encoder
        self.encoder = TransformerEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attention_dropout=attention_dropout,
            extract_layers=self.extract_layers
        )
        
        # Classification head
        self.head = ClassificationHead(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_classes=1,
            dropout=dropout
        )
        
        # Auxiliary classifiers for intermediate supervision
        self.aux_heads = nn.ModuleDict({
            str(layer_idx): nn.Linear(embed_dim, 1)
            for layer_idx in self.extract_layers
        })
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m: nn.Module) -> None:
        """Initialize module weights."""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        return_intermediate: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[int, torch.Tensor]]]:
        """Forward pass through SpoofFormer.
        
        Args:
            x: Input images of shape [B, C, H, W]
            return_intermediate: Whether to return intermediate features
        
        Returns:
            If return_intermediate=False: Logits [B, 1]
            If return_intermediate=True: Tuple of (logits, intermediate_features)
        """
        # Patch embedding
        x = self.patch_embed(x)
        
        # Transformer encoding
        if return_intermediate:
            x, intermediate_features = self.encoder(x, return_intermediate=True)
        else:
            x = self.encoder(x, return_intermediate=False)
        
        # Classification
        logits = self.head(x)
        
        if return_intermediate:
            return logits, intermediate_features
        return logits
    
    def load_pretrained(self, checkpoint_path: str, strict: bool = False) -> None:
        """Load pretrained weights from checkpoint.
        
        Handles mismatched classification head dimensions by skipping
        incompatible weights.
        
        Args:
            checkpoint_path: Path to checkpoint file
            strict: Whether to strictly enforce matching keys
        """
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'model' in state_dict:
            state_dict = state_dict['model']
        elif 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        
        # Filter out incompatible keys (classification head)
        model_state = self.state_dict()
        filtered_state = {}
        
        for k, v in state_dict.items():
            # Skip classification head weights
            if 'head' in k or 'aux_heads' in k:
                continue
            
            # Handle key name differences
            if k in model_state and v.shape == model_state[k].shape:
                filtered_state[k] = v
        
        # Load filtered weights
        missing, unexpected = self.load_state_dict(filtered_state, strict=False)
        
        print(f"Loaded pretrained weights from {checkpoint_path}")
        print(f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
    
    @classmethod
    def from_config(cls, config) -> "SpoofFormer":
        """Create SpoofFormer from ModelConfig.
        
        Args:
            config: ModelConfig instance
        
        Returns:
            SpoofFormer model
        """
        return cls(
            img_size=config.img_size,
            patch_size=config.patch_size,
            in_channels=config.in_channels,
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            mlp_ratio=config.mlp_ratio,
            dropout=config.dropout,
            attention_dropout=config.attention_dropout,
            extract_layers=config.extract_layers
        )
