"""
Transformer Encoder components for Vision Transformer.

Contains TransformerBlock and TransformerEncoder with support for
intermediate feature extraction.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union


class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention mechanism.
    
    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        attention_dropout: Dropout rate for attention weights
    """
    
    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        attention_dropout: float = 0.0
    ) -> None:
        super().__init__()
        
        assert embed_dim % num_heads == 0, \
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Combined QKV projection
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_drop = nn.Dropout(attention_dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply multi-head self-attention.
        
        Args:
            x: Input tensor of shape [B, N, D]
        
        Returns:
            Output tensor of shape [B, N, D]
        """
        B, N, D = x.shape
        
        # Compute Q, K, V: [B, N, 3*D] -> [B, N, 3, num_heads, head_dim]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        
        # Permute: [B, N, 3, num_heads, head_dim] -> [3, B, num_heads, N, head_dim]
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, D)
        
        # Output projection
        x = self.proj(x)
        
        return x


class MLP(nn.Module):
    """Feed-Forward Network (MLP) for transformer.
    
    Args:
        embed_dim: Input/output dimension
        mlp_ratio: Hidden dimension ratio
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        embed_dim: int = 768,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1
    ) -> None:
        super().__init__()
        
        hidden_dim = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.drop = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply MLP.
        
        Args:
            x: Input tensor of shape [B, N, D]
        
        Returns:
            Output tensor of shape [B, N, D]
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """Single Transformer encoder block with pre-norm.
    
    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        mlp_ratio: MLP hidden dimension ratio
        dropout: Dropout rate for MLP
        attention_dropout: Dropout rate for attention
    """
    
    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attention_dropout: float = 0.0
    ) -> None:
        super().__init__()
        
        # Pre-norm layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            attention_dropout=attention_dropout
        )
        
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(
            embed_dim=embed_dim,
            mlp_ratio=mlp_ratio,
            dropout=dropout
        )
        
        self.drop = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply transformer block.
        
        Args:
            x: Input tensor of shape [B, N, D]
        
        Returns:
            Output tensor of shape [B, N, D]
        """
        # Self-attention with residual
        x = x + self.drop(self.attn(self.norm1(x)))
        
        # MLP with residual
        x = x + self.drop(self.mlp(self.norm2(x)))
        
        return x


class TransformerEncoder(nn.Module):
    """Transformer encoder with intermediate feature extraction.
    
    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer blocks
        mlp_ratio: MLP hidden dimension ratio
        dropout: Dropout rate
        attention_dropout: Attention dropout rate
        extract_layers: Layer indices to extract intermediate features from
    """
    
    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attention_dropout: float = 0.0,
        extract_layers: Optional[List[int]] = None
    ) -> None:
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.extract_layers = extract_layers or [8, 11]
        
        # Validate extract_layers
        for layer_idx in self.extract_layers:
            if layer_idx >= num_layers:
                raise ValueError(
                    f"extract_layers value ({layer_idx}) must be less than "
                    f"num_layers ({num_layers})"
                )
        
        # Stack of transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attention_dropout=attention_dropout
            )
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        return_intermediate: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[int, torch.Tensor]]]:
        """Apply transformer encoder.
        
        Args:
            x: Input tensor of shape [B, N, D]
            return_intermediate: Whether to return intermediate features
        
        Returns:
            If return_intermediate=False: Output tensor [B, N, D]
            If return_intermediate=True: Tuple of (output, intermediate_features)
                where intermediate_features is dict mapping layer_idx to CLS token [B, D]
        """
        intermediate_features: Dict[int, torch.Tensor] = {}
        
        for layer_idx, block in enumerate(self.blocks):
            x = block(x)
            
            # Extract intermediate features (CLS token) if needed
            if return_intermediate and layer_idx in self.extract_layers:
                intermediate_features[layer_idx] = x[:, 0, :]  # CLS token
        
        # Apply final layer norm
        x = self.norm(x)
        
        if return_intermediate:
            return x, intermediate_features
        return x
