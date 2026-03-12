"""
Classification Head for binary liveness prediction.

Maps the CLS token representation to real/spoof classification.
"""

import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """Binary classification head for face anti-spoofing.
    
    Takes the CLS token from transformer output and produces
    binary classification logits.
    
    Args:
        embed_dim: Input embedding dimension
        hidden_dim: Hidden layer dimension
        num_classes: Number of output classes (1 for binary)
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        embed_dim: int = 768,
        hidden_dim: int = 256,
        num_classes: int = 1,
        dropout: float = 0.1
    ) -> None:
        super().__init__()
        
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # MLP classification head
        self.head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify CLS token.
        
        Args:
            x: Transformer output of shape [B, N, D] or CLS token [B, D]
        
        Returns:
            Logits of shape [B, num_classes]
        """
        # If full sequence is passed, extract CLS token (position 0)
        if x.dim() == 3:
            x = x[:, 0, :]
        
        return self.head(x)
