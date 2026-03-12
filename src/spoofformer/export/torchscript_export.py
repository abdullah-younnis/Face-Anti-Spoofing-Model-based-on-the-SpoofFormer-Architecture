"""
TorchScript export functionality for SpoofFormer.
"""

import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def export_torchscript(
    model: torch.nn.Module,
    output_path: str,
    img_size: int = 224
) -> None:
    """Export SpoofFormer to TorchScript via tracing.
    
    Args:
        model: SpoofFormer model
        output_path: Path to save TorchScript model
        img_size: Input image size
    """
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, img_size, img_size)
    
    # Create output directory
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Trace model
    with torch.no_grad():
        traced_model = torch.jit.trace(model, dummy_input)
    
    # Save
    traced_model.save(output_path)
    logger.info(f"Exported TorchScript model to {output_path}")
