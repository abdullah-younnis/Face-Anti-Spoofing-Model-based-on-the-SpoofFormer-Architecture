"""
ONNX export functionality for SpoofFormer.
"""

import logging
from pathlib import Path
from typing import Optional

import torch
import numpy as np

logger = logging.getLogger(__name__)


class ExportError(Exception):
    """Error during model export."""
    pass


def export_onnx(
    model: torch.nn.Module,
    output_path: str,
    img_size: int = 224,
    opset_version: int = 14,
    dynamic_batch: bool = True
) -> None:
    """Export SpoofFormer to ONNX format.
    
    Args:
        model: SpoofFormer model
        output_path: Path to save ONNX model
        img_size: Input image size
        opset_version: ONNX opset version (>= 14 recommended)
        dynamic_batch: Whether to use dynamic batch dimension
    
    Raises:
        ExportError: If export fails
    """
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, img_size, img_size)
    
    # Dynamic axes for batch dimension
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    
    # Create output directory
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dynamic_axes
        )
        logger.info(f"Exported ONNX model to {output_path}")
        
    except Exception as e:
        raise ExportError(f"ONNX export failed: {e}")


def verify_onnx_export(
    pytorch_model: torch.nn.Module,
    onnx_path: str,
    img_size: int = 224,
    tolerance: float = 1e-5
) -> bool:
    """Verify ONNX model produces equivalent outputs.
    
    Args:
        pytorch_model: Original PyTorch model
        onnx_path: Path to ONNX model
        img_size: Input image size
        tolerance: Maximum allowed difference
    
    Returns:
        True if outputs match within tolerance
    """
    import onnxruntime as ort
    
    pytorch_model.eval()
    
    # Create test input
    test_input = torch.randn(1, 3, img_size, img_size)
    
    # PyTorch inference
    with torch.no_grad():
        pytorch_output = pytorch_model(test_input, return_intermediate=False)
    
    # ONNX inference
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    onnx_output = session.run(None, {input_name: test_input.numpy()})[0]
    
    # Compare outputs
    diff = np.abs(pytorch_output.numpy() - onnx_output).max()
    
    if diff <= tolerance:
        logger.info(f"ONNX verification passed (max diff: {diff:.2e})")
        return True
    else:
        logger.warning(f"ONNX verification failed (max diff: {diff:.2e})")
        return False
