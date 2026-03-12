"""
Model quantization for SpoofFormer.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def quantize_model(
    input_path: str,
    output_path: str,
    quantization_type: str = "int8"
) -> None:
    """Quantize ONNX model to INT8.
    
    Args:
        input_path: Path to input ONNX model
        output_path: Path to save quantized model
        quantization_type: Quantization type ("int8")
    """
    from onnxruntime.quantization import quantize_dynamic, QuantType
    
    # Create output directory
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Quantize
    if quantization_type == "int8":
        quantize_dynamic(
            input_path,
            output_path,
            weight_type=QuantType.QInt8
        )
    else:
        raise ValueError(f"Unknown quantization type: {quantization_type}")
    
    logger.info(f"Quantized model saved to {output_path}")
