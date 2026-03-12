"""
SpoofFormer inference components.

This module contains:
- InferenceEngine: Production inference with multiple backends
- LivenessResult: Structured inference output
- Preprocessing pipeline for inference
"""

from spoofformer.inference.engine import InferenceEngine
from spoofformer.inference.result import LivenessResult
from spoofformer.inference.preprocessing import preprocess_image, detect_and_align_face

__all__ = [
    "InferenceEngine",
    "LivenessResult",
    "preprocess_image",
    "detect_and_align_face",
]
