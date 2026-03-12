"""
Preprocessing utilities for inference.
"""

from typing import Optional, Tuple

import numpy as np
import torch
import torchvision.transforms as T

from spoofformer.data.dataset import IMAGENET_MEAN, IMAGENET_STD


def preprocess_image(
    image: np.ndarray,
    img_size: int = 224
) -> torch.Tensor:
    """Preprocess image for model inference.
    
    Args:
        image: Input image as numpy array [H, W, C] with values in [0, 255]
        img_size: Target image size
    
    Returns:
        Preprocessed tensor [1, C, H, W]
    """
    # Convert to PIL-like format if needed
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    # Convert to tensor and normalize
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    
    tensor = transform(image)
    return tensor.unsqueeze(0)  # Add batch dimension


def detect_and_align_face(
    image: np.ndarray,
    face_detector: Optional[object] = None
) -> Optional[np.ndarray]:
    """Detect and align face in image.
    
    Args:
        image: Input image [H, W, C]
        face_detector: Optional face detector
    
    Returns:
        Cropped face image or None if no face detected
    """
    if face_detector is None:
        return image
    
    try:
        # Face detector should return cropped face
        result = face_detector(image)
        return result
    except Exception:
        return None
