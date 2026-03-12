"""
Face Anti-Spoofing Dataset.

PyTorch Dataset for loading face anti-spoofing data from
directory structure with real/ and spoof/ subfolders.
"""

import logging
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image

logger = logging.getLogger(__name__)

# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class FASDataset(Dataset):
    """Face Anti-Spoofing Dataset.
    
    Loads images from directory structure:
    ```
    root_dir/
        real/
            img1.jpg
            img2.png
            ...
        spoof/
            img1.jpg
            img2.png
            ...
    ```
    
    Args:
        root_dir: Root directory containing real/ and spoof/ folders
        transform: Optional transform to apply to images
        face_detector: Optional face detector for cropping faces
    """
    
    SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    
    def __init__(
        self,
        root_dir: str,
        transform: Optional[Callable] = None,
        face_detector: Optional[Callable] = None
    ) -> None:
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.face_detector = face_detector
        
        # Load image paths and labels
        self.samples: List[Tuple[Path, int]] = []
        self._load_samples()
    
    def _load_samples(self) -> None:
        """Load image paths from real/ and spoof/ directories."""
        real_dir = self.root_dir / "real"
        spoof_dir = self.root_dir / "spoof"
        
        # Load real images (label=1)
        if real_dir.exists():
            for img_path in real_dir.iterdir():
                if img_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                    self.samples.append((img_path, 1))
        
        # Load spoof images (label=0)
        if spoof_dir.exists():
            for img_path in spoof_dir.iterdir():
                if img_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                    self.samples.append((img_path, 0))
        
        logger.info(
            f"Loaded {len(self.samples)} samples from {self.root_dir} "
            f"(real: {sum(1 for _, l in self.samples if l == 1)}, "
            f"spoof: {sum(1 for _, l in self.samples if l == 0)})"
        )
    
    def __len__(self) -> int:
        return len(self.samples)
    
    @property
    def labels(self) -> List[int]:
        """Get list of all labels."""
        return [label for _, label in self.samples]
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get image and label.
        
        Args:
            idx: Sample index
        
        Returns:
            Tuple of (image_tensor, label) where label=1 for real, 0 for spoof
        """
        img_path, label = self.samples[idx]
        
        try:
            # Load image
            image = Image.open(img_path).convert('RGB')
            
            # Apply face detection if provided
            if self.face_detector is not None:
                image = self._detect_and_crop_face(image)
            
            # Apply transforms
            if self.transform is not None:
                image = self.transform(image)
            
            return image, label
            
        except Exception as e:
            logger.warning(f"Error loading {img_path}: {e}")
            # Return a placeholder on error
            if self.transform is not None:
                # Try to get a valid sample
                return self.__getitem__((idx + 1) % len(self))
            raise
    
    def _detect_and_crop_face(self, image: Image.Image) -> Image.Image:
        """Detect and crop face from image.
        
        Args:
            image: PIL Image
        
        Returns:
            Cropped face image or original if no face detected
        """
        if self.face_detector is None:
            return image
        
        try:
            # Face detector should return bounding box or cropped image
            result = self.face_detector(image)
            if result is not None:
                return result
        except Exception as e:
            logger.warning(f"Face detection failed: {e}")
        
        return image
