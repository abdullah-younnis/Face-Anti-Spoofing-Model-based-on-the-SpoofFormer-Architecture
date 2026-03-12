"""
Data transforms for SpoofFormer.

Provides standard transforms for training and evaluation.
"""

import torchvision.transforms as T
from spoofformer.data.dataset import IMAGENET_MEAN, IMAGENET_STD


def get_transforms(
    train: bool = True,
    img_size: int = 224,
    augment_strength: str = "normal"
) -> T.Compose:
    """Get transforms for training or evaluation.
    
    Args:
        train: Whether to include training augmentations
        img_size: Target image size
        augment_strength: "light", "normal", or "strong"
    
    Returns:
        Composed transforms
    """
    if train:
        if augment_strength == "strong":
            # Strong augmentation to prevent overfitting on small datasets
            return T.Compose([
                T.Resize((int(img_size * 1.1), int(img_size * 1.1))),
                T.RandomCrop((img_size, img_size)),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(degrees=30),
                T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                T.RandomGrayscale(p=0.1),
                T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
                T.RandomPerspective(distortion_scale=0.2, p=0.3),
                T.ToTensor(),
                T.RandomErasing(p=0.25, scale=(0.02, 0.2)),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
        elif augment_strength == "light":
            return T.Compose([
                T.Resize((img_size, img_size)),
                T.RandomHorizontalFlip(p=0.5),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
        else:  # normal
            return T.Compose([
                T.Resize((img_size, img_size)),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(degrees=15),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
    else:
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
