#!/usr/bin/env python3
"""
Training script for SpoofFormer.

Usage:
    python train.py --data_root dataset --epochs 100 --batch_size 32
    
    # Use predefined model config
    python train.py --config configs/model_configs.yaml --model_version small
    
    # List available model versions
    python train.py --config configs/model_configs.yaml --list_versions
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent / "src"))

from torch.utils.data import DataLoader, WeightedRandomSampler

from spoofformer.models import SpoofFormer
from spoofformer.config import ModelConfig, TrainingConfig
from spoofformer.data import FASDataset, get_transforms
from spoofformer.training import Trainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model_config_from_yaml(config_path: str, version: str) -> ModelConfig:
    """Load model configuration from YAML file."""
    with open(config_path, 'r') as f:
        configs = yaml.safe_load(f)
    
    if version not in configs:
        available = list(configs.keys())
        raise ValueError(f"Unknown model version '{version}'. Available: {available}")
    
    cfg = configs[version]
    return ModelConfig(
        img_size=cfg.get('img_size', 224),
        patch_size=cfg.get('patch_size', 16),
        in_channels=cfg.get('in_channels', 3),
        embed_dim=cfg['embed_dim'],
        num_heads=cfg['num_heads'],
        num_layers=cfg['num_layers'],
        mlp_ratio=cfg.get('mlp_ratio', 4.0),
        dropout=cfg.get('dropout', 0.1),
        attention_dropout=cfg.get('attention_dropout', 0.0),
        extract_layers=cfg.get('extract_layers', [8, 11])
    )


def list_model_versions(config_path: str) -> None:
    """Print available model versions from config file."""
    with open(config_path, 'r') as f:
        configs = yaml.safe_load(f)
    
    print("\nAvailable model versions:")
    print("-" * 60)
    for name, cfg in configs.items():
        params = cfg['embed_dim'] * cfg['num_layers'] * 4  # rough estimate
        print(f"  {name:12} - embed_dim={cfg['embed_dim']}, "
              f"heads={cfg['num_heads']}, layers={cfg['num_layers']}")
    print("-" * 60)
    print("\nUsage: python train.py --config configs/model_configs.yaml --model_version <name>")


def parse_args():
    parser = argparse.ArgumentParser(description='Train SpoofFormer')
    
    # Data
    parser.add_argument('--data_root', type=str, default='dataset',
                        help='Root directory for dataset')
    parser.add_argument('--img_size', type=int, default=None,
                        help='Input image size (overrides config)')
    
    # Model configuration
    parser.add_argument('--config', type=str, default=None,
                        help='Path to model config YAML file')
    parser.add_argument('--model_version', type=str, default='small',
                        help='Model version from config file')
    parser.add_argument('--list_versions', action='store_true',
                        help='List available model versions and exit')
    
    # Legacy model size option (for backward compatibility)
    parser.add_argument('--model_size', type=str, default=None,
                        choices=['small', 'base'],
                        help='Model size (deprecated, use --config instead)')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Checkpointing
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--save_every', type=int, default=5,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--early_stopping', type=int, default=10,
                        help='Early stopping patience (0 to disable)')
    
    # Reproducibility
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # WandB
    parser.add_argument('--wandb_project', type=str, default=None,
                        help='WandB project name (None to disable)')
    parser.add_argument('--wandb_run', type=str, default=None,
                        help='WandB run name (optional)')
    
    # Class imbalance handling
    parser.add_argument('--balance_classes', action='store_true',
                        help='Use weighted sampling to balance classes')
    parser.add_argument('--focal_loss', action='store_true',
                        help='Use focal loss instead of BCE')
    parser.add_argument('--class_weight', action='store_true',
                        help='Use class-weighted BCE loss')
    
    # Regularization / Anti-overfitting
    parser.add_argument('--augment', type=str, default='normal',
                        choices=['light', 'normal', 'strong'],
                        help='Data augmentation strength (strong recommended for small datasets)')
    parser.add_argument('--label_smoothing', type=float, default=0.0,
                        help='Label smoothing factor (0.1-0.2 recommended to prevent overconfidence)')
    parser.add_argument('--dropout', type=float, default=None,
                        help='Override model dropout (higher values like 0.3 reduce overfitting)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Handle --list_versions
    if args.list_versions:
        config_path = args.config or 'configs/model_configs.yaml'
        if not Path(config_path).exists():
            logger.error(f"Config file not found: {config_path}")
            return 1
        list_model_versions(config_path)
        return 0
    
    # Create model config
    if args.config:
        # Load from YAML config file
        if not Path(args.config).exists():
            logger.error(f"Config file not found: {args.config}")
            return 1
        model_config = load_model_config_from_yaml(args.config, args.model_version)
        logger.info(f"Loaded model config '{args.model_version}' from {args.config}")
    elif args.model_size:
        # Legacy: use built-in configs
        if args.model_size == 'small':
            model_config = ModelConfig.vit_small()
        else:
            model_config = ModelConfig.vit_base()
        logger.info(f"Using built-in config: {args.model_size}")
    else:
        # Default: try to load from default config file, fallback to built-in
        default_config = Path('configs/model_configs.yaml')
        if default_config.exists():
            model_config = load_model_config_from_yaml(str(default_config), args.model_version)
            logger.info(f"Loaded model config '{args.model_version}' from {default_config}")
        else:
            model_config = ModelConfig.vit_small()
            logger.info("Using built-in config: small")
    
    # Override img_size if specified
    if args.img_size:
        model_config.img_size = args.img_size
    
    # Override dropout if specified (for regularization)
    if args.dropout is not None:
        model_config.dropout = args.dropout
        logger.info(f"Using dropout: {args.dropout}")
    
    # Create training config
    train_config = TrainingConfig(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        learning_rate=args.lr,
        epochs=args.epochs,
        checkpoint_dir=args.checkpoint_dir,
        save_every=args.save_every,
        seed=args.seed
    )
    
    # Create model
    logger.info(f"Creating SpoofFormer (embed_dim={model_config.embed_dim}, "
                f"heads={model_config.num_heads}, layers={model_config.num_layers})")
    model = SpoofFormer.from_config(model_config)
    
    # Create datasets
    img_size = args.img_size or model_config.img_size
    train_transform = get_transforms(train=True, img_size=img_size, augment_strength=args.augment)
    val_transform = get_transforms(train=False, img_size=img_size)
    
    if args.augment == 'strong':
        logger.info("Using STRONG data augmentation (recommended for small datasets)")
    
    train_dir = Path(args.data_root) / 'train'
    val_dir = Path(args.data_root) / 'val'
    
    if not train_dir.exists():
        logger.error(f"Training directory not found: {train_dir}")
        logger.info("Expected structure: dataset/train/real/, dataset/train/spoof/")
        return 1
    
    train_dataset = FASDataset(str(train_dir), transform=train_transform)
    
    # Calculate class weights for imbalance handling
    labels = train_dataset.labels
    num_real = sum(labels)
    num_spoof = len(labels) - num_real
    total = len(labels)
    
    logger.info(f"Training set: {num_real} real, {num_spoof} spoof (ratio: {num_spoof/num_real:.2f}:1)")
    
    # Class weights: inverse frequency
    weight_real = total / (2 * num_real)
    weight_spoof = total / (2 * num_spoof)
    
    class_weights = None
    if args.class_weight or args.focal_loss:
        class_weights = torch.tensor([weight_spoof, weight_real])
        logger.info(f"Class weights: spoof={weight_spoof:.2f}, real={weight_real:.2f}")
    
    # Weighted sampler for balanced batches
    sampler = None
    shuffle = True
    if args.balance_classes:
        sample_weights = [weight_real if label == 1 else weight_spoof for label in labels]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        shuffle = False
        logger.info("Using weighted random sampler for class balancing")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = None
    if val_dir.exists():
        val_dataset = FASDataset(str(val_dir), transform=val_transform)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )
    
    # Create trainer and train
    trainer = Trainer(
        model, 
        train_config,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run,
        model_config=model_config.__dict__,
        class_weights=class_weights if args.class_weight else None,
        use_focal_loss=args.focal_loss,
        label_smoothing=args.label_smoothing
    )
    early_stopping = args.early_stopping if args.early_stopping > 0 else None
    trainer.fit(train_loader, val_loader, early_stopping=early_stopping)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())


