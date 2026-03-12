"""
Training pipeline for SpoofFormer.

Implements training loop with intermediate supervision,
gradient clipping, checkpointing, and WandB logging.
"""

import logging
import os
import random
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from spoofformer.config import TrainingConfig
from spoofformer.training.metrics import compute_metrics, EvaluationMetrics

logger = logging.getLogger(__name__)

# Optional WandB import
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    wandb = None


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class WarmupCosineScheduler:
    """Learning rate scheduler with warmup and cosine decay."""
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_lr: float = 1e-6
    ) -> None:
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]
        self.current_epoch = 0
    
    def step(self) -> None:
        """Update learning rate."""
        self.current_epoch += 1
        
        if self.current_epoch <= self.warmup_epochs:
            # Linear warmup
            scale = self.current_epoch / self.warmup_epochs
        else:
            # Cosine decay
            progress = (self.current_epoch - self.warmup_epochs) / (
                self.total_epochs - self.warmup_epochs
            )
            scale = 0.5 * (1 + np.cos(np.pi * progress))
        
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg['lr'] = max(base_lr * scale, self.min_lr)
    
    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance.
    
    Focuses on hard examples by down-weighting easy ones.
    FL(p) = -alpha * (1-p)^gamma * log(p)
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        ce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        
        # p_t = p if y=1 else 1-p
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        # Focal weight
        focal_weight = (1 - p_t) ** self.gamma
        
        # Alpha weighting
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        loss = alpha_t * focal_weight * ce_loss
        return loss.mean()


class Trainer:
    """Training pipeline for SpoofFormer.
    
    Args:
        model: SpoofFormer model
        config: Training configuration
        device: Device to train on
        wandb_project: WandB project name (None to disable)
        wandb_run_name: WandB run name (optional)
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        device: Optional[str] = None,
        wandb_project: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
        model_config: Optional[Dict] = None,
        class_weights: Optional[torch.Tensor] = None,
        use_focal_loss: bool = False,
        label_smoothing: float = 0.0
    ) -> None:
        self.model = model
        self.config = config
        self.model_config = model_config or {}  # Store model config for checkpoint
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.wandb_enabled = False
        self.label_smoothing = label_smoothing
        
        if label_smoothing > 0:
            logger.info(f"Using label smoothing: {label_smoothing}")
        
        # Initialize WandB if requested
        if wandb_project and HAS_WANDB:
            wandb.init(
                project=wandb_project,
                name=wandb_run_name,
                config={
                    **config.__dict__,
                    **(model_config or {}),
                    'use_focal_loss': use_focal_loss,
                    'label_smoothing': label_smoothing,
                    'class_weights': class_weights.tolist() if class_weights is not None else None
                }
            )
            self.wandb_enabled = True
            logger.info(f"WandB initialized: {wandb_project}/{wandb_run_name or 'unnamed'}")
        elif wandb_project and not HAS_WANDB:
            logger.warning("WandB requested but not installed. Run: pip install wandb")
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Set seed for reproducibility
        set_seed(config.seed)
        
        # Loss function with class imbalance handling
        if use_focal_loss:
            self.criterion = FocalLoss(alpha=0.25, gamma=2.0)
            logger.info("Using Focal Loss for class imbalance")
        elif class_weights is not None:
            # pos_weight for BCEWithLogitsLoss: weight for positive class (real=1)
            pos_weight = class_weights[1] / class_weights[0]
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(self.device))
            logger.info(f"Using weighted BCE loss with pos_weight={pos_weight:.2f}")
        else:
            self.criterion = nn.BCEWithLogitsLoss()
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Scheduler
        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            warmup_epochs=config.warmup_epochs,
            total_epochs=config.epochs
        )
        
        # Best metrics tracking
        self.best_acer = float('inf')
        
        # Create checkpoint directory
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """Train for one epoch.
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
        
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device).float().unsqueeze(1)
            
            # Keep original labels for metrics (before smoothing)
            original_labels = labels.clone()
            
            # Apply label smoothing if enabled (only for loss computation)
            if self.label_smoothing > 0:
                labels = labels * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
            
            # Forward pass with intermediate features
            logits, intermediate = self.model(images, return_intermediate=True)
            
            # Compute BCE loss
            bce_loss = self.criterion(logits, labels)
            
            # Compute intermediate supervision loss
            intermediate_loss = 0.0
            for layer_idx, features in intermediate.items():
                aux_logits = self.model.aux_heads[str(layer_idx)](features)
                intermediate_loss += self.criterion(aux_logits, labels)
            intermediate_loss /= len(intermediate)
            
            # Combined loss
            loss = (
                self.config.bce_weight * bce_loss +
                self.config.intermediate_weight * intermediate_loss
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Accumulate metrics (use original binary labels, not smoothed)
            total_loss += loss.item()
            preds = torch.sigmoid(logits).detach().cpu().numpy()
            all_preds.extend(preds.flatten())
            all_labels.extend(original_labels.cpu().numpy().flatten())
        
        # Update learning rate
        self.scheduler.step()
        
        # Compute metrics
        metrics = compute_metrics(np.array(all_preds), np.array(all_labels))
        
        return {
            'loss': total_loss / len(train_loader),
            'accuracy': metrics.accuracy,
            'auc': metrics.auc,
            'lr': self.scheduler.get_lr()
        }
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> EvaluationMetrics:
        """Validate model.
        
        Args:
            val_loader: Validation data loader
        
        Returns:
            Evaluation metrics
        """
        self.model.eval()
        all_preds = []
        all_labels = []
        
        for images, labels in val_loader:
            images = images.to(self.device)
            logits = self.model(images, return_intermediate=False)
            preds = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(preds.flatten())
            all_labels.extend(labels.numpy().flatten())
        
        return compute_metrics(np.array(all_preds), np.array(all_labels))
    
    def save_checkpoint(
        self,
        epoch: int,
        metrics: EvaluationMetrics,
        is_best: bool = False
    ) -> None:
        """Save model checkpoint.
        
        Args:
            epoch: Current epoch
            metrics: Evaluation metrics
            is_best: Whether this is the best model
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics.to_dict(),
            'config': self.model_config,  # Save MODEL config (not training config)
            'training_config': self.config.__dict__
        }
        
        # Save regular checkpoint
        path = Path(self.config.checkpoint_dir) / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")
        
        # Save best model
        if is_best:
            best_path = Path(self.config.checkpoint_dir) / 'best_model.pth'
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model to {best_path}")
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        early_stopping: Optional[int] = None
    ) -> None:
        """Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
            early_stopping: Stop if no improvement for N epochs (None to disable)
        """
        logger.info(f"Starting training for {self.config.epochs} epochs")
        logger.info(f"Device: {self.device}")
        if early_stopping:
            logger.info(f"Early stopping patience: {early_stopping} epochs")
        
        # Watch model with WandB
        if self.wandb_enabled:
            wandb.watch(self.model, log='all', log_freq=100)
        
        epochs_without_improvement = 0
        
        for epoch in range(1, self.config.epochs + 1):
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            
            log_msg = (
                f"Epoch {epoch}/{self.config.epochs} | "
                f"Loss: {train_metrics['loss']:.4f} | "
                f"Acc: {train_metrics['accuracy']:.4f} | "
                f"AUC: {train_metrics['auc']:.4f} | "
                f"LR: {train_metrics['lr']:.6f}"
            )
            
            # Prepare WandB log dict
            wandb_log = {
                'epoch': epoch,
                'train/loss': train_metrics['loss'],
                'train/accuracy': train_metrics['accuracy'],
                'train/auc': train_metrics['auc'],
                'learning_rate': train_metrics['lr']
            }
            
            # Validate
            is_best = False
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                log_msg += f" | Val ACER: {val_metrics.acer:.4f}"
                
                # Add validation metrics to WandB
                wandb_log.update({
                    'val/acer': val_metrics.acer,
                    'val/auc': val_metrics.auc,
                    'val/accuracy': val_metrics.accuracy,
                    'val/eer': val_metrics.eer,
                    'val/apcer': val_metrics.apcer,
                    'val/bpcer': val_metrics.bpcer
                })
                
                if val_metrics.acer < self.best_acer:
                    self.best_acer = val_metrics.acer
                    is_best = True
                    epochs_without_improvement = 0
                    wandb_log['best_acer'] = self.best_acer
                else:
                    epochs_without_improvement += 1
            
            logger.info(log_msg)
            
            # Log to WandB
            if self.wandb_enabled:
                wandb.log(wandb_log)
            
            # Save checkpoint
            if epoch % self.config.save_every == 0 or is_best:
                metrics = val_metrics if val_loader else EvaluationMetrics(
                    auc=train_metrics['auc'],
                    eer=0.0,
                    apcer=0.0,
                    bpcer=0.0,
                    acer=0.0,
                    accuracy=train_metrics['accuracy'],
                    threshold=0.5
                )
                self.save_checkpoint(epoch, metrics, is_best)
            
            # Early stopping check
            if early_stopping and epochs_without_improvement >= early_stopping:
                logger.info(f"Early stopping triggered after {epoch} epochs (no improvement for {early_stopping} epochs)")
                break
        
        # Finish WandB run
        if self.wandb_enabled:
            wandb.finish()
        
        logger.info("Training complete!")
