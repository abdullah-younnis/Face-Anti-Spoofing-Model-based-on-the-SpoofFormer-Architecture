"""
SpoofFormer training components.

This module contains:
- Trainer: Main training loop with intermediate supervision
- Loss functions for BCE and intermediate feature supervision
- Learning rate schedulers with warmup
- Evaluation metrics
"""

from spoofformer.training.trainer import Trainer, WarmupCosineScheduler, set_seed
from spoofformer.training.metrics import EvaluationMetrics, compute_metrics

__all__ = [
    "Trainer",
    "WarmupCosineScheduler",
    "set_seed",
    "EvaluationMetrics",
    "compute_metrics",
]
