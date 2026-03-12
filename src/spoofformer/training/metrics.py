"""
Evaluation metrics for face anti-spoofing.

Computes AUC, EER, APCER, BPCER, ACER and other metrics.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


@dataclass
class EvaluationMetrics:
    """Face anti-spoofing evaluation metrics.
    
    Attributes:
        auc: Area Under ROC Curve
        eer: Equal Error Rate
        apcer: Attack Presentation Classification Error Rate
        bpcer: Bona Fide Presentation Classification Error Rate
        acer: Average Classification Error Rate = (APCER + BPCER) / 2
        accuracy: Overall accuracy
        threshold: Optimal threshold used
    """
    auc: float
    eer: float
    apcer: float
    bpcer: float
    acer: float
    accuracy: float
    threshold: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return asdict(self)
    
    def __str__(self) -> str:
        return (
            f"AUC: {self.auc:.4f} | EER: {self.eer:.4f} | "
            f"ACER: {self.acer:.4f} | Acc: {self.accuracy:.4f}"
        )


def compute_eer(fpr: np.ndarray, tpr: np.ndarray) -> Tuple[float, float]:
    """Compute Equal Error Rate.
    
    Args:
        fpr: False positive rates
        tpr: True positive rates
    
    Returns:
        Tuple of (EER, threshold at EER)
    """
    fnr = 1 - tpr
    
    # Find the point where FPR = FNR
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2
    
    return float(eer), float(idx)


def compute_apcer_bpcer(
    predictions: np.ndarray,
    labels: np.ndarray,
    threshold: float
) -> Tuple[float, float]:
    """Compute APCER and BPCER.
    
    Args:
        predictions: Predicted scores (0=spoof, 1=real)
        labels: Ground truth labels (0=spoof, 1=real)
        threshold: Decision threshold
    
    Returns:
        Tuple of (APCER, BPCER)
    """
    # Binary predictions
    pred_labels = (predictions >= threshold).astype(int)
    
    # Spoof samples (label=0)
    spoof_mask = labels == 0
    # Real samples (label=1)
    real_mask = labels == 1
    
    # APCER: False acceptance of spoofs (spoof predicted as real)
    if spoof_mask.sum() > 0:
        apcer = (pred_labels[spoof_mask] == 1).sum() / spoof_mask.sum()
    else:
        apcer = 0.0
    
    # BPCER: False rejection of real (real predicted as spoof)
    if real_mask.sum() > 0:
        bpcer = (pred_labels[real_mask] == 0).sum() / real_mask.sum()
    else:
        bpcer = 0.0
    
    return float(apcer), float(bpcer)


def compute_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    threshold: Optional[float] = None
) -> EvaluationMetrics:
    """Compute all FAS evaluation metrics.
    
    Args:
        predictions: Predicted scores in [0, 1] (after sigmoid)
        labels: Ground truth labels {0, 1}
        threshold: Decision threshold (if None, optimal is computed)
    
    Returns:
        EvaluationMetrics with all computed metrics
    """
    predictions = np.asarray(predictions).flatten()
    labels = np.asarray(labels).flatten()
    
    # Compute AUC
    auc = roc_auc_score(labels, predictions)
    
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(labels, predictions)
    
    # Compute EER
    eer, eer_idx = compute_eer(fpr, tpr)
    
    # Use optimal threshold if not provided
    if threshold is None:
        # Use threshold at EER
        threshold = float(thresholds[int(eer_idx)]) if int(eer_idx) < len(thresholds) else 0.5
    
    # Compute APCER and BPCER
    apcer, bpcer = compute_apcer_bpcer(predictions, labels, threshold)
    
    # Compute ACER
    acer = (apcer + bpcer) / 2
    
    # Compute accuracy
    pred_labels = (predictions >= threshold).astype(int)
    accuracy = (pred_labels == labels).mean()
    
    return EvaluationMetrics(
        auc=float(auc),
        eer=float(eer),
        apcer=float(apcer),
        bpcer=float(bpcer),
        acer=float(acer),
        accuracy=float(accuracy),
        threshold=float(threshold)
    )
