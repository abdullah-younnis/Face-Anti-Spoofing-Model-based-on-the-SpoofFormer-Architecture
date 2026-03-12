"""
Liveness result dataclass for inference output.
"""

from dataclasses import dataclass


@dataclass
class LivenessResult:
    """Structured output for liveness prediction.
    
    Attributes:
        liveness_score: Probability score from 0.0 (spoof) to 1.0 (real)
        prediction: "real", "spoof", or "unknown"
        confidence: Confidence of the prediction
        distance: Distance from decision boundary (positive = real, negative = spoof)
    """
    liveness_score: float
    prediction: str
    confidence: float
    distance: float = 0.0
    
    def __str__(self) -> str:
        return (
            f"LivenessResult(score={self.liveness_score:.3f}, "
            f"prediction='{self.prediction}', confidence={self.confidence:.3f}, "
            f"distance={self.distance:.3f})"
        )
