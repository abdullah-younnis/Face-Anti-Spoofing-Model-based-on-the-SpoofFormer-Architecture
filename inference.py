#!/usr/bin/env python3
"""
Inference script for SpoofFormer.

Usage:
    python inference.py --model checkpoints/best_model.pth --image test.jpg
    python inference.py --model exports/model.onnx --backend onnx --image test.jpg
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from spoofformer.inference import InferenceEngine

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='SpoofFormer Inference')
    
    parser.add_argument('--model', type=str, required=True,
                        help='Path to model file')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--backend', type=str, default='pytorch',
                        choices=['pytorch', 'onnx', 'torchscript'],
                        help='Inference backend')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Decision threshold')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file (optional)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Check model exists
    if not Path(args.model).exists():
        logger.error(f"Model not found: {args.model}")
        return 1
    
    # Check image exists
    if not Path(args.image).exists():
        logger.error(f"Image not found: {args.image}")
        return 1
    
    # Load image
    image = cv2.imread(args.image)
    if image is None:
        logger.error(f"Failed to load image: {args.image}")
        return 1
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create inference engine
    logger.info(f"Loading model from {args.model} (backend: {args.backend})")
    engine = InferenceEngine(
        model_path=args.model,
        backend=args.backend,
        device=args.device,
        threshold=args.threshold
    )
    
    # Run inference
    result = engine.predict(image)
    
    # Output result
    output = {
        'liveness_score': result.liveness_score,
        'prediction': result.prediction,
        'confidence': result.confidence,
        'distance': result.distance
    }
    
    print(f"\nResult:")
    print(f"  Liveness Score: {result.liveness_score:.4f}")
    print(f"  Prediction: {result.prediction}")
    print(f"  Confidence: {result.confidence:.4f}")
    print(f"  Distance from Boundary: {result.distance:.4f}")
    
    # Save to file if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(output, f, indent=2)
        logger.info(f"Saved result to {args.output}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
