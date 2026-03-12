"""
Inference engine for SpoofFormer.

Supports PyTorch, ONNX, and TorchScript backends.
"""

import logging
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch

from spoofformer.inference.result import LivenessResult
from spoofformer.inference.preprocessing import preprocess_image, detect_and_align_face

logger = logging.getLogger(__name__)


class InferenceEngine:
    """Production inference engine for face anti-spoofing.
    
    Supports multiple backends:
    - pytorch: Standard PyTorch model
    - onnx: ONNX Runtime
    - torchscript: TorchScript traced model
    
    Args:
        model_path: Path to model file
        backend: Inference backend ("pytorch", "onnx", "torchscript")
        device: Device to run inference on
        threshold: Decision threshold for real/spoof
        img_size: Input image size
    """
    
    def __init__(
        self,
        model_path: str,
        backend: str = "pytorch",
        device: str = "cuda",
        threshold: float = 0.5,
        img_size: int = 224
    ) -> None:
        self.model_path = Path(model_path)
        self.backend = backend
        self.device = device if torch.cuda.is_available() else "cpu"
        self.threshold = threshold
        self.img_size = img_size
        
        self.model = None
        self.onnx_session = None
        
        self._load_model()
    
    def _load_model(self) -> None:
        """Load model based on backend."""
        if self.backend == "pytorch":
            self._load_pytorch()
        elif self.backend == "onnx":
            self._load_onnx()
        elif self.backend == "torchscript":
            self._load_torchscript()
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
    
    def _load_pytorch(self) -> None:
        """Load PyTorch model from checkpoint."""
        from spoofformer.models import SpoofFormer
        
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        
        # Get config from checkpoint or auto-detect from state dict
        config = checkpoint.get('config', {})
        
        # If config is empty or missing key params, try to detect from state dict
        if not config or 'embed_dim' not in config:
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            config = self._detect_config_from_state_dict(state_dict)
            logger.info(f"Auto-detected model config: embed_dim={config['embed_dim']}, layers={config['num_layers']}")
        
        # Create model with saved config
        model = SpoofFormer(
            img_size=config.get('img_size', 224),
            patch_size=config.get('patch_size', 16),
            in_channels=config.get('in_channels', 3),
            embed_dim=config.get('embed_dim', 384),
            num_heads=config.get('num_heads', 6),
            num_layers=config.get('num_layers', 12),
            mlp_ratio=config.get('mlp_ratio', 4.0),
            dropout=config.get('dropout', 0.1),
            extract_layers=config.get('extract_layers', [8, 11]),
        )
        
        # Load weights
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(self.device)
        model.eval()
        self.model = model
        logger.info(f"Loaded PyTorch model from {self.model_path}")
    
    def _detect_config_from_state_dict(self, state_dict: dict) -> dict:
        """Auto-detect model configuration from state dict weights."""
        config = {
            'img_size': 224,
            'patch_size': 16,
            'in_channels': 3,
            'mlp_ratio': 4.0,
            'dropout': 0.1,
        }
        
        # Detect embed_dim from cls_token shape
        if 'cls_token' in state_dict:
            config['embed_dim'] = state_dict['cls_token'].shape[-1]
        elif 'patch_embed.proj.weight' in state_dict:
            config['embed_dim'] = state_dict['patch_embed.proj.weight'].shape[0]
        else:
            config['embed_dim'] = 384  # Default to small
        
        # Detect num_heads from embed_dim (standard ratios)
        embed_dim = config['embed_dim']
        if embed_dim == 192:
            config['num_heads'] = 3
        elif embed_dim == 384:
            config['num_heads'] = 6
        elif embed_dim == 768:
            config['num_heads'] = 12
        elif embed_dim == 1024:
            config['num_heads'] = 16
        else:
            config['num_heads'] = embed_dim // 64
        
        # Detect num_layers by counting encoder blocks
        num_layers = 0
        for key in state_dict.keys():
            if key.startswith('encoder.blocks.'):
                layer_idx = int(key.split('.')[2])
                num_layers = max(num_layers, layer_idx + 1)
        config['num_layers'] = num_layers if num_layers > 0 else 12
        
        # Detect extract_layers from aux_heads
        extract_layers = []
        for key in state_dict.keys():
            if key.startswith('aux_heads.') and key.endswith('.weight'):
                layer_idx = int(key.split('.')[1])
                extract_layers.append(layer_idx)
        config['extract_layers'] = sorted(extract_layers) if extract_layers else [8, 11]
        
        return config
    
    def _load_onnx(self) -> None:
        """Load ONNX model."""
        import onnxruntime as ort
        
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        if self.device == 'cpu':
            providers = ['CPUExecutionProvider']
        
        self.onnx_session = ort.InferenceSession(
            str(self.model_path),
            providers=providers
        )
        logger.info(f"Loaded ONNX model from {self.model_path}")
    
    def _load_torchscript(self) -> None:
        """Load TorchScript model."""
        self.model = torch.jit.load(str(self.model_path), map_location=self.device)
        self.model.eval()
        logger.info(f"Loaded TorchScript model from {self.model_path}")
    
    def predict(
        self,
        image: np.ndarray,
        face_detector: Optional[object] = None
    ) -> LivenessResult:
        """Run inference on a single image.
        
        Args:
            image: Input image [H, W, C] with values in [0, 255]
            face_detector: Optional face detector
        
        Returns:
            LivenessResult with score, prediction, and confidence
        """
        # Face detection
        if face_detector is not None:
            face = detect_and_align_face(image, face_detector)
            if face is None:
                return LivenessResult(
                    liveness_score=0.0,
                    prediction="unknown",
                    confidence=0.0
                )
            image = face
        
        # Preprocess
        tensor = preprocess_image(image, self.img_size)
        
        # Inference
        if self.backend == "onnx":
            score = self._infer_onnx(tensor)
        else:
            score = self._infer_pytorch(tensor)
        
        # Calculate distance from decision boundary
        # Distance is positive for real (score > threshold), negative for spoof
        distance = score - self.threshold
        
        # Confidence = how far from decision boundary, scaled to [0, 1]
        # Max confidence = 1.0 (at score 0 or 1), min = 0.0 (at threshold)
        confidence = abs(score - self.threshold) / max(self.threshold, 1 - self.threshold)
        confidence = min(confidence, 1.0)  # Clamp to [0, 1]
        
        # Decision
        if score >= self.threshold:
            prediction = "real"
        else:
            prediction = "spoof"
        
        return LivenessResult(
            liveness_score=score,
            prediction=prediction,
            confidence=confidence,
            distance=distance
        )
    
    def predict_batch(
        self,
        images: List[np.ndarray],
        face_detector: Optional[object] = None
    ) -> List[LivenessResult]:
        """Run inference on multiple images.
        
        Args:
            images: List of input images
            face_detector: Optional face detector
        
        Returns:
            List of LivenessResult
        """
        return [self.predict(img, face_detector) for img in images]
    
    def _infer_pytorch(self, tensor: torch.Tensor) -> float:
        """Run PyTorch inference."""
        tensor = tensor.to(self.device)
        with torch.no_grad():
            # TorchScript models don't support keyword arguments
            if self.backend == "torchscript":
                logits = self.model(tensor)
            else:
                logits = self.model(tensor, return_intermediate=False)
            score = torch.sigmoid(logits).item()
        return score
    
    def _infer_onnx(self, tensor: torch.Tensor) -> float:
        """Run ONNX inference."""
        input_name = self.onnx_session.get_inputs()[0].name
        output_name = self.onnx_session.get_outputs()[0].name
        
        logits = self.onnx_session.run(
            [output_name],
            {input_name: tensor.numpy()}
        )[0]
        
        score = 1 / (1 + np.exp(-logits[0, 0]))  # Sigmoid
        return float(score)
    
    def warmup(self, num_iterations: int = 10) -> None:
        """Warmup model for consistent latency.
        
        Args:
            num_iterations: Number of warmup iterations
        """
        dummy_input = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        for _ in range(num_iterations):
            self.predict(dummy_input)
        
        logger.info(f"Completed {num_iterations} warmup iterations")
