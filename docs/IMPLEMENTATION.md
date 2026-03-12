# SpoofFormer Implementation Details

## Overview

This document provides a comprehensive technical description of the SpoofFormer face anti-spoofing system, including architectural decisions, implementation details, and the rationale behind design choices.

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Architecture](#architecture)
3. [Model Components](#model-components)
4. [Training Pipeline](#training-pipeline)
5. [Anti-Overfitting Techniques](#anti-overfitting-techniques)
6. [Class Imbalance Handling](#class-imbalance-handling)
7. [Loss Functions](#loss-functions)
8. [Evaluation Metrics](#evaluation-metrics)
9. [Inference System](#inference-system)
10. [Model Configuration System](#model-configuration-system)
11. [Checkpoint System](#checkpoint-system)
12. [WandB Integration](#wandb-integration)
13. [Design Decisions](#design-decisions)
14. [Training Results](#training-results)

---

## Problem Statement

Face anti-spoofing (FAS) is a binary classification task that distinguishes between:
- **Real (live) faces**: Genuine human faces captured directly by a camera
- **Spoof (fake) faces**: Presentation attacks including printed photos, screen replays, and 3D masks

The challenge lies in detecting subtle artifacts that differentiate spoofs from real faces, such as:
- Moire patterns from screen displays
- Paper texture and reflection artifacts
- Lack of 3D depth information
- Unnatural color distributions

## Architecture

### Why Vision Transformer (ViT)?

We chose a Vision Transformer architecture over traditional CNNs for several reasons:

1. **Global Context**: Transformers capture long-range dependencies through self-attention, enabling the model to correlate distant image regions (e.g., detecting inconsistent lighting across the face).

2. **Patch-based Processing**: ViT processes images as sequences of patches, which aligns well with detecting localized spoofing artifacts that may appear in specific regions.

3. **Intermediate Feature Extraction**: Transformer layers produce hierarchical representations that can be leveraged for multi-scale supervision.

4. **Transfer Learning**: Pre-trained ViT models on ImageNet provide strong initialization for downstream tasks.

### Model Configurations

We provide multiple model sizes via YAML configuration (`configs/model_configs.yaml`):

| Configuration | Embed Dim | Heads | Layers | Parameters | Use Case |
|--------------|-----------|-------|--------|------------|----------|
| ViT-Tiny     | 192       | 3     | 12     | ~5M        | Small datasets, fast training |
| ViT-Small    | 384       | 6     | 12     | ~22M       | Production, balanced |
| ViT-Base     | 768       | 12    | 12     | ~86M       | High accuracy |
| ViT-Large    | 1024      | 16    | 24     | ~307M      | Research, large datasets |
| Mobile       | 192       | 3     | 6      | ~2.5M      | Edge deployment |
| Small-HR     | 384       | 6     | 12     | ~22M       | High-resolution (384x384) |

**Rationale**: 
- ViT-Tiny is recommended for small datasets (like NUAA with ~10k images) to prevent overfitting
- ViT-Small offers a good balance between accuracy and inference speed for production
- Mobile variant is optimized for edge deployment with minimal layers
- ViT-Large would severely overfit on small datasets

## Model Components

### 1. Patch Embedding

```
Input Image (224x224x3) -> Patches (14x14 grid) -> Linear Projection -> Patch Embeddings
```

**Implementation Details**:
- Patch size: 16x16 pixels
- Number of patches: (224/16)^2 = 196
- Projection: Conv2d with kernel_size=patch_size, stride=patch_size
- Output: 196 patch embeddings + 1 CLS token = 197 tokens

### 2. Positional Encoding

We use learnable positional embeddings:

```python
self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
```

**Rationale**: Learnable embeddings allow the model to adapt positional information to the specific task of face anti-spoofing.

### 3. Transformer Encoder

Each transformer block consists of:

```
Input -> LayerNorm -> Multi-Head Attention -> Residual -> LayerNorm -> MLP -> Residual -> Output
```

**Multi-Head Self-Attention**:
- Head dimension = embed_dim / num_heads
- Attention dropout for regularization

**MLP Block**:
- Two-layer feedforward network with GELU activation
- Hidden dimension = 4x embed_dim (MLP ratio)
- Dropout for regularization

### 4. Classification Head

```python
Sequential(
    Linear(embed_dim, 256),
    GELU(),
    Dropout(dropout_rate),
    Linear(256, 1)  # Binary classification
)
```

### 5. Auxiliary Heads for Intermediate Supervision

```python
aux_heads = {
    layer_idx: Linear(embed_dim, 1)
    for layer_idx in extract_layers  # e.g., [8, 11]
}
```

**Purpose**: Intermediate supervision provides gradient flow to earlier layers and multi-scale feature learning.

## Training Pipeline

### Data Augmentation

Three augmentation strength levels are available via `--augment` flag:

**Light** (minimal augmentation):
```python
- Resize(224, 224)
- HorizontalFlip(p=0.5)
- Normalize(ImageNet mean/std)
```

**Normal** (default):
```python
- Resize(224, 224)
- HorizontalFlip(p=0.5)
- RandomRotation(degrees=15)
- ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
- Normalize(ImageNet mean/std)
```

**Strong** (recommended for small datasets):
```python
- Resize(246, 246) + RandomCrop(224, 224)
- HorizontalFlip(p=0.5)
- RandomRotation(degrees=30)
- ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
- RandomGrayscale(p=0.1)
- GaussianBlur(kernel_size=5)
- RandomPerspective(distortion_scale=0.2, p=0.3)
- RandomErasing(p=0.25)
- Normalize(ImageNet mean/std)
```

### Optimizer and Scheduler

**AdamW Optimizer**:
```python
optimizer = AdamW(params, lr=1e-4, weight_decay=0.05)
```

**Warmup + Cosine Decay Schedule**:
```
LR = base_lr x warmup_factor  (epochs 1-5)
LR = base_lr x 0.5 x (1 + cos(pi x progress))  (epochs 6+)
```

### Gradient Clipping

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

## Anti-Overfitting Techniques

Small datasets like NUAA (~10k images) are prone to overfitting. We implement several techniques:

### 1. Strong Data Augmentation (`--augment strong`)

Forces the model to learn actual spoofing features rather than dataset-specific patterns:
- Color jitter simulates different lighting conditions
- Gaussian blur simulates camera blur
- Random perspective simulates different viewing angles
- Random erasing (cutout) prevents reliance on specific regions

### 2. Label Smoothing (`--label_smoothing 0.1`)

Prevents overconfident predictions by softening labels:
```python
# Instead of hard labels (0 or 1):
smoothed_label = label * (1 - smoothing) + 0.5 * smoothing
# With smoothing=0.1: 0 -> 0.05, 1 -> 0.95
```

**Implementation Note**: Original binary labels are preserved for metric computation (AUC calculation requires binary labels), while smoothed labels are used only for loss computation.

**Benefits**:
- Reduces overconfidence
- Improves calibration
- Acts as regularization

### 3. Higher Dropout (`--dropout 0.3`)

Override model's default dropout for stronger regularization:
```python
model_config.dropout = 0.3  # Default is 0.1
```

### 4. Smaller Model (`--model_version tiny`)

Fewer parameters = less capacity to memorize:
- ViT-Tiny: ~5M parameters
- ViT-Small: ~22M parameters

For NUAA dataset, ViT-Tiny is recommended.

### 5. Early Stopping (`--early_stopping 20`)

Stop training when validation ACER stops improving:
```python
if epochs_without_improvement >= patience:
    stop_training()
```

## Class Imbalance Handling

The NUAA dataset has class imbalance (4084 real vs 6007 spoof, ~1.47:1 ratio).

### 1. Weighted Random Sampling (`--balance_classes`)

Oversamples minority class (real) so each batch has balanced classes:
```python
weight_real = total / (2 * num_real)
weight_spoof = total / (2 * num_spoof)
sampler = WeightedRandomSampler(weights, num_samples, replacement=True)
```

### 2. Focal Loss (`--focal_loss`)

Down-weights easy examples, focusing on hard cases:
```python
FL(p) = -alpha * (1-p)^gamma * log(p)
# gamma=2.0 makes easy examples contribute less
# alpha=0.25 balances positive/negative
```

### 3. Class-Weighted BCE (`--class_weight`)

Applies higher weight to minority class in loss:
```python
pos_weight = weight_real / weight_spoof
loss = BCEWithLogitsLoss(pos_weight=pos_weight)
```

### Recommended Combination

For best results on imbalanced small datasets:
```bash
python train.py --balance_classes --focal_loss --augment strong --label_smoothing 0.1
```

## Loss Functions

### Primary Loss: Binary Cross-Entropy with Logits

```python
loss_bce = BCEWithLogitsLoss()(logits, labels)
```

### Intermediate Supervision Loss

```python
loss_intermediate = mean([
    BCEWithLogitsLoss()(aux_head(features[layer]), labels)
    for layer in extract_layers
])
```

### Combined Loss

```python
total_loss = bce_weight * loss_bce + intermediate_weight * loss_intermediate
```

Default weights: `bce_weight=1.0`, `intermediate_weight=0.5`

## Evaluation Metrics

All metrics are computed during training and logged to WandB.

### Standard Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| AUC | Area under ROC curve | Overall discrimination ability |
| EER | FPR = FNR point | Equal error rate |
| Accuracy | (TP + TN) / Total | Overall correctness |

### Face Anti-Spoofing Specific Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| APCER | FP / (FP + TN) | Attack Presentation Classification Error Rate |
| BPCER | FN / (FN + TP) | Bona Fide Presentation Classification Error Rate |
| ACER | (APCER + BPCER) / 2 | Average Classification Error Rate |

**ACER** is the primary metric for model selection and early stopping.

### Metrics Implementation

```python
# metrics.py computes all metrics
@dataclass
class EvaluationMetrics:
    auc: float      # Area Under ROC Curve
    eer: float      # Equal Error Rate
    apcer: float    # Attack Presentation Classification Error Rate
    bpcer: float    # Bona Fide Presentation Classification Error Rate
    acer: float     # Average Classification Error Rate
    accuracy: float # Overall accuracy
    threshold: float # Optimal threshold used
```

## Inference System

### Supported Backends

1. **PyTorch**: Full model with all features
2. **ONNX**: Cross-platform deployment, optimized inference
3. **TorchScript**: PyTorch-native deployment without Python dependency

### Model Configuration Auto-Detection

The inference engine automatically detects model configuration from checkpoints:

```python
def _detect_config_from_state_dict(state_dict):
    """Auto-detect model config from checkpoint weights."""
    # Detect embed_dim from patch embedding projection
    embed_dim = state_dict['patch_embed.proj.weight'].shape[0]
    
    # Detect num_heads from attention weights
    num_heads = state_dict['encoder.layers.0.attn.qkv.weight'].shape[0] // (3 * head_dim)
    
    # Detect num_layers by counting encoder layers
    num_layers = max(int(k.split('.')[2]) for k in state_dict if 'encoder.layers' in k) + 1
    
    # Detect extract_layers from aux_heads
    extract_layers = [int(k.split('.')[1]) for k in state_dict if 'aux_heads' in k and 'weight' in k]
```

This ensures backward compatibility with checkpoints that don't have embedded config.

### Preprocessing Pipeline

```python
def preprocess(image):
    image = cv2.resize(image, (224, 224))
    image = image.astype(np.float32) / 255.0
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image = (image - mean) / std
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    return tensor
```

### Output Interpretation

The inference system provides four output fields:

```python
@dataclass
class LivenessResult:
    liveness_score: float  # P(real) in [0, 1]
    prediction: str        # "real" or "spoof"
    confidence: float      # Distance from boundary, scaled to [0, 1]
    distance: float        # Raw distance from threshold
```

**Confidence Calculation** (Distance from Decision Boundary):

```python
# Distance is positive for real (score > threshold), negative for spoof
distance = score - threshold

# Confidence = how far from decision boundary, scaled to [0, 1]
# Max confidence = 1.0 (at score 0 or 1), min = 0.0 (at threshold)
confidence = abs(score - threshold) / max(threshold, 1 - threshold)
confidence = min(confidence, 1.0)  # Clamp to [0, 1]
```

**Example outputs with threshold 0.5**:

| Liveness Score | Prediction | Confidence | Distance | Meaning |
|----------------|------------|------------|----------|---------|
| 0.92 | real | 0.84 | +0.42 | Very confident real |
| 0.51 | real | 0.02 | +0.01 | Uncertain, barely real |
| 0.10 | spoof | 0.80 | -0.40 | Very confident spoof |
| 0.49 | spoof | 0.02 | -0.01 | Uncertain, barely spoof |

## Model Configuration System

### YAML-Based Configuration

Model architectures are defined in `configs/model_configs.yaml`:

```yaml
tiny:
  embed_dim: 192
  num_heads: 3
  num_layers: 12
  mlp_ratio: 4.0
  dropout: 0.1
  extract_layers: [8, 11]
  img_size: 224
  patch_size: 16
  in_channels: 3
```

### Usage

```bash
# List available versions
python train.py --list_versions

# Use specific version
python train.py --model_version tiny

# Override specific parameters
python train.py --model_version tiny --dropout 0.3
```

## Checkpoint System

### Checkpoint Contents

Checkpoints save both model weights and configuration:

```python
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'metrics': metrics.to_dict(),
    'config': model_config,           # Model architecture config
    'training_config': training_config # Training hyperparameters
}
```

### Backward Compatibility

For checkpoints without embedded config, the system auto-detects configuration from weight shapes (see Model Configuration Auto-Detection above).

## WandB Integration

### Logged Metrics

**Training metrics** (per epoch):
- `train/loss`: Combined BCE + intermediate supervision loss
- `train/accuracy`: Training accuracy
- `train/auc`: Training AUC
- `learning_rate`: Current learning rate

**Validation metrics** (per epoch):
- `val/acer`: Average Classification Error Rate
- `val/auc`: Area Under ROC Curve
- `val/accuracy`: Validation accuracy
- `val/eer`: Equal Error Rate
- `val/apcer`: Attack Presentation Classification Error Rate
- `val/bpcer`: Bona Fide Presentation Classification Error Rate
- `best_acer`: Best ACER achieved so far

### Configuration Logging

WandB also logs:
- Model configuration (embed_dim, num_heads, etc.)
- Training configuration (epochs, batch_size, lr, etc.)
- Anti-overfitting settings (label_smoothing, dropout, etc.)
- Class imbalance settings (focal_loss, class_weights, etc.)

### Usage

```bash
pip install wandb
wandb login

python train.py \
    --data_root dataset \
    --wandb_project spoofformer \
    --wandb_run experiment-1
```

## Design Decisions

### Why 12 Transformer Layers?

- Matches standard ViT configurations
- Provides sufficient depth for hierarchical feature learning
- Allows intermediate supervision at layers 8 and 11

### Why Extract Layers [8, 11]?

- Layer 8: Captures mid-level features (texture, local patterns)
- Layer 11: Captures high-level features (semantic understanding)
- Provides complementary supervision signals

### Why 224x224 Input Size?

- Standard ImageNet size, enabling transfer learning
- Sufficient resolution for face anti-spoofing
- Balanced between detail preservation and computational cost

### Why Patch Size 16?

- Standard ViT configuration
- 14x14 = 196 patches provides good spatial granularity
- Smaller patches increase computation quadratically

### Why Binary Classification?

- Simpler problem formulation
- Real-world deployment typically only needs real/spoof distinction
- Multi-class can be added as future extension

### Why Distance-Based Confidence?

The confidence score uses distance from decision boundary rather than raw probability because:
- Semantic clarity: "Confidence" means certainty, not class probability
- Symmetric interpretation: Both real and spoof predictions have comparable confidence scales
- Practical utility: Distance from boundary better indicates prediction reliability

## Training Results

Trained on NUAA dataset (~10k images) with ViT-Tiny and anti-overfitting settings:

| Metric | Value |
|--------|-------|
| Accuracy | 98.97% |
| AUC | 0.9994 |
| Val ACER | 0.0005 |
| Epochs | 42 (early stopped at patience 20) |

**Training Command**:
```bash
python train.py \
    --data_root dataset \
    --epochs 100 \
    --model_version tiny \
    --augment strong \
    --label_smoothing 0.1 \
    --dropout 0.3 \
    --balance_classes \
    --focal_loss \
    --early_stopping 20
```

## Recommended Training Configuration

For small datasets like NUAA:

```bash
python train.py \
    --data_root dataset \
    --epochs 100 \
    --model_version tiny \
    --augment strong \
    --label_smoothing 0.1 \
    --dropout 0.3 \
    --balance_classes \
    --focal_loss \
    --early_stopping 20
```

This configuration:
- Uses smallest model to prevent overfitting
- Applies strongest augmentation
- Adds label smoothing for calibration
- Increases dropout for regularization
- Balances classes with weighted sampling
- Uses focal loss for hard example mining
- Stops early when validation plateaus

## Future Improvements

1. **Multi-scale Input**: Process images at multiple resolutions
2. **Attention Visualization**: Add interpretability tools
3. **Domain Adaptation**: Improve cross-dataset generalization
4. **Lightweight Variants**: Distillation for mobile deployment
5. **Multi-modal Fusion**: Combine RGB with depth or IR when available
6. **Test-Time Augmentation**: Average predictions over augmented versions

## References

1. Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" (ViT)
2. Wang et al., "Deep Learning for Face Anti-Spoofing: A Survey"
3. ISO/IEC 30107-3: Biometric presentation attack detection
4. Lin et al., "Focal Loss for Dense Object Detection"
