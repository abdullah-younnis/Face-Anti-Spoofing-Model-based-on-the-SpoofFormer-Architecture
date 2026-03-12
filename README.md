# SpoofFormer: Vision Transformer for Face Anti-Spoofing

A PyTorch implementation of a Vision Transformer (ViT) based face anti-spoofing model with intermediate supervision.

![Model Architecture](docs/Model-Arch.jpg)

## Quick Start

```bash
# 1. Install
git clone https://github.com/abdullah-younnis/Face-Anti-Spoofing-Model-based-on-the-SpoofFormer-Architecture.git
cd spoofformer
pip install -r requirements.txt

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# 3. Download dataset
pip install kagglehub
python scripts/download_dataset.py

# 4. Train
python train.py --data_root dataset --model_version tiny --augment strong --epochs 100

# 5. Inference
python inference.py --model checkpoints/best_model.pth --image path/to/face.jpg
```

## Features

- Vision Transformer architecture (ViT-Tiny/Small/Base/Large/Mobile)
- Intermediate supervision for improved training
- Multiple inference backends (PyTorch, ONNX, TorchScript)
- WandB integration for experiment tracking
- Anti-overfitting techniques (strong augmentation, label smoothing, dropout)
- Class imbalance handling (weighted sampling, focal loss)
- YAML-based model configuration

## Training Results

Trained on NUAA dataset (~10k images) with ViT-Tiny and anti-overfitting settings:

| Metric | Value |
|--------|-------|
| Accuracy | 98.97% |
| AUC | 0.9994 |
| Val ACER | 0.0005 |
| Epochs | 42 (early stopped) |

## Model Configurations

| Config | Embed Dim | Heads | Layers | Parameters | Use Case |
|--------|-----------|-------|--------|------------|----------|
| Tiny | 192 | 3 | 12 | ~5M | Small datasets |
| Small | 384 | 6 | 12 | ~22M | Production |
| Base | 768 | 12 | 12 | ~86M | High accuracy |
| Large | 1024 | 16 | 24 | ~307M | Large datasets |
| Mobile | 192 | 3 | 6 | ~2.5M | Edge deployment |

## Installation

```bash
# Clone and setup
git clone https://github.com/yourusername/spoofformer.git
cd spoofformer

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

## Dataset

### Download NUAA Dataset

The download script automatically downloads, splits (80/20 train/val), resizes to 224x224, and organizes the dataset into the required structure.

```bash
pip install kagglehub
python scripts/download_dataset.py

# Limit images
python scripts/download_dataset.py --max-images 5000

# Synthetic data for testing
python scripts/download_dataset.py --synthetic --synthetic-count 100
```

### Expected Structure

```
dataset/
    train/
        real/
        spoof/
    val/
        real/
        spoof/
```

## Training

### Recommended Training (Small Datasets)

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

### Basic Training

```bash
python train.py --data_root dataset --epochs 100
```

### List Model Versions

```bash
python train.py --list_versions
```

### With WandB

```bash
pip install wandb
wandb login

python train.py \
    --data_root dataset \
    --epochs 100 \
    --wandb_project spoofformer \
    --wandb_run experiment-1
```

### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| --data_root | dataset | Dataset directory |
| --epochs | 100 | Training epochs |
| --batch_size | 32 | Batch size |
| --lr | 1e-4 | Learning rate |
| --model_version | small | Model version (tiny/small/base/large/mobile) |
| --augment | normal | Augmentation (light/normal/strong) |
| --label_smoothing | 0.0 | Label smoothing (0.1-0.2 recommended) |
| --dropout | None | Override dropout |
| --balance_classes | False | Weighted sampling |
| --focal_loss | False | Use focal loss |
| --early_stopping | 10 | Early stopping patience |

## Inference

Model configuration is auto-detected from checkpoint.

```bash
# PyTorch
python inference.py \
    --model checkpoints/best_model.pth \
    --image path/to/image.jpg

# ONNX
python inference.py \
    --model exports/model.onnx \
    --image path/to/image.jpg \
    --backend onnx

# TorchScript
python inference.py \
    --model exports/model.pt \
    --image path/to/image.jpg \
    --backend torchscript
```

### Output

```
Result:
  Liveness Score: 0.9234
  Prediction: real
  Confidence: 0.8468
  Distance from Boundary: 0.4234
```

| Field | Description |
|-------|-------------|
| Liveness Score | Probability of being real (0.0 = spoof, 1.0 = real) |
| Prediction | "real" or "spoof" based on threshold |
| Confidence | How far from decision boundary, scaled to [0, 1] |
| Distance | Raw distance from threshold (positive = real, negative = spoof) |

Examples with threshold 0.5:

| Score | Prediction | Confidence | Meaning |
|-------|------------|------------|---------|
| 0.92 | real | 0.84 | Very confident real |
| 0.51 | real | 0.02 | Uncertain, barely real |
| 0.10 | spoof | 0.80 | Very confident spoof |

## Model Export

Model configuration is auto-detected from checkpoint.

```bash
# Export both ONNX and TorchScript
python scripts/export_model.py --checkpoint checkpoints/best_model.pth

# ONNX only
python scripts/export_model.py --checkpoint checkpoints/best_model.pth --format onnx

# TorchScript only
python scripts/export_model.py --checkpoint checkpoints/best_model.pth --format torchscript
```

## Docker

### Build

```bash
# CPU
docker build -t spoofformer:latest --target training .

# GPU
docker build -t spoofformer:gpu -f Dockerfile.gpu .

# Inference only
docker build -t spoofformer:inference --target inference .
```

### Run

```bash
# Training (GPU)
docker run --gpus all \
    -v $(pwd)/dataset:/app/dataset:ro \
    -v $(pwd)/checkpoints:/app/checkpoints \
    spoofformer:gpu \
    python train.py --data_root dataset --model_version tiny --augment strong

# Inference
docker run \
    -v $(pwd)/checkpoints:/app/checkpoints:ro \
    -v $(pwd)/input:/app/input:ro \
    spoofformer:inference \
    python inference.py --model checkpoints/best_model.pth --image input/test.jpg
```

### Docker Compose

```bash
docker-compose run train-gpu   # GPU training
docker-compose run inference   # Inference
docker-compose run export      # Export model
```

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| AUC | Area under ROC curve |
| EER | Equal Error Rate |
| ACER | Average Classification Error Rate |
| APCER | Attack Presentation Classification Error Rate |
| BPCER | Bona Fide Presentation Classification Error Rate |

## Project Structure

```
spoofformer/
    configs/model_configs.yaml
    src/spoofformer/
        models/
        data/
        training/
        inference/
        export/
    scripts/
        download_dataset.py
        export_model.py
    train.py
    inference.py
    tests/
    docs/IMPLEMENTATION.md
```

## Testing

```bash
pytest tests/ -v
```

## License

MIT License

## References

- [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)
- [Deep Learning for Face Anti-Spoofing: A Survey](https://arxiv.org/abs/2106.14948)
- [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)

