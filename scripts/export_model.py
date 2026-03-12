#!/usr/bin/env python3
"""
Export SpoofFormer model to ONNX and TorchScript formats.

Usage:
    python scripts/export_model.py --checkpoint checkpoints/best_model.pth
    python scripts/export_model.py --checkpoint checkpoints/best_model.pth --output-dir exports
    python scripts/export_model.py --checkpoint checkpoints/best_model.pth --format onnx
    python scripts/export_model.py --checkpoint checkpoints/best_model.pth --format torchscript
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from spoofformer.models import SpoofFormer
from spoofformer.export import export_onnx, export_torchscript


def detect_config_from_state_dict(state_dict: dict) -> dict:
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
        config['embed_dim'] = 384
    
    # Detect num_heads from embed_dim
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


def load_model_from_checkpoint(checkpoint_path: str) -> SpoofFormer:
    """Load model from checkpoint file."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    config = checkpoint.get('config', {})
    
    # If config is empty or missing key params, auto-detect from state dict
    if not config or 'embed_dim' not in config:
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        config = detect_config_from_state_dict(state_dict)
        print(f"Auto-detected model config: embed_dim={config['embed_dim']}, "
              f"heads={config['num_heads']}, layers={config['num_layers']}")
    
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
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model


def main():
    parser = argparse.ArgumentParser(description='Export SpoofFormer model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pth)')
    parser.add_argument('--output-dir', type=str, default='exports',
                        help='Output directory for exported models')
    parser.add_argument('--format', type=str, default='both',
                        choices=['onnx', 'torchscript', 'both'],
                        help='Export format')
    parser.add_argument('--img-size', type=int, default=224,
                        help='Input image size')
    parser.add_argument('--opset', type=int, default=14,
                        help='ONNX opset version')
    
    args = parser.parse_args()
    
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        return 1
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading model from {args.checkpoint}")
    model = load_model_from_checkpoint(args.checkpoint)
    print("Model loaded")
    
    if args.format in ['onnx', 'both']:
        onnx_path = output_dir / "model.onnx"
        print(f"\nExporting to ONNX: {onnx_path}")
        try:
            export_onnx(
                model=model,
                output_path=str(onnx_path),
                img_size=args.img_size,
                opset_version=args.opset
            )
            print(f"ONNX export complete: {onnx_path}")
        except Exception as e:
            print(f"ONNX export failed: {e}")
    
    if args.format in ['torchscript', 'both']:
        ts_path = output_dir / "model.pt"
        print(f"\nExporting to TorchScript: {ts_path}")
        try:
            export_torchscript(
                model=model,
                output_path=str(ts_path),
                img_size=args.img_size
            )
            print(f"TorchScript export complete: {ts_path}")
        except Exception as e:
            print(f"TorchScript export failed: {e}")
    
    print("\nExport complete!")
    print(f"Output directory: {output_dir.absolute()}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
