#!/usr/bin/env python3
"""
NUAA Face Anti-Spoofing Dataset Downloader/Organizer

Downloads and organizes the NUAA dataset into train/val splits.
Dataset: aleksandrpikul222/nuaaaa (Kaggle)

Structure:
dataset/
    train/
        real/    (ClientRaw - live faces)
        spoof/   (ImposterRaw - fake/printed faces)
    val/
        real/
        spoof/

Train: 1743 real / 1748 fake
Test:  3362 real / 5761 fake

Usage:
    python scripts/download_dataset.py
    python scripts/download_dataset.py --max-images 1000
    python scripts/download_dataset.py --synthetic
"""

import os
import sys
import shutil
import argparse
import random
import json
from pathlib import Path
from typing import Tuple, List, Dict, Optional

import numpy as np
from tqdm import tqdm
from PIL import Image


class NUAADatasetOrganizer:
    def __init__(
        self,
        kaggle_dataset: str = "aleksandrpikul222/nuaaaa",
        output_dir: str = "./dataset",
        max_images: Optional[int] = None,
        image_size: Tuple[int, int] = (224, 224),
        seed: int = 42
    ):
        self.kaggle_dataset = kaggle_dataset
        self.output_dir = Path(output_dir)
        self.max_images = max_images
        self.image_size = image_size
        self.seed = seed
        
        random.seed(seed)
        np.random.seed(seed)
        
        self.data_root = None
    
    def download_from_kaggle(self) -> bool:
        """Download dataset from Kaggle using kagglehub."""
        print("Downloading NUAA dataset from Kaggle...")
        
        try:
            import kagglehub
            path = kagglehub.dataset_download(self.kaggle_dataset)
            print(f"Downloaded to: {path}")
            self.data_root = Path(path)
            return True
        except ImportError:
            print("Error: kagglehub not installed. Install with: pip install kagglehub")
            return False
        except Exception as e:
            print(f"Download failed: {e}")
            print(f"Manual download: https://www.kaggle.com/datasets/{self.kaggle_dataset}")
            return False
    
    def find_data_root(self) -> bool:
        """Find the data root in kagglehub cache."""
        try:
            cache_base = Path.home() / ".cache" / "kagglehub" / "datasets"
            dataset_path = cache_base / self.kaggle_dataset.replace("/", os.sep)
            if dataset_path.exists():
                versions = sorted([d for d in dataset_path.iterdir() if d.is_dir()], reverse=True)
                if versions:
                    self.data_root = versions[0]
                    print(f"Found cached dataset: {self.data_root}")
                    return True
        except Exception:
            pass
        
        return False
    
    def scan_dataset(self) -> Dict[str, List[Path]]:
        """Scan the NUAA dataset structure."""
        print("Scanning dataset structure...")
        
        if not self.data_root:
            return {}
        
        result = {
            'train_real': [],
            'train_spoof': [],
            'val_real': [],
            'val_spoof': [],
        }
        
        # Find image directories (ClientRaw = real, ImposterRaw = spoof)
        client_dir = None
        imposter_dir = None
        
        for root, dirs, files in os.walk(self.data_root):
            root_path = Path(root)
            dirs_lower = {d.lower(): d for d in dirs}
            
            if 'clientraw' in dirs_lower and client_dir is None:
                client_dir = root_path / dirs_lower['clientraw']
                print(f"  Found ClientRaw (real): {client_dir}")
            if 'imposterraw' in dirs_lower and imposter_dir is None:
                imposter_dir = root_path / dirs_lower['imposterraw']
                print(f"  Found ImposterRaw (spoof): {imposter_dir}")
        
        # Collect images from directories
        def collect_images(directory: Optional[Path]) -> List[Path]:
            if not directory or not directory.exists():
                return []
            images = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG', '*.BMP']:
                images.extend(list(directory.glob(f"**/{ext}")))
            return images
        
        all_real = collect_images(client_dir)
        all_spoof = collect_images(imposter_dir)
        
        print(f"  Total: {len(all_real)} real, {len(all_spoof)} spoof")
        
        if not all_real and not all_spoof:
            print("  Error: No images found in ClientRaw/ImposterRaw directories")
            return result
        
        # Shuffle and split 80/20
        random.shuffle(all_real)
        random.shuffle(all_spoof)
        
        real_split = int(len(all_real) * 0.8)
        spoof_split = int(len(all_spoof) * 0.8)
        
        result['train_real'] = all_real[:real_split]
        result['val_real'] = all_real[real_split:]
        result['train_spoof'] = all_spoof[:spoof_split]
        result['val_spoof'] = all_spoof[spoof_split:]
        
        for key, paths in result.items():
            print(f"  {key}: {len(paths)} images")
        
        return result
    
    def copy_images(self, data: Dict[str, List[Path]]):
        """Copy and resize images to output directory."""
        print(f"Organizing dataset into {self.output_dir}...")
        
        # Create output directories
        for split in ['train', 'val']:
            for label in ['real', 'spoof']:
                (self.output_dir / split / label).mkdir(parents=True, exist_ok=True)
        
        stats = {'train': {'real': 0, 'spoof': 0}, 'val': {'real': 0, 'spoof': 0}}
        failed = 0
        
        # Calculate limits per category if max_images is set
        if self.max_images:
            per_split = self.max_images // 2
            per_class = per_split // 2
        else:
            per_class = None
        
        mapping = {
            ('train', 'real'): data['train_real'],
            ('train', 'spoof'): data['train_spoof'],
            ('val', 'real'): data['val_real'],
            ('val', 'spoof'): data['val_spoof'],
        }
        
        for (split, label), paths in mapping.items():
            if per_class:
                paths = paths[:per_class]
            
            for idx, src_path in enumerate(tqdm(paths, desc=f"{split}/{label}")):
                try:
                    dst_path = self.output_dir / split / label / f"{idx:06d}.jpg"
                    
                    with Image.open(src_path) as img:
                        img = img.convert('RGB')
                        try:
                            resample = Image.Resampling.LANCZOS
                        except AttributeError:
                            resample = Image.LANCZOS
                        img = img.resize(self.image_size, resample)
                        img.save(dst_path, 'JPEG', quality=95)
                    
                    stats[split][label] += 1
                except Exception as e:
                    failed += 1
                    if failed <= 3:
                        print(f"  Failed: {src_path} - {e}")
        
        # Print statistics
        print("\nOrganization Complete!")
        print("=" * 50)
        for split in ['train', 'val']:
            total = stats[split]['real'] + stats[split]['spoof']
            print(f"\n{split.upper()}: {total} images")
            print(f"  real:  {stats[split]['real']:>6}")
            print(f"  spoof: {stats[split]['spoof']:>6}")
        
        if failed > 0:
            print(f"\nWarning: Failed to process {failed} images")
        
        # Save metadata
        metadata = {
            'dataset_name': 'NUAA-Organized',
            'source': self.kaggle_dataset,
            'max_images': self.max_images,
            'image_size': list(self.image_size),
            'stats': stats
        }
        with open(self.output_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def run(self):
        """Main execution pipeline."""
        print("=" * 60)
        print("NUAA Face Anti-Spoofing Dataset Organizer")
        if self.max_images:
            print(f"   Max images: {self.max_images}")
        print("=" * 60)
        
        # Check cache first
        if not self.find_data_root():
            if not self.download_from_kaggle():
                print("\nError: Cannot proceed without data.")
                sys.exit(1)
        
        # Scan dataset
        data = self.scan_dataset()
        
        if not any(data.values()):
            print("Error: No images found in dataset!")
            sys.exit(1)
        
        # Copy images
        self.copy_images(data)
        
        print("\n" + "=" * 60)
        print("Dataset ready!")
        print(f"Location: {self.output_dir.absolute()}")
        print("=" * 60)
        print("\nNext: python train.py --data_root ./dataset")


def create_synthetic_dataset(output_dir: str, num_images: int = 100, 
                            image_size: Tuple[int, int] = (224, 224), seed: int = 42):
    """Create synthetic dataset for testing without downloading."""
    print("=" * 60)
    print("Creating Synthetic Test Dataset")
    print(f"   Total images: {num_images}")
    print("=" * 60)
    
    random.seed(seed)
    np.random.seed(seed)
    
    output_path = Path(output_dir)
    
    for split in ['train', 'val']:
        for label in ['real', 'spoof']:
            (output_path / split / label).mkdir(parents=True, exist_ok=True)
    
    train_per_class = int(num_images * 0.8 / 2)
    val_per_class = int(num_images * 0.2 / 2)
    
    stats = {'train': {'real': 0, 'spoof': 0}, 'val': {'real': 0, 'spoof': 0}}
    
    for split, count in [('train', train_per_class), ('val', val_per_class)]:
        for label in ['real', 'spoof']:
            print(f"  Generating {count} {split}/{label} images...")
            for i in range(count):
                img = np.random.randint(100, 200, (image_size[1], image_size[0], 3), dtype=np.uint8)
                if label == 'spoof':
                    img[::8, :, :] = np.clip(img[::8, :, :].astype(int) + 30, 0, 255).astype(np.uint8)
                
                Image.fromarray(img, 'RGB').save(
                    output_path / split / label / f"{i:06d}.jpg", 'JPEG', quality=95
                )
                stats[split][label] += 1
    
    metadata = {'dataset_name': 'Synthetic-Test', 'num_images': num_images, 'stats': stats}
    with open(output_path / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nSynthetic dataset created at: {output_path.absolute()}")


def main():
    parser = argparse.ArgumentParser(description='NUAA Dataset Organizer')
    parser.add_argument('--output-dir', default='./dataset', help='Output directory')
    parser.add_argument('--max-images', type=int, default=None, help='Limit total images')
    parser.add_argument('--image-size', type=int, nargs=2, default=[224, 224])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--synthetic', action='store_true', help='Generate synthetic test data')
    parser.add_argument('--synthetic-count', type=int, default=100)
    
    args = parser.parse_args()
    
    if args.synthetic:
        create_synthetic_dataset(args.output_dir, args.synthetic_count, 
                                tuple(args.image_size), args.seed)
    else:
        organizer = NUAADatasetOrganizer(
            output_dir=args.output_dir,
            max_images=args.max_images,
            image_size=tuple(args.image_size),
            seed=args.seed
        )
        organizer.run()


if __name__ == "__main__":
    main()
