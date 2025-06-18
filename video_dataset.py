#!/usr/bin/env python3
"""
Video Frame Dataset Loader for DeepFake Detection
================================================

This module provides custom dataset classes for loading and preprocessing
video frames extracted from deepfake detection datasets. It supports
various data organization structures and provides efficient loading
with proper data augmentation.

Classes:
    VideoFrameDataset: Main dataset class for loading extracted frames
    BalancedVideoFrameDataset: Balanced sampling dataset for imbalanced data
    VideoFrameDataModule: PyTorch Lightning data module wrapper
"""

import os
import random
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Callable
import logging

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


class VideoFrameDataset(Dataset):
    """
    Dataset class for loading video frames from extracted frame directories.
    
    Supports multiple directory structures:
    1. Standard structure: root/class_name/video_id/frames/
    2. Flat structure: root/class_name/frame_files
    3. Custom structure with metadata file
    """
    
    def __init__(self,
                 root_dir: Union[str, Path],
                 split: str = 'train',
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 class_names: Optional[List[str]] = None,
                 frames_per_video: Optional[int] = None,
                 random_frame_selection: bool = True,
                 frame_extensions: List[str] = None,
                 min_frames_per_video: int = 1,
                 metadata_file: Optional[str] = None,
                 cache_paths: bool = True):
        """
        Initialize the dataset.
        
        Args:
            root_dir: Root directory containing the frame data
            split: Data split ('train', 'val', 'test')
            transform: Transform to apply to frames
            target_transform: Transform to apply to labels
            class_names: List of class names (auto-detected if None)
            frames_per_video: Number of frames to sample per video (None = all)
            random_frame_selection: Whether to randomly select frames
            frame_extensions: Valid frame file extensions
            min_frames_per_video: Minimum frames required per video
            metadata_file: Optional metadata file with video information
            cache_paths: Whether to cache file paths for faster loading
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.frames_per_video = frames_per_video
        self.random_frame_selection = random_frame_selection
        self.min_frames_per_video = min_frames_per_video
        self.cache_paths = cache_paths
        
        if frame_extensions is None:
            self.frame_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        else:
            self.frame_extensions = frame_extensions
            
        # Load metadata if provided
        self.metadata = {}
        if metadata_file and Path(metadata_file).exists():
            with open(metadata_file, 'r') as f:
                self.metadata = json.load(f)
        
        # Cache for faster loading (must be initialized before building dataset)
        self._path_cache = {} if cache_paths else None
        
        # Discover classes and build dataset
        self.class_names = class_names or self._discover_classes()
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.class_names)}
        
        # Build the dataset
        self.samples = self._build_dataset()
        
        logger.info(f"Created dataset with {len(self.samples)} samples from {len(self.class_names)} classes")
    
    def _discover_classes(self) -> List[str]:
        """Automatically discover class names from directory structure."""
        split_dir = self.root_dir / self.split
        if not split_dir.exists():
            split_dir = self.root_dir  # Fallback to root if split dir doesn't exist
            
        classes = []
        for item in split_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                classes.append(item.name)
        
        classes.sort()
        logger.info(f"Discovered classes: {classes}")
        return classes
    
    def _is_valid_frame(self, path: Path) -> bool:
        """Check if a file is a valid frame."""
        return path.suffix.lower() in self.frame_extensions
    
    def _get_video_frames(self, video_dir: Path) -> List[Path]:
        """Get all valid frame files from a video directory."""
        if self._path_cache and str(video_dir) in self._path_cache:
            return self._path_cache[str(video_dir)]
            
        frames = []
        for frame_file in video_dir.iterdir():
            if self._is_valid_frame(frame_file):
                frames.append(frame_file)
        
        frames.sort()  # Ensure consistent ordering
        
        if self._path_cache:
            self._path_cache[str(video_dir)] = frames
            
        return frames
    
    def _build_dataset(self) -> List[Dict]:
        """Build the dataset by scanning directories."""
        samples = []
        
        # Determine the base directory
        split_dir = self.root_dir / self.split
        if not split_dir.exists():
            split_dir = self.root_dir
            logger.warning(f"Split directory {self.split} not found, using root directory")
        
        # Scan each class directory
        for class_name in self.class_names:
            class_dir = split_dir / class_name
            if not class_dir.exists():
                logger.warning(f"Class directory not found: {class_dir}")
                continue
                
            class_idx = self.class_to_idx[class_name]
            
            # Two possible structures:
            # 1. class_dir contains video subdirectories
            # 2. class_dir contains frame files directly
            
            # Check if there are subdirectories (video folders)
            subdirs = [d for d in class_dir.iterdir() if d.is_dir()]
            
            if subdirs:
                # Structure 1: class_dir/video_id/frames
                for video_dir in subdirs:
                    frames = self._get_video_frames(video_dir)
                    
                    if len(frames) >= self.min_frames_per_video:
                        samples.append({
                            'video_id': video_dir.name,
                            'class_name': class_name,
                            'class_idx': class_idx,
                            'frames': frames,
                            'video_dir': video_dir
                        })
            else:
                # Structure 2: class_dir contains frames directly
                frames = self._get_video_frames(class_dir)
                
                if len(frames) >= self.min_frames_per_video:
                    samples.append({
                        'video_id': class_name,
                        'class_name': class_name,
                        'class_idx': class_idx,
                        'frames': frames,
                        'video_dir': class_dir
                    })
        
        return samples
    
    def _select_frames(self, frames: List[Path]) -> List[Path]:
        """Select frames from a video according to the sampling strategy."""
        if self.frames_per_video is None:
            return frames
            
        if len(frames) <= self.frames_per_video:
            return frames
            
        if self.random_frame_selection:
            return random.sample(frames, self.frames_per_video)
        else:
            # Uniform sampling
            indices = np.linspace(0, len(frames) - 1, self.frames_per_video, dtype=int)
            return [frames[i] for i in indices]
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample from the dataset.
        
        Returns:
            Tuple of (frame_tensor, class_idx)
            If frames_per_video > 1, returns (stacked_frames, class_idx)
        """
        sample = self.samples[idx]
        frames = sample['frames']
        class_idx = sample['class_idx']
        
        # Select frames according to strategy
        selected_frames = self._select_frames(frames)
        
        # Load and process frames
        processed_frames = []
        for frame_path in selected_frames:
            try:
                # Load image
                image = Image.open(frame_path).convert('RGB')
                
                # Apply transforms
                if self.transform:
                    image = self.transform(image)
                    
                processed_frames.append(image)
                
            except Exception as e:
                logger.warning(f"Failed to load frame {frame_path}: {e}")
                # Skip this frame or use a default
                continue
        
        if not processed_frames:
            raise RuntimeError(f"No valid frames found for sample {idx}")
        
        # Handle multiple frames
        if len(processed_frames) == 1:
            frame_tensor = processed_frames[0]
        else:
            # Stack frames along a new dimension
            frame_tensor = torch.stack(processed_frames)
        
        # Apply target transform
        if self.target_transform:
            class_idx = self.target_transform(class_idx)
            
        return frame_tensor, class_idx
    
    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for balanced training."""
        class_counts = torch.zeros(len(self.class_names))
        
        for sample in self.samples:
            class_counts[sample['class_idx']] += 1
            
        # Inverse frequency weighting
        total_samples = len(self.samples)
        class_weights = total_samples / (len(self.class_names) * class_counts)
        
        return class_weights
    
    def get_sample_weights(self) -> List[float]:
        """Get per-sample weights for balanced sampling."""
        class_weights = self.get_class_weights()
        sample_weights = []
        
        for sample in self.samples:
            sample_weights.append(class_weights[sample['class_idx']].item())
            
        return sample_weights


class BalancedVideoFrameDataset(VideoFrameDataset):
    """
    Balanced version of VideoFrameDataset that ensures equal representation
    of all classes in each epoch through weighted sampling.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Ensure _path_cache is properly initialized (redundant safety check)
        if not hasattr(self, '_path_cache'):
            self._path_cache = {} if kwargs.get('cache_paths', False) else None
        self.sample_weights = self.get_sample_weights()
    
    def get_balanced_sampler(self) -> WeightedRandomSampler:
        """Get a weighted random sampler for balanced training."""
        return WeightedRandomSampler(
            weights=self.sample_weights,
            num_samples=len(self.samples),
            replacement=True
        )


class VideoFrameDataModule:
    """
    Data module for managing train/val/test datasets and dataloaders.
    """
    
    def __init__(self,
                 data_dir: Union[str, Path],
                 batch_size: int = 32,
                 num_workers: int = 4,
                 pin_memory: bool = True,
                 train_transform: Optional[Callable] = None,
                 val_transform: Optional[Callable] = None,
                 test_transform: Optional[Callable] = None,
                 class_names: Optional[List[str]] = None,
                 frames_per_video: Optional[int] = None,
                 balanced_sampling: bool = False,
                 **dataset_kwargs):
        """
        Initialize the data module.
        
        Args:
            data_dir: Root directory containing train/val/test splits
            batch_size: Batch size for dataloaders
            num_workers: Number of worker processes for data loading
            pin_memory: Whether to pin memory for faster GPU transfer
            train_transform: Transform pipeline for training data
            val_transform: Transform pipeline for validation data
            test_transform: Transform pipeline for test data
            class_names: List of class names
            frames_per_video: Number of frames to sample per video
            balanced_sampling: Whether to use balanced sampling for training
            **dataset_kwargs: Additional arguments for dataset initialization
        """
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.class_names = class_names
        self.frames_per_video = frames_per_video
        self.balanced_sampling = balanced_sampling
        self.dataset_kwargs = dataset_kwargs
        
        # Set up transforms
        self.train_transform = train_transform or self._get_default_train_transform()
        self.val_transform = val_transform or self._get_default_val_transform()
        self.test_transform = test_transform or self._get_default_val_transform()
        
        # Initialize datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def _get_default_train_transform(self):
        """Get default training transforms with augmentation."""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _get_default_val_transform(self):
        """Get default validation transforms without augmentation."""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def setup(self, stage: str = None):
        """Set up datasets for the specified stage."""
        if stage == 'fit' or stage is None:
            # Training dataset
            if (self.data_dir / 'train').exists():
                dataset_class = BalancedVideoFrameDataset if self.balanced_sampling else VideoFrameDataset
                self.train_dataset = dataset_class(
                    root_dir=self.data_dir,
                    split='train',
                    transform=self.train_transform,
                    class_names=self.class_names,
                    frames_per_video=self.frames_per_video,
                    random_frame_selection=True,
                    **self.dataset_kwargs
                )
                
                # Auto-detect class names from training data
                if self.class_names is None:
                    self.class_names = self.train_dataset.class_names
            
            # Validation dataset
            if (self.data_dir / 'val').exists():
                self.val_dataset = VideoFrameDataset(
                    root_dir=self.data_dir,
                    split='val',
                    transform=self.val_transform,
                    class_names=self.class_names,
                    frames_per_video=self.frames_per_video,
                    random_frame_selection=False,
                    **self.dataset_kwargs
                )
        
        if stage == 'test' or stage is None:
            # Test dataset
            if (self.data_dir / 'test').exists():
                self.test_dataset = VideoFrameDataset(
                    root_dir=self.data_dir,
                    split='test',
                    transform=self.test_transform,
                    class_names=self.class_names,
                    frames_per_video=self.frames_per_video,
                    random_frame_selection=False,
                    **self.dataset_kwargs
                )
    
    def train_dataloader(self) -> DataLoader:
        """Get training dataloader."""
        if self.train_dataset is None:
            raise RuntimeError("Training dataset not initialized. Call setup() first.")
        
        sampler = None
        if self.balanced_sampling and hasattr(self.train_dataset, 'get_balanced_sampler'):
            sampler = self.train_dataset.get_balanced_sampler()
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0
        )
    
    def val_dataloader(self) -> DataLoader:
        """Get validation dataloader."""
        if self.val_dataset is None:
            raise RuntimeError("Validation dataset not initialized. Call setup() first.")
        
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0
        )
    
    def test_dataloader(self) -> DataLoader:
        """Get test dataloader."""
        if self.test_dataset is None:
            raise RuntimeError("Test dataset not initialized. Call setup() first.")
        
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0
        )
    
    def get_class_names(self) -> List[str]:
        """Get class names."""
        if self.class_names is None:
            raise RuntimeError("Class names not available. Call setup() first.")
        return self.class_names
    
    def get_num_classes(self) -> int:
        """Get number of classes."""
        return len(self.get_class_names())


# Utility functions
def create_video_transforms(input_size: int = 224, 
                          augment: bool = True) -> transforms.Compose:
    """
    Create transform pipeline for video frames.
    
    Args:
        input_size: Target size for frames
        augment: Whether to apply data augmentation
        
    Returns:
        Transform pipeline
    """
    transform_list = [transforms.Resize((input_size, input_size))]
    
    if augment:
        transform_list.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        ])
    
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return transforms.Compose(transform_list)


def analyze_dataset(dataset: VideoFrameDataset) -> Dict:
    """
    Analyze dataset statistics.
    
    Args:
        dataset: Dataset to analyze
        
    Returns:
        Dictionary with dataset statistics
    """
    stats = {
        'total_samples': len(dataset),
        'num_classes': len(dataset.class_names),
        'class_names': dataset.class_names,
        'class_distribution': {},
        'frames_per_video': [],
        'total_frames': 0
    }
    
    # Analyze class distribution
    for sample in dataset.samples:
        class_name = sample['class_name']
        if class_name not in stats['class_distribution']:
            stats['class_distribution'][class_name] = 0
        stats['class_distribution'][class_name] += 1
        
        # Count frames
        num_frames = len(sample['frames'])
        stats['frames_per_video'].append(num_frames)
        stats['total_frames'] += num_frames
    
    # Calculate frame statistics
    frames_array = np.array(stats['frames_per_video'])
    stats['avg_frames_per_video'] = float(np.mean(frames_array))
    stats['min_frames_per_video'] = int(np.min(frames_array))
    stats['max_frames_per_video'] = int(np.max(frames_array))
    stats['std_frames_per_video'] = float(np.std(frames_array))
    
    return stats