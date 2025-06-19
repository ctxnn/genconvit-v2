#!/usr/bin/env python3
"""
End-to-End Video DeepFake Detection Training Pipeline
===================================================

This script provides a complete pipeline for training a DeepFake detection model
starting from raw video files. It handles:

1. Frame extraction from videos
2. Dataset organization and splitting
3. Model training with the GenConViT architecture
4. Evaluation and model saving

Usage:
    python train_from_videos.py --video_dir /path/to/videos --output_dir /path/to/output
    
    # With custom parameters
    python train_from_videos.py --video_dir ./videos --output_dir ./training_data \
        --frame_interval 10 --batch_size 16 --epochs 50 --lr 0.0001
"""

import os
import sys
import json
import time
import shutil
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from sklearn.model_selection import train_test_split

# Import our modules
from extract_frames import FrameExtractor
from video_dataset import VideoFrameDataModule, create_video_transforms, analyze_dataset
from genconvit import GenConViT, vae_loss, evaluate

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class VideoTrainingPipeline:
    """
    Complete pipeline for training DeepFake detection models from raw videos.
    """
    
    def __init__(self, 
                 video_dir: str,
                 output_dir: str,
                 frame_interval: int = 5,
                 target_size: Tuple[int, int] = (224, 224),
                 max_frames_per_video: Optional[int] = None,
                 face_detection: bool = False,
                 train_split: float = 0.7,
                 val_split: float = 0.15,
                 test_split: float = 0.15,
                 batch_size: int = 32,
                 epochs: int = 20,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-5,
                 beta: float = 1.0,
                 num_workers: int = 4,
                 balanced_sampling: bool = True,
                 frames_per_video: Optional[int] = None,
                 resume_from_checkpoint: Optional[str] = None,
                 class_mapping: Optional[Dict[str, str]] = None):
        """
        Initialize the training pipeline.
        
        Args:
            video_dir: Directory containing video files organized by class
            output_dir: Directory to save extracted frames and models
            frame_interval: Extract every Nth frame
            target_size: Resize frames to this size
            max_frames_per_video: Maximum frames to extract per video
            face_detection: Whether to use face detection
            train_split: Proportion of data for training
            val_split: Proportion of data for validation
            test_split: Proportion of data for testing
            batch_size: Training batch size
            epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
            beta: Weight for VAE loss
            num_workers: Number of data loading workers
            balanced_sampling: Whether to use balanced sampling
            frames_per_video: Number of frames to sample per video during training
            resume_from_checkpoint: Path to checkpoint to resume from
            class_mapping: Optional mapping of directory names to class names
        """
        self.video_dir = Path(video_dir)
        self.output_dir = Path(output_dir)
        self.frame_interval = frame_interval
        self.target_size = target_size
        self.max_frames_per_video = max_frames_per_video
        self.face_detection = face_detection
        
        # Data splits
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        
        # Training parameters
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.beta = beta
        self.num_workers = num_workers
        self.balanced_sampling = balanced_sampling
        self.frames_per_video = frames_per_video
        self.resume_from_checkpoint = resume_from_checkpoint
        
        # Class mapping
        self.class_mapping = class_mapping or {}
        
        # Paths
        self.frames_dir = self.output_dir / "extracted_frames"
        self.models_dir = self.output_dir / "models"
        self.logs_dir = self.output_dir / "logs"
        
        # Ensure output directories exist
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
    def discover_video_structure(self) -> Dict[str, List[Path]]:
        """
        Discover the structure of video files and organize by class.
        
        Returns:
            Dictionary mapping class names to lists of video files
        """
        logger.info("Discovering video file structure...")
        
        # Supported video extensions
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.m4v']
        
        class_videos = {}
        
        # Check if videos are organized in subdirectories (class folders)
        subdirs = [d for d in self.video_dir.iterdir() if d.is_dir()]
        
        if subdirs:
            # Videos are organized by class in subdirectories
            for class_dir in subdirs:
                class_name = self.class_mapping.get(class_dir.name, class_dir.name)
                videos = []
                
                for ext in video_extensions:
                    videos.extend(class_dir.rglob(f"*{ext}"))
                    videos.extend(class_dir.rglob(f"*{ext.upper()}"))
                
                if videos:
                    class_videos[class_name] = videos
                    logger.info(f"Found {len(videos)} videos in class '{class_name}'")
        else:
            # All videos in one directory - need to infer classes from filenames
            logger.warning("No class subdirectories found. Attempting to infer classes from filenames.")
            
            all_videos = []
            for ext in video_extensions:
                all_videos.extend(self.video_dir.glob(f"*{ext}"))
                all_videos.extend(self.video_dir.glob(f"*{ext.upper()}"))
            
            # Try to infer classes from common naming patterns
            for video in all_videos:
                # Common patterns: real_xxx, fake_xxx, deepfake_xxx, original_xxx
                name_lower = video.stem.lower()
                
                if any(pattern in name_lower for pattern in ['real', 'original', 'genuine']):
                    class_name = 'real'
                elif any(pattern in name_lower for pattern in ['fake', 'deepfake', 'synthetic']):
                    class_name = 'fake'
                else:
                    # Default class
                    class_name = 'unknown'
                
                if class_name not in class_videos:
                    class_videos[class_name] = []
                class_videos[class_name].append(video)
        
        # Log summary
        total_videos = sum(len(videos) for videos in class_videos.values())
        logger.info(f"Discovered {total_videos} videos across {len(class_videos)} classes:")
        for class_name, videos in class_videos.items():
            logger.info(f"  {class_name}: {len(videos)} videos")
        
        return class_videos
    
    def split_videos(self, class_videos: Dict[str, List[Path]]) -> Dict[str, Dict[str, List[Path]]]:
        """
        Split videos into train/val/test sets while maintaining class balance.
        
        Args:
            class_videos: Dictionary mapping class names to video lists
            
        Returns:
            Dictionary with train/val/test splits for each class
        """
        logger.info("Splitting videos into train/val/test sets...")
        
        splits = {'train': {}, 'val': {}, 'test': {}}
        
        for class_name, videos in class_videos.items():
            # Shuffle videos for random splitting
            videos_shuffled = videos.copy()
            np.random.shuffle(videos_shuffled)
            
            n_videos = len(videos_shuffled)
            n_train = int(n_videos * self.train_split)
            n_val = int(n_videos * self.val_split)
            
            # Split videos
            train_videos = videos_shuffled[:n_train]
            val_videos = videos_shuffled[n_train:n_train + n_val]
            test_videos = videos_shuffled[n_train + n_val:]
            
            splits['train'][class_name] = train_videos
            splits['val'][class_name] = val_videos
            splits['test'][class_name] = test_videos
            
            logger.info(f"Class '{class_name}': {len(train_videos)} train, "
                       f"{len(val_videos)} val, {len(test_videos)} test")
        
        return splits
    
    def extract_frames(self, video_splits: Dict[str, Dict[str, List[Path]]]):
        """
        Extract frames from all videos and organize by split and class.
        
        Args:
            video_splits: Dictionary with train/val/test splits for each class
        """
        logger.info("Starting frame extraction...")
        
        for split_name, class_videos in video_splits.items():
            split_dir = self.frames_dir / split_name
            
            for class_name, videos in class_videos.items():
                if not videos:
                    continue
                    
                class_dir = split_dir / class_name
                class_dir.mkdir(parents=True, exist_ok=True)
                
                logger.info(f"Extracting frames for {split_name}/{class_name} "
                           f"({len(videos)} videos)...")
                
                # Create frame extractor for this class
                extractor = FrameExtractor(
                    input_dir=str(self.video_dir),  # Will be overridden
                    output_dir=str(class_dir),
                    frame_interval=self.frame_interval,
                    target_size=self.target_size,
                    max_frames_per_video=self.max_frames_per_video,
                    face_detection=self.face_detection
                )
                
                # Extract frames from each video
                for video_path in videos:
                    video_output_dir = class_dir / video_path.stem
                    stats = extractor.extract_frames_from_video(video_path, video_output_dir)
                    
                    if not stats['success']:
                        logger.warning(f"Failed to extract frames from {video_path}")
        
        logger.info("Frame extraction completed!")
    
    def create_data_module(self) -> VideoFrameDataModule:
        """
        Create a data module for training.
        
        Returns:
            Configured VideoFrameDataModule
        """
        logger.info("Creating data module...")
        
        # Create transforms
        train_transform = create_video_transforms(
            input_size=self.target_size[0], 
            augment=True
        )
        val_transform = create_video_transforms(
            input_size=self.target_size[0], 
            augment=False
        )
        
        data_module = VideoFrameDataModule(
            data_dir=self.frames_dir,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            train_transform=train_transform,
            val_transform=val_transform,
            frames_per_video=self.frames_per_video,
            balanced_sampling=self.balanced_sampling
        )
        
        return data_module
    
    def train_model(self, data_module: VideoFrameDataModule) -> Dict:
        """
        Train the GenConViT model.
        
        Args:
            data_module: Configured data module
            
        Returns:
            Dictionary with training statistics
        """
        logger.info("Starting model training...")
        
        # Setup data module
        data_module.setup('fit')
        
        # Get dataloaders
        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()
        
        # Print dataset statistics
        if data_module.train_dataset:
            train_stats = analyze_dataset(data_module.train_dataset)
            logger.info(f"Training dataset: {train_stats['total_samples']} samples, "
                       f"{train_stats['num_classes']} classes")
            logger.info(f"Class distribution: {train_stats['class_distribution']}")
        
        # Create model
        num_classes = data_module.get_num_classes()
        model = GenConViT(num_classes=num_classes).to(self.device)
        
        # Optimizer and scheduler
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate, 
                             weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)
        criterion = nn.CrossEntropyLoss()
        
        # Load checkpoint if resuming
        start_epoch = 1
        best_acc = 0.0
        
        if self.resume_from_checkpoint and Path(self.resume_from_checkpoint).exists():
            logger.info(f"Resuming from checkpoint: {self.resume_from_checkpoint}")
            checkpoint = torch.load(self.resume_from_checkpoint, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_acc = checkpoint['best_acc']
        
        # Training loop
        training_stats = {
            'epochs': [],
            'train_loss': [],
            'train_acc': [],
            'val_acc': [],
            'learning_rates': [],
            'best_acc': best_acc
        }
        
        for epoch in range(start_epoch, self.epochs + 1):
            model.train()
            total_loss = 0.0
            total = 0
            correct = 0
            
            for batch_idx, (imgs, labels) in enumerate(train_loader):
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                logits, la, lb, mu, logvar = model(imgs)
                
                # Combined loss
                loss_cls = criterion(logits, labels)
                loss_vae = vae_loss(model.vae(imgs)[0], imgs, mu, logvar)
                loss = loss_cls + self.beta * loss_vae
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item() * imgs.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
                # Log progress
                if batch_idx % 50 == 0:
                    logger.info(f"Epoch {epoch}, Batch {batch_idx}, "
                               f"Loss: {loss.item():.4f}")
            
            train_acc = correct / total
            avg_loss = total_loss / total
            val_acc = evaluate(model, val_loader, self.device)
            
            # Step scheduler
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            
            # Log epoch results
            logger.info(f"Epoch {epoch}: Loss {avg_loss:.4f}, "
                       f"Train Acc {train_acc:.4f}, Val Acc {val_acc:.4f}, "
                       f"LR {current_lr:.6f}")
            
            # Save statistics
            training_stats['epochs'].append(epoch)
            training_stats['train_loss'].append(avg_loss)
            training_stats['train_acc'].append(train_acc)
            training_stats['val_acc'].append(val_acc)
            training_stats['learning_rates'].append(current_lr)
            
            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                training_stats['best_acc'] = best_acc
                
                best_model_path = self.models_dir / 'best_model.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_acc': best_acc,
                    'class_names': data_module.get_class_names(),
                    'training_stats': training_stats
                }, best_model_path)
                
                logger.info(f"New best model saved with validation accuracy: {best_acc:.4f}")
            
            # Save checkpoint every 5 epochs
            if epoch % 5 == 0:
                checkpoint_path = self.models_dir / f'checkpoint_epoch_{epoch}.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_acc': best_acc,
                    'class_names': data_module.get_class_names(),
                    'training_stats': training_stats
                }, checkpoint_path)
        
        logger.info(f"Training completed! Best validation accuracy: {best_acc:.4f}")
        
        return training_stats
    
    def evaluate_model(self, data_module: VideoFrameDataModule) -> Dict:
        """
        Evaluate the trained model on test data.
        
        Args:
            data_module: Configured data module
            
        Returns:
            Dictionary with evaluation results
        """
        logger.info("Evaluating model on test data...")
        
        # Setup test data
        data_module.setup('test')
        
        # Load best model
        best_model_path = self.models_dir / 'best_model.pth'
        if not best_model_path.exists():
            logger.error("No trained model found!")
            return {}
        
        checkpoint = torch.load(best_model_path, map_location=self.device)
        
        # Create model
        num_classes = len(checkpoint['class_names'])
        model = GenConViT(num_classes=num_classes).to(self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Get test loader
        if data_module.test_dataset:
            test_loader = data_module.test_dataloader()
            test_acc = evaluate(model, test_loader, self.device)
            
            logger.info(f"Test accuracy: {test_acc:.4f}")
            
            return {
                'test_accuracy': test_acc,
                'model_path': str(best_model_path),
                'class_names': checkpoint['class_names']
            }
        else:
            logger.warning("No test dataset available")
            return {}
    
    def run_pipeline(self) -> Dict:
        """
        Run the complete training pipeline.
        
        Returns:
            Dictionary with pipeline results
        """
        start_time = time.time()
        
        logger.info("="*60)
        logger.info("STARTING VIDEO DEEPFAKE DETECTION TRAINING PIPELINE")
        logger.info("="*60)
        
        try:
            # Step 1: Discover video structure
            class_videos = self.discover_video_structure()
            
            if not class_videos:
                raise ValueError("No videos found in the input directory!")
            
            # Step 2: Split videos
            video_splits = self.split_videos(class_videos)
            
            # Step 3: Extract frames (skip if already done)
            if not any((self.frames_dir / split).exists() for split in ['train', 'val', 'test']):
                self.extract_frames(video_splits)
            else:
                logger.info("Frame extraction directories already exist, skipping extraction...")
            
            # Step 4: Create data module
            data_module = self.create_data_module()
            
            # Step 5: Train model
            training_stats = self.train_model(data_module)
            
            # Step 6: Evaluate model
            eval_results = self.evaluate_model(data_module)
            
            # Step 7: Save pipeline results
            pipeline_results = {
                'pipeline_config': {
                    'video_dir': str(self.video_dir),
                    'output_dir': str(self.output_dir),
                    'frame_interval': self.frame_interval,
                    'target_size': self.target_size,
                    'batch_size': self.batch_size,
                    'epochs': self.epochs,
                    'learning_rate': self.learning_rate
                },
                'video_statistics': {
                    'total_classes': len(class_videos),
                    'class_distribution': {k: len(v) for k, v in class_videos.items()}
                },
                'training_stats': training_stats,
                'evaluation_results': eval_results,
                'total_time_minutes': (time.time() - start_time) / 60
            }
            
            results_path = self.output_dir / 'pipeline_results.json'
            with open(results_path, 'w') as f:
                json.dump(pipeline_results, f, indent=2, default=str)
            
            logger.info("="*60)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info(f"Total time: {pipeline_results['total_time_minutes']:.2f} minutes")
            logger.info(f"Best validation accuracy: {training_stats['best_acc']:.4f}")
            if eval_results:
                logger.info(f"Test accuracy: {eval_results['test_accuracy']:.4f}")
            logger.info(f"Results saved to: {results_path}")
            logger.info("="*60)
            
            return pipeline_results
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end video deepfake detection training pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python train_from_videos.py --video_dir ./videos --output_dir ./training_output
  
  # With custom parameters
  python train_from_videos.py --video_dir ./videos --output_dir ./output \
    --frame_interval 10 --batch_size 16 --epochs 50 --lr 0.0001
  
  # With face detection and balanced sampling
  python train_from_videos.py --video_dir ./videos --output_dir ./output \
    --face_detection --balanced_sampling --max_frames 200
        """
    )
    
    # Data arguments
    parser.add_argument('--video_dir', type=str, required=True,
                       help='Directory containing video files (organized by class)')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save extracted frames and trained models')
    
    # Frame extraction arguments
    parser.add_argument('--frame_interval', type=int, default=5,
                       help='Extract every Nth frame (default: 5)')
    parser.add_argument('--target_width', type=int, default=224,
                       help='Target frame width (default: 224)')
    parser.add_argument('--target_height', type=int, default=224,
                       help='Target frame height (default: 224)')
    parser.add_argument('--max_frames', type=int, default=None,
                       help='Maximum frames to extract per video (default: no limit)')
    parser.add_argument('--face_detection', action='store_true',
                       help='Enable face detection and cropping')
    
    # Data splitting arguments
    parser.add_argument('--train_split', type=float, default=0.7,
                       help='Proportion of data for training (default: 0.7)')
    parser.add_argument('--val_split', type=float, default=0.15,
                       help='Proportion of data for validation (default: 0.15)')
    parser.add_argument('--test_split', type=float, default=0.15,
                       help='Proportion of data for testing (default: 0.15)')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Training batch size (default: 32)')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs (default: 20)')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate (default: 1e-4)')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay (default: 1e-5)')
    parser.add_argument('--beta', type=float, default=1.0,
                       help='Weight for VAE loss (default: 1.0)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers (default: 4)')
    parser.add_argument('--balanced_sampling', action='store_true',
                       help='Use balanced sampling for training')
    parser.add_argument('--frames_per_video', type=int, default=None,
                       help='Number of frames to sample per video during training')
    
    # Other arguments
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                       help='Path to checkpoint to resume training from')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Validate arguments
    if args.train_split + args.val_split + args.test_split != 1.0:
        logger.error("Train, validation, and test splits must sum to 1.0")
        return 1
    
    # Create pipeline
    pipeline = VideoTrainingPipeline(
        video_dir=args.video_dir,
        output_dir=args.output_dir,
        frame_interval=args.frame_interval,
        target_size=(args.target_width, args.target_height),
        max_frames_per_video=args.max_frames,
        face_detection=args.face_detection,
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=args.test_split,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        beta=args.beta,
        num_workers=args.num_workers,
        balanced_sampling=args.balanced_sampling,
        frames_per_video=args.frames_per_video,
        resume_from_checkpoint=args.resume_from_checkpoint
    )
    
    try:
        # Run the pipeline
        results = pipeline.run_pipeline()
        logger.info("Pipeline completed successfully!")
        return 0
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        return 1


if __name__ == '__main__':
    exit(main())