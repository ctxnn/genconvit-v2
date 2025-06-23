#!/usr/bin/env python3
"""
Classification-Only GenConViT Training Script
===========================================

This script eliminates NaN issues by focusing ONLY on classification.
No reconstruction losses - pure classification training.

Usage:
    python training/train_classification_only.py --data ./data --batch-size 16 --lr 0.001
"""

import argparse
import os
import json
import logging
import time
import warnings
from pathlib import Path
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

# Import our modules
import sys
sys.path.append(str(Path(__file__).parent.parent))

from models import create_genconvit
from utils import VideoFrameDataModule, create_video_transforms, analyze_dataset

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_classification_only.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def classification_only_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Pure classification loss - no reconstruction components.
    """
    return F.cross_entropy(logits, targets)


class ClassificationOnlyTrainer:
    """Pure classification trainer - no reconstruction losses"""
    
    def __init__(self, 
                 model,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 device: torch.device,
                 config: Dict[str, Any]):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        
        # Conservative optimizer to prevent gradient explosion
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay'],
            eps=1e-8,
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True,
            min_lr=1e-6
        )
        
        # Tracking
        self.best_val_acc = 0.0
        self.best_val_loss = float('inf')
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch - classification only"""
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Handle video frames if needed
            if len(images.shape) == 5:
                batch_size, num_frames = images.shape[0], images.shape[1]
                images = images.view(-1, images.shape[2], images.shape[3], images.shape[4])
                labels = labels.repeat_interleave(num_frames)
            
            self.optimizer.zero_grad()
            
            # Forward pass - only get classification logits
            try:
                logits, _, _, _, _ = self.model(images)
            except Exception as e:
                logger.error(f"Forward pass failed at batch {batch_idx}: {e}")
                continue
            
            # Pure classification loss
            loss = classification_only_loss(logits, labels)
            
            # Check for NaN
            if torch.isnan(loss) or torch.isinf(loss):
                logger.error(f"NaN/Inf loss at batch {batch_idx}: {loss.item()}")
                continue
            
            # Backward pass
            loss.backward()
            
            # Very aggressive gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.1)
            
            # Check gradients and skip if still too large
            if grad_norm > 1.0:
                logger.warning(f"Large gradient norm: {grad_norm:.4f}, skipping step")
                self.optimizer.zero_grad()
                continue
            
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            # Log progress
            if batch_idx % 50 == 0:
                current_acc = 100. * correct / total
                logger.info(f'Batch {batch_idx}/{len(self.train_loader)}, '
                           f'Loss: {loss.item():.4f}, '
                           f'Acc: {current_acc:.2f}%, '
                           f'Grad Norm: {grad_norm:.4f}')
        
        return {
            'loss': total_loss / len(self.train_loader),
            'accuracy': correct / total
        }
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Handle video frames if needed
                if len(images.shape) == 5:
                    batch_size, num_frames = images.shape[0], images.shape[1]
                    images = images.view(-1, images.shape[2], images.shape[3], images.shape[4])
                    labels = labels.repeat_interleave(num_frames)
                
                # Forward pass
                try:
                    logits, _, _, _, _ = self.model(images)
                except Exception as e:
                    logger.error(f"Validation forward pass failed: {e}")
                    continue
                
                # Classification loss
                loss = classification_only_loss(logits, labels)
                
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    total_loss += loss.item()
                
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        return {
            'loss': total_loss / len(self.val_loader) if len(self.val_loader) > 0 else float('inf'),
            'accuracy': correct / total if total > 0 else 0.0
        }
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint_path = self.config['save_path']
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'history': self.history
        }
        
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            # Save best model with timestamp
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            best_path = checkpoint_path.replace('.pth', f'_best_{timestamp}.pth')
            torch.save(checkpoint, best_path)
            logger.info(f"New best model saved: {checkpoint_path}")
            logger.info(f"Timestamped best model saved: {best_path}")
    
    def train(self) -> Dict[str, Any]:
        """Main training loop - classification only"""
        logger.info("Starting CLASSIFICATION-ONLY training:")
        logger.info(f"  - Learning rate: {self.config['learning_rate']}")
        logger.info(f"  - No reconstruction losses")
        logger.info(f"  - Pure classification focus")
        
        start_time = time.time()
        early_stop_counter = 0
        early_stop_patience = self.config.get('early_stopping_patience', 15)
        
        for epoch in range(1, self.config['epochs'] + 1):
            epoch_start = time.time()
            
            # Training
            train_metrics = self.train_epoch()
            
            # Validation  
            val_metrics = self.validate_epoch()
            
            # Update learning rate scheduler
            self.scheduler.step(val_metrics['loss'])
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['learning_rates'].append(current_lr)
            
            # Check for best model
            is_best = val_metrics['accuracy'] > self.best_val_acc
            if is_best:
                self.best_val_acc = val_metrics['accuracy']
                self.best_val_loss = val_metrics['loss']
                early_stop_counter = 0
            else:
                early_stop_counter += 1
            
            # Log epoch results
            epoch_time = time.time() - epoch_start
            logger.info(f"Epoch {epoch}/{self.config['epochs']} ({epoch_time:.2f}s):")
            logger.info(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
            logger.info(f"  Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
            logger.info(f"  LR: {current_lr:.6f}")
            
            # Save checkpoint
            if epoch % self.config['save_every'] == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
            
            # Early stopping check
            if early_stop_counter >= early_stop_patience:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break
            
            # Success check
            if val_metrics['accuracy'] > 0.8:
                logger.info(f"Great accuracy achieved: {val_metrics['accuracy']:.4f}!")
                break
        
        total_time = time.time() - start_time
        
        # Final results
        results = {
            'best_val_accuracy': self.best_val_acc,
            'best_val_loss': self.best_val_loss,
            'total_epochs': epoch,
            'total_time_minutes': total_time / 60,
            'history': self.history,
            'config': self.config
        }
        
        logger.info(f"Training completed!")
        logger.info(f"Best validation accuracy: {self.best_val_acc:.4f}")
        logger.info(f"Training time: {total_time / 60:.2f} minutes")
        
        return results


def create_config(args) -> Dict[str, Any]:
    """Create configuration dictionary from arguments"""
    return {
        # Model parameters
        'ae_latent': args.ae_latent,
        'vae_latent': args.vae_latent,
        'dropout_rate': args.dropout_rate,
        
        # Training parameters
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.lr,
        'weight_decay': args.weight_decay,
        
        # Early stopping
        'early_stopping_patience': args.early_stopping_patience,
        
        # Other
        'num_workers': args.num_workers,
        'save_path': args.save_path,
        'save_every': args.save_every,
        'balanced_sampling': args.balanced_sampling,
        'frames_per_video': args.frames_per_video
    }


def main():
    parser = argparse.ArgumentParser(description='Classification-Only GenConViT Training')
    
    # Data arguments
    parser.add_argument('--data', type=str, required=True,
                       help='Data root directory with train/val/test folders')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for training (default: 16)')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers (default: 4)')
    parser.add_argument('--balanced-sampling', action='store_true',
                       help='Use balanced sampling for imbalanced datasets')
    parser.add_argument('--frames-per-video', type=int, default=None,
                       help='Number of frames to sample per video')
    
    # Model arguments
    parser.add_argument('--ae-latent', type=int, default=256,
                       help='AutoEncoder latent dimension (default: 256)')
    parser.add_argument('--vae-latent', type=int, default=256,
                       help='VAE latent dimension (default: 256)')
    parser.add_argument('--dropout-rate', type=float, default=0.1,
                       help='Dropout rate (default: 0.1)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs (default: 50)')
    parser.add_argument('--lr', type=float, default=0.0001,
                       help='Learning rate (default: 0.0001)')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                       help='Weight decay for regularization (default: 1e-5)')
    
    # Early stopping arguments
    parser.add_argument('--early-stopping-patience', type=int, default=10,
                       help='Early stopping patience (default: 10)')
    
    # Checkpoint arguments
    parser.add_argument('--save-path', type=str, default='models/genconvit_v2_classification_only.pth',
                       help='Path to save the model')
    parser.add_argument('--save-every', type=int, default=5,
                       help='Save checkpoint every N epochs (default: 5)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create configuration
    config = create_config(args)
    
    logger.info("CLASSIFICATION-ONLY TRAINING CONFIGURATION:")
    logger.info(f"  Learning rate: {config['learning_rate']}")
    logger.info(f"  Dropout rate: {config['dropout_rate']}")
    logger.info(f"  No reconstruction losses!")
    
    # Create data module
    train_transform = create_video_transforms(input_size=224, augment=True)
    val_transform = create_video_transforms(input_size=224, augment=False)
    
    data_module = VideoFrameDataModule(
        data_dir=args.data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_transform=train_transform,
        val_transform=val_transform,
        frames_per_video=args.frames_per_video,
        balanced_sampling=args.balanced_sampling
    )
    
    # Setup datasets
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
    
    if data_module.val_dataset:
        val_stats = analyze_dataset(data_module.val_dataset)
        logger.info(f"Validation dataset: {val_stats['total_samples']} samples")
    
    # Create model
    num_classes = data_module.get_num_classes()
    model = create_genconvit(
        num_classes=num_classes,
        dropout_rate=args.dropout_rate,
        ae_latent=args.ae_latent,
        vae_latent=args.vae_latent
    )
    
    logger.info(f"Created GenConViT model with {num_classes} classes")
    
    # Create output directory
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    
    # Create trainer
    trainer = ClassificationOnlyTrainer(model, train_loader, val_loader, device, config)
    
    # Resume from checkpoint if specified
    if args.resume and os.path.exists(args.resume):
        logger.info(f"Resuming training from: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        trainer.best_val_acc = checkpoint.get('best_val_acc', 0.0)
        trainer.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    
    # Train the model
    results = trainer.train()
    
    # Save final results
    results_path = args.save_path.replace('.pth', '_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Training results saved to: {results_path}")
    logger.info("Classification-only training completed successfully!")
    
    # Print final status
    if results['best_val_accuracy'] > 0.7:
        logger.info("🎉 Excellent! Classification is working well.")
        logger.info("   You can now add back reconstruction losses gradually if needed.")
    elif results['best_val_accuracy'] > 0.6:
        logger.info("👍 Good progress! Classification is learning.")
    else:
        logger.info("⚠️  Still having issues. The problem might be in the data or model architecture.")


if __name__ == '__main__':
    main()