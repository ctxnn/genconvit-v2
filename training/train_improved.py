#!/usr/bin/env python3
"""
Improved GenConViT Training Script
=================================

This script provides enhanced training for GenConViT with:
1. Fixed model architecture with proper feature fusion
2. ReduceLROnPlateau scheduler for adaptive learning rate
3. Early stopping with patience
4. Enhanced regularization and data augmentation
5. Better loss monitoring and logging
6. Comprehensive checkpointing system

Usage:
    python training/train_improved.py --data ./data --batch-size 16 --epochs 100
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
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

# Import our modules with new structure
import sys
sys.path.append(str(Path(__file__).parent.parent))

from models import GenConViT, combined_loss, create_genconvit
from utils import VideoFrameDataModule, create_video_transforms, analyze_dataset

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_improved.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping to avoid overfitting"""
    
    def __init__(self, patience: int = 7, min_delta: float = 0.0, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.early_stop = False
        self.best_weights = None
        
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            self.early_stop = True
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
                
        return self.early_stop
    
    def save_checkpoint(self, model: nn.Module):
        """Save model weights"""
        self.best_weights = model.state_dict().copy()


class ImprovedTrainer:
    """Enhanced training class with all improvements"""
    
    def __init__(self, 
                 model: GenConViT,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 device: torch.device,
                 config: Dict[str, Any]):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        
        # Setup optimizer with weight decay
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay'],
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # IMPROVED: ReduceLROnPlateau scheduler instead of StepLR
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=config['lr_gamma'],
            patience=config['lr_patience'],
            min_lr=config['min_lr'],
            verbose=True
        )
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config['early_stopping_patience'],
            min_delta=config['early_stopping_min_delta']
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': [],
            'loss_components': []
        }
        
        # Best metrics tracking
        self.best_val_acc = 0.0
        self.best_val_loss = float('inf')
        
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch with detailed loss monitoring"""
        self.model.train()
        
        total_loss = 0.0
        total_cls_loss = 0.0
        total_ae_loss = 0.0
        total_vae_loss = 0.0
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
            
            # Forward pass
            logits, logits_a, logits_b, mu, logvar = self.model(images)
            
            # Get reconstructions for loss computation
            ae_reconstructed, vae_reconstructed = self.model.get_reconstructions(images)
            
            # IMPROVED: Use combined loss function
            loss, loss_dict = combined_loss(
                logits=logits,
                targets=labels,
                ae_reconstructed=ae_reconstructed,
                vae_reconstructed=vae_reconstructed,
                original=images,
                mu=mu,
                logvar=logvar,
                classification_weight=self.config['classification_weight'],
                ae_weight=self.config['ae_weight'],
                vae_weight=self.config['vae_weight'],
                vae_beta=self.config['vae_beta']
            )
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            total_cls_loss += loss_dict['classification_loss']
            total_ae_loss += loss_dict['ae_reconstruction_loss']
            total_vae_loss += loss_dict['vae_loss']
            
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            # Log progress
            if batch_idx % self.config['log_interval'] == 0:
                logger.info(f'Batch {batch_idx}/{len(self.train_loader)}, '
                           f'Loss: {loss.item():.4f}, '
                           f'Acc: {100. * correct / total:.2f}%')
        
        return {
            'loss': total_loss / len(self.train_loader),
            'accuracy': correct / total,
            'cls_loss': total_cls_loss / len(self.train_loader),
            'ae_loss': total_ae_loss / len(self.train_loader),
            'vae_loss': total_vae_loss / len(self.train_loader)
        }
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        
        total_loss = 0.0
        total_cls_loss = 0.0
        total_ae_loss = 0.0
        total_vae_loss = 0.0
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
                logits, logits_a, logits_b, mu, logvar = self.model(images)
                
                # Get reconstructions for loss computation
                ae_reconstructed, vae_reconstructed = self.model.get_reconstructions(images)
                
                # Compute loss
                loss, loss_dict = combined_loss(
                    logits=logits,
                    targets=labels,
                    ae_reconstructed=ae_reconstructed,
                    vae_reconstructed=vae_reconstructed,
                    original=images,
                    mu=mu,
                    logvar=logvar,
                    classification_weight=self.config['classification_weight'],
                    ae_weight=self.config['ae_weight'],
                    vae_weight=self.config['vae_weight'],
                    vae_beta=self.config['vae_beta']
                )
                
                # Statistics
                total_loss += loss.item()
                total_cls_loss += loss_dict['classification_loss']
                total_ae_loss += loss_dict['ae_reconstruction_loss']
                total_vae_loss += loss_dict['vae_loss']
                
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        return {
            'loss': total_loss / len(self.val_loader),
            'accuracy': correct / total,
            'cls_loss': total_cls_loss / len(self.val_loader),
            'ae_loss': total_ae_loss / len(self.val_loader),
            'vae_loss': total_vae_loss / len(self.val_loader)
        }
    
    def save_checkpoint(self, epoch: int, is_best: bool = False, save_path: str = None):
        """Save model checkpoint"""
        if save_path is None:
            save_path = self.config['save_path']
            
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
        
        torch.save(checkpoint, save_path)
        
        if is_best:
            best_path = save_path.replace('.pth', '_best.pth')
            torch.save(checkpoint, best_path)
            logger.info(f"New best model saved: {best_path}")
    
    def train(self) -> Dict[str, Any]:
        """Main training loop with all improvements"""
        logger.info("Starting improved training with:")
        logger.info(f"  - ReduceLROnPlateau scheduler")
        logger.info(f"  - Early stopping (patience: {self.config['early_stopping_patience']})")
        logger.info(f"  - Enhanced regularization")
        logger.info(f"  - Fixed model architecture")
        
        start_time = time.time()
        
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
            self.history['loss_components'].append({
                'train_cls': train_metrics['cls_loss'],
                'train_ae': train_metrics['ae_loss'],
                'train_vae': train_metrics['vae_loss'],
                'val_cls': val_metrics['cls_loss'],
                'val_ae': val_metrics['ae_loss'],
                'val_vae': val_metrics['vae_loss']
            })
            
            # Check for best model
            is_best = val_metrics['accuracy'] > self.best_val_acc
            if is_best:
                self.best_val_acc = val_metrics['accuracy']
                self.best_val_loss = val_metrics['loss']
            
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
            if self.early_stopping(val_metrics['loss'], self.model):
                logger.info(f"Early stopping triggered at epoch {epoch}")
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
        
        # Scheduler parameters
        'lr_gamma': args.lr_gamma,
        'lr_patience': args.lr_patience,
        'min_lr': args.min_lr,
        
        # Early stopping
        'early_stopping_patience': args.early_stopping_patience,
        'early_stopping_min_delta': args.early_stopping_min_delta,
        
        # Loss weights
        'classification_weight': args.classification_weight,
        'ae_weight': args.ae_weight,
        'vae_weight': args.vae_weight,
        'vae_beta': args.vae_beta,
        
        # Other
        'num_workers': args.num_workers,
        'save_path': args.save_path,
        'save_every': args.save_every,
        'log_interval': args.log_interval,
        'balanced_sampling': args.balanced_sampling,
        'frames_per_video': args.frames_per_video
    }


def main():
    parser = argparse.ArgumentParser(description='Improved GenConViT Training')
    
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
    parser.add_argument('--dropout-rate', type=float, default=0.5,
                       help='Dropout rate (default: 0.5)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate (default: 1e-4)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='Weight decay for regularization (default: 1e-4)')
    
    # IMPROVED: ReduceLROnPlateau scheduler arguments
    parser.add_argument('--lr-gamma', type=float, default=0.5,
                       help='Learning rate reduction factor (default: 0.5)')
    parser.add_argument('--lr-patience', type=int, default=5,
                       help='Scheduler patience in epochs (default: 5)')
    parser.add_argument('--min-lr', type=float, default=1e-7,
                       help='Minimum learning rate (default: 1e-7)')
    
    # Early stopping arguments
    parser.add_argument('--early-stopping-patience', type=int, default=10,
                       help='Early stopping patience (default: 10)')
    parser.add_argument('--early-stopping-min-delta', type=float, default=0.001,
                       help='Early stopping minimum delta (default: 0.001)')
    
    # Loss weight arguments
    parser.add_argument('--classification-weight', type=float, default=1.0,
                       help='Weight for classification loss (default: 1.0)')
    parser.add_argument('--ae-weight', type=float, default=0.1,
                       help='Weight for AE reconstruction loss (default: 0.1)')
    parser.add_argument('--vae-weight', type=float, default=0.1,
                       help='Weight for VAE loss (default: 0.1)')
    parser.add_argument('--vae-beta', type=float, default=1.0,
                       help='Beta parameter for VAE KL divergence (default: 1.0)')
    
    # Checkpoint arguments
    parser.add_argument('--save-path', type=str, default='models/genconvit_improved.pth',
                       help='Path to save the model (default: models/genconvit_improved.pth)')
    parser.add_argument('--save-every', type=int, default=5,
                       help='Save checkpoint every N epochs (default: 5)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    # Other arguments
    parser.add_argument('--log-interval', type=int, default=50,
                       help='Batch interval for logging (default: 50)')
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
    
    # Create data module with enhanced augmentation
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
    
    # Create model with improvements
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
    trainer = ImprovedTrainer(model, train_loader, val_loader, device, config)
    
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
    logger.info("Training completed successfully!")


if __name__ == '__main__':
    main()