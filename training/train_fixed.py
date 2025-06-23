#!/usr/bin/env python3
"""
Fixed GenConViT Training Script
==============================

This script fixes the learning issues by:
1. Focusing on classification loss first
2. Using higher learning rate initially
3. Simplified loss computation
4. Better gradient flow monitoring
5. Proper loss weight balancing

Usage:
    python training/train_fixed.py --data ./data --batch-size 16 --lr 0.001
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

from models import GenConViT, create_genconvit
from utils import VideoFrameDataModule, create_video_transforms, analyze_dataset

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_fixed.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def simplified_loss(logits: torch.Tensor, 
                   targets: torch.Tensor,
                   ae_reconstructed: torch.Tensor,
                   vae_reconstructed: torch.Tensor,
                   original: torch.Tensor,
                   mu: torch.Tensor,
                   logvar: torch.Tensor,
                   classification_weight: float = 10.0,  # Much higher weight for classification
                   ae_weight: float = 0.01,  # Very low weight for AE
                   vae_weight: float = 0.01,  # Very low weight for VAE
                   vae_beta: float = 0.1) -> Tuple[torch.Tensor, dict]:
    """
    Simplified loss function that focuses on classification first.
    """
    # Classification loss (main focus)
    cls_loss = F.cross_entropy(logits, targets)
    
    # Reconstruction losses (minimal weight)
    ae_recon_loss = F.mse_loss(ae_reconstructed, original)
    
    # VAE loss components
    vae_recon_loss = F.mse_loss(vae_reconstructed, original)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl_loss = kl_loss / (original.size(0) * original.size(1) * original.size(2) * original.size(3))
    vae_total_loss = vae_recon_loss + vae_beta * kl_loss
    
    # Combined loss with heavy emphasis on classification
    total_loss = (classification_weight * cls_loss + 
                  ae_weight * ae_recon_loss + 
                  vae_weight * vae_total_loss)
    
    # Loss dictionary for monitoring
    loss_dict = {
        'total_loss': total_loss.item(),
        'classification_loss': cls_loss.item(),
        'ae_reconstruction_loss': ae_recon_loss.item(), 
        'vae_loss': vae_total_loss.item(),
        'kl_loss': kl_loss.item()
    }
    
    return total_loss, loss_dict


class EarlyStopping:
    """Early stopping to avoid overfitting"""
    
    def __init__(self, patience: int = 7, min_delta: float = 0.001, restore_best_weights: bool = True):
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


class FixedTrainer:
    """Fixed training class that focuses on getting classification working"""
    
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
        
        # Optimizer with higher learning rate
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay'],
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler - less aggressive
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,  # Increased patience
            verbose=True,
            min_lr=1e-6
        )
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.get('early_stopping_patience', 15),
            min_delta=config.get('early_stopping_min_delta', 0.001)
        )
        
        # Tracking
        self.best_val_acc = 0.0
        self.best_val_loss = float('inf')
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': [],
            'loss_components': []
        }
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch with focus on classification"""
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
            
            # Get reconstructions
            ae_reconstructed, vae_reconstructed = self.model.get_reconstructions(images)
            
            # Use simplified loss function
            loss, loss_dict = simplified_loss(
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
            
            # Check for NaN before backward pass
            if torch.isnan(loss):
                logger.error(f"NaN loss detected at batch {batch_idx}! Skipping batch.")
                continue
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
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
            
            # Log progress more frequently
            if batch_idx % 25 == 0:  # Log every 25 batches
                current_acc = 100. * correct / total
                logger.info(f'Batch {batch_idx}/{len(self.train_loader)}, '
                           f'Loss: {loss.item():.4f}, '
                           f'Cls Loss: {loss_dict["classification_loss"]:.4f}, '
                           f'Acc: {current_acc:.2f}%')
        
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
                
                # Get reconstructions
                ae_reconstructed, vae_reconstructed = self.model.get_reconstructions(images)
                
                # Compute loss
                loss, loss_dict = simplified_loss(
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
    
    def save_checkpoint(self, epoch: int, is_best: bool = False, checkpoint_path: str = None):
        """Save model checkpoint"""
        if checkpoint_path is None:
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
        """Main training loop with fixed learning approach"""
        logger.info("Starting FIXED training with:")
        logger.info(f"  - Higher learning rate: {self.config['learning_rate']}")
        logger.info(f"  - Classification weight: {self.config['classification_weight']}")
        logger.info(f"  - AE weight: {self.config['ae_weight']}")
        logger.info(f"  - VAE weight: {self.config['vae_weight']}")
        logger.info(f"  - Early stopping patience: {self.config.get('early_stopping_patience', 15)}")
        
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
            logger.info(f"  Loss Components - Cls: {train_metrics['cls_loss']:.4f}, "
                       f"AE: {train_metrics['ae_loss']:.4f}, VAE: {train_metrics['vae_loss']:.4f}")
            
            # Save checkpoint
            if epoch % self.config['save_every'] == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
            
            # Early stopping check
            if self.early_stopping(val_metrics['loss'], self.model):
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break
            
            # Stop if accuracy is improving significantly
            if val_metrics['accuracy'] > 0.8:
                logger.info(f"Great accuracy achieved: {val_metrics['accuracy']:.4f}! "
                           f"Consider reducing reconstruction loss weights further.")
        
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
        'early_stopping_min_delta': args.early_stopping_min_delta,
        
        # Loss weights - FIXED VALUES
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
    parser = argparse.ArgumentParser(description='Fixed GenConViT Training - Focus on Classification')
    
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
    parser.add_argument('--lr', type=float, default=0.001,  # INCREASED DEFAULT LR
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='Weight decay for regularization (default: 1e-4)')
    
    # Early stopping arguments
    parser.add_argument('--early-stopping-patience', type=int, default=15,  # Increased patience
                       help='Early stopping patience (default: 15)')
    parser.add_argument('--early-stopping-min-delta', type=float, default=0.001,
                       help='Early stopping minimum delta (default: 0.001)')
    
    # FIXED Loss weight arguments - Better defaults
    parser.add_argument('--classification-weight', type=float, default=10.0,  # Much higher
                       help='Weight for classification loss (default: 10.0)')
    parser.add_argument('--ae-weight', type=float, default=0.01,  # Much lower
                       help='Weight for AE reconstruction loss (default: 0.01)')
    parser.add_argument('--vae-weight', type=float, default=0.01,  # Much lower
                       help='Weight for VAE loss (default: 0.01)')
    parser.add_argument('--vae-beta', type=float, default=0.1,  # Lower beta
                       help='Beta parameter for VAE KL divergence (default: 0.1)')
    
    # Checkpoint arguments
    parser.add_argument('--save-path', type=str, default='models/genconvit_v2_fixed.pth',
                       help='Path to save the model (default: models/genconvit_v2_fixed.pth)')
    parser.add_argument('--save-every', type=int, default=5,
                       help='Save checkpoint every N epochs (default: 5)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    # Other arguments
    parser.add_argument('--log-interval', type=int, default=25,  # More frequent logging
                       help='Batch interval for logging (default: 25)')
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
    
    # Log the fixed configuration
    logger.info("FIXED TRAINING CONFIGURATION:")
    logger.info(f"  Learning rate: {config['learning_rate']}")
    logger.info(f"  Classification weight: {config['classification_weight']}")
    logger.info(f"  AE weight: {config['ae_weight']}")
    logger.info(f"  VAE weight: {config['vae_weight']}")
    logger.info(f"  VAE beta: {config['vae_beta']}")
    
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
    trainer = FixedTrainer(model, train_loader, val_loader, device, config)
    
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
    logger.info("Fixed training completed successfully!")
    
    # Print final recommendations
    if results['best_val_accuracy'] > 0.7:
        logger.info("🎉 Great! Classification is working well.")
        logger.info("   You can now consider increasing reconstruction loss weights gradually.")
    elif results['best_val_accuracy'] > 0.6:
        logger.info("👍 Good progress! Classification is improving.")
        logger.info("   Continue with current settings or try even lower reconstruction weights.")
    else:
        logger.info("⚠️  Still having issues. Try:")
        logger.info("   1. Even higher classification weight (--classification-weight 20.0)")
        logger.info("   2. Even lower reconstruction weights (--ae-weight 0.001 --vae-weight 0.001)")
        logger.info("   3. Higher learning rate (--lr 0.002)")


if __name__ == '__main__':
    main()