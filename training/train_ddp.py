#!/usr/bin/env python3
"""
Improved GenConViT Distributed Training Script (DDP)
===================================================

This script provides enhanced distributed training for GenConViT with:
1. Fixed model architecture with proper feature fusion
2. ReduceLROnPlateau scheduler for adaptive learning rate
3. Enhanced regularization and data augmentation
4. Better loss monitoring and logging
5. Comprehensive checkpointing system
6. Efficient multi-GPU training with PyTorch DDP

Usage:
    python training/train_ddp.py --data ./data --batch-size 16 --epochs 100 --world-size 4
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
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import numpy as np

# Import our modules with new structure
import sys
sys.path.append(str(Path(__file__).parent.parent))

from models import GenConViT, combined_loss, create_genconvit
from utils import VideoFrameDataset, create_video_transforms

warnings.filterwarnings('ignore')

def setup_logging(rank: int):
    """Setup logging for distributed training"""
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('training_ddp_improved.log'),
                logging.StreamHandler()
            ]
        )
    else:
        logging.basicConfig(level=logging.WARNING)

def setup_ddp(rank: int, world_size: int, backend: str = 'nccl'):
    """Initialize DDP"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_ddp():
    """Clean up DDP"""
    dist.destroy_process_group()

def reduce_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Reduce tensor across all processes"""
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt

def create_datasets(data_dir: str, input_size: int = 224):
    """Create train and validation datasets"""
    train_transform = create_video_transforms(input_size=input_size, augment=True)
    val_transform = create_video_transforms(input_size=input_size, augment=False)

    train_dataset = VideoFrameDataset(
        root_dir=data_dir,
        split='train',
        transform=train_transform,
        frames_per_video=1,
        random_frame_selection=True
    )

    val_dataset = VideoFrameDataset(
        root_dir=data_dir,
        split='val',
        transform=val_transform,
        frames_per_video=1,
        random_frame_selection=False
    )

    return train_dataset, val_dataset

def train_epoch(model: DDP,
                train_loader: DataLoader,
                optimizer: optim.Optimizer,
                device: torch.device,
                rank: int,
                config: Dict[str, Any]) -> Dict[str, float]:
    """Train for one epoch with improved loss computation"""
    model.train()

    total_loss = 0.0
    total_cls_loss = 0.0
    total_ae_loss = 0.0
    total_vae_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        # Handle video frames if needed
        if len(images.shape) == 5:
            batch_size, num_frames = images.shape[0], images.shape[1]
            images = images.view(-1, images.shape[2], images.shape[3], images.shape[4])
            labels = labels.repeat_interleave(num_frames)

        optimizer.zero_grad()

        # Forward pass
        logits, logits_a, logits_b, mu, logvar = model(images)

        # Get reconstructions for loss computation
        ae_reconstructed, vae_reconstructed = model.module.get_reconstructions(images)

        # IMPROVED: Use combined loss function
        loss, loss_dict = combined_loss(
            logits=logits,
            targets=labels,
            ae_reconstructed=ae_reconstructed,
            vae_reconstructed=vae_reconstructed,
            original=images,
            mu=mu,
            logvar=logvar,
            classification_weight=config['classification_weight'],
            ae_weight=config['ae_weight'],
            vae_weight=config['vae_weight'],
            vae_beta=config['vae_beta']
        )

        # Backward pass
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Statistics
        total_loss += loss.item()
        total_cls_loss += loss_dict['classification_loss']
        total_ae_loss += loss_dict['ae_reconstruction_loss']
        total_vae_loss += loss_dict['vae_loss']

        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if rank == 0 and batch_idx % config['log_interval'] == 0:
            logging.info(f'Batch {batch_idx}/{len(train_loader)}, '
                        f'Loss: {loss.item():.4f}, '
                        f'Acc: {100. * correct / total:.2f}%')

    return {
        'loss': total_loss / len(train_loader),
        'accuracy': correct / total,
        'cls_loss': total_cls_loss / len(train_loader),
        'ae_loss': total_ae_loss / len(train_loader),
        'vae_loss': total_vae_loss / len(train_loader)
    }

def validate_epoch(model: DDP,
                   val_loader: DataLoader,
                   device: torch.device,
                   config: Dict[str, Any]) -> Dict[str, float]:
    """Validate for one epoch with improved loss computation"""
    model.eval()

    total_loss = 0.0
    total_cls_loss = 0.0
    total_ae_loss = 0.0
    total_vae_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            # Handle video frames if needed
            if len(images.shape) == 5:
                batch_size, num_frames = images.shape[0], images.shape[1]
                images = images.view(-1, images.shape[2], images.shape[3], images.shape[4])
                labels = labels.repeat_interleave(num_frames)

            # Forward pass
            logits, logits_a, logits_b, mu, logvar = model(images)

            # Get reconstructions for loss computation
            ae_reconstructed, vae_reconstructed = model.module.get_reconstructions(images)

            # Compute loss
            loss, loss_dict = combined_loss(
                logits=logits,
                targets=labels,
                ae_reconstructed=ae_reconstructed,
                vae_reconstructed=vae_reconstructed,
                original=images,
                mu=mu,
                logvar=logvar,
                classification_weight=config['classification_weight'],
                ae_weight=config['ae_weight'],
                vae_weight=config['vae_weight'],
                vae_beta=config['vae_beta']
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
        'loss': total_loss / len(val_loader),
        'accuracy': correct / total,
        'cls_loss': total_cls_loss / len(val_loader),
        'ae_loss': total_ae_loss / len(val_loader),
        'vae_loss': total_vae_loss / len(val_loader)
    }

def save_checkpoint(model: DDP,
                    optimizer: optim.Optimizer,
                    scheduler: optim.lr_scheduler._LRScheduler,
                    epoch: int,
                    best_acc: float,
                    save_path: str,
                    class_names: list,
                    config: Dict[str, Any],
                    history: Dict[str, list]):
    """Save model checkpoint with complete state"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_acc': best_acc,
        'class_names': class_names,
        'config': config,
        'history': history
    }
    torch.save(checkpoint, save_path)

def train_ddp(rank: int, world_size: int, args):
    """Main DDP training function with all improvements"""
    # Setup DDP
    setup_ddp(rank, world_size)
    setup_logging(rank)

    # Set device
    device = torch.device(f'cuda:{rank}')

    if rank == 0:
        logging.info(f"Starting improved distributed training on {world_size} GPUs")
        logging.info(f"Arguments: {vars(args)}")

    # Create configuration
    config = {
        'classification_weight': args.classification_weight,
        'ae_weight': args.ae_weight,
        'vae_weight': args.vae_weight,
        'vae_beta': args.vae_beta,
        'log_interval': args.log_interval,
        'lr_patience': args.lr_patience,
        'min_lr': args.min_lr
    }

    # Create datasets
    train_dataset, val_dataset = create_datasets(args.data, args.input_size)

    # Create distributed samplers
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    if rank == 0:
        logging.info(f"Training samples: {len(train_dataset)}")
        logging.info(f"Validation samples: {len(val_dataset)}")
        logging.info(f"Classes: {train_dataset.class_names}")

    # Create IMPROVED model with fixed architecture
    num_classes = len(train_dataset.class_names)
    model = create_genconvit(
        num_classes=num_classes,
        dropout_rate=args.dropout_rate,
        ae_latent=args.ae_latent,
        vae_latent=args.vae_latent
    )
    model = model.to(device)

    # Wrap model with DDP
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)

    # Create optimizer with weight decay
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    # IMPROVED: ReduceLROnPlateau scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=args.lr_gamma,
        patience=args.lr_patience,
        min_lr=args.min_lr,
        verbose=(rank == 0)
    )

    # Load checkpoint if resuming
    start_epoch = 1
    best_acc = 0.0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rates': []
    }

    if args.resume and os.path.exists(args.resume):
        if rank == 0:
            logging.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']
        if 'history' in checkpoint:
            history = checkpoint['history']

    # Training loop with improvements
    for epoch in range(start_epoch, args.epochs + 1):
        train_sampler.set_epoch(epoch)

        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device, rank, config)

        # Validate
        val_metrics = validate_epoch(model, val_loader, device, config)

        # Reduce metrics across all processes
        if world_size > 1:
            train_loss = reduce_tensor(torch.tensor(train_metrics['loss']).to(device)).item()
            train_acc = reduce_tensor(torch.tensor(train_metrics['accuracy']).to(device)).item()
            val_loss = reduce_tensor(torch.tensor(val_metrics['loss']).to(device)).item()
            val_acc = reduce_tensor(torch.tensor(val_metrics['accuracy']).to(device)).item()
        else:
            train_loss = train_metrics['loss']
            train_acc = train_metrics['accuracy']
            val_loss = val_metrics['loss']
            val_acc = val_metrics['accuracy']

        # Update learning rate scheduler
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['learning_rates'].append(current_lr)

        # Log and save
        if rank == 0:
            logging.info(f'Epoch {epoch}/{args.epochs}:')
            logging.info(f'  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}')
            logging.info(f'  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}')
            logging.info(f'  LR: {current_lr:.6f}')

            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                # Save best model with descriptive name
                best_path = args.save_path.replace('.pth', '_best.pth')
                save_checkpoint(model, optimizer, scheduler, epoch, best_acc,
                              best_path, train_dataset.class_names, config, history)

                # Also save with timestamp for easier identification
                import datetime
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                timestamped_path = args.save_path.replace('.pth', f'_best_{timestamp}.pth')
                save_checkpoint(model, optimizer, scheduler, epoch, best_acc,
                              timestamped_path, train_dataset.class_names, config, history)

                logging.info(f"New best model saved: {best_path}")
                logging.info(f"Timestamped best model saved: {timestamped_path}")
            
            # Save periodic checkpoint
            if epoch % args.save_every == 0:
                checkpoint_path = f"models/genconvit_v2_ddp_epoch_{epoch}.pth"
                save_checkpoint(model, optimizer, scheduler, epoch, best_acc, 
                              checkpoint_path, train_dataset.class_names, config, history)
                logging.info(f"Periodic checkpoint saved: {checkpoint_path}")

    if rank == 0:
        logging.info(f"Training completed! Best validation accuracy: {best_acc:.4f}")

        # Save final results
        results = {
            'best_val_accuracy': best_acc,
            'total_epochs': args.epochs,
            'history': history,
            'config': config
        }

        results_path = args.save_path.replace('.pth', '_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        logging.info(f"Results saved to: {results_path}")

    cleanup_ddp()

def main():
    parser = argparse.ArgumentParser(description='Improved GenConViT Distributed Training')

    # Data arguments
    parser.add_argument('--data', type=str, required=True,
                       help='Data root directory with train/val/test folders')
    parser.add_argument('--input-size', type=int, default=224,
                       help='Input image size (default: 224)')

    # Training arguments
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size per GPU (default: 16)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate (default: 1e-4)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='Weight decay (default: 1e-4)')

    # IMPROVED: ReduceLROnPlateau scheduler arguments
    parser.add_argument('--lr-gamma', type=float, default=0.5,
                       help='Learning rate reduction factor (default: 0.5)')
    parser.add_argument('--lr-patience', type=int, default=5,
                       help='Scheduler patience in epochs (default: 5)')
    parser.add_argument('--min-lr', type=float, default=1e-7,
                       help='Minimum learning rate (default: 1e-7)')

    # Model arguments
    parser.add_argument('--ae-latent', type=int, default=256,
                       help='AutoEncoder latent dimension (default: 256)')
    parser.add_argument('--vae-latent', type=int, default=256,
                       help='VAE latent dimension (default: 256)')
    parser.add_argument('--dropout-rate', type=float, default=0.5,
                       help='Dropout rate (default: 0.5)')

    # Loss weight arguments
    parser.add_argument('--classification-weight', type=float, default=1.0,
                       help='Weight for classification loss (default: 1.0)')
    parser.add_argument('--ae-weight', type=float, default=0.1,
                       help='Weight for AE reconstruction loss (default: 0.1)')
    parser.add_argument('--vae-weight', type=float, default=0.1,
                       help='Weight for VAE loss (default: 0.1)')
    parser.add_argument('--vae-beta', type=float, default=1.0,
                       help='Beta parameter for VAE KL divergence (default: 1.0)')

    # System arguments
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers per GPU (default: 4)')
    parser.add_argument('--world-size', type=int, default=-1,
                       help='Number of GPUs to use (default: all available)')

    # Checkpoint arguments
    parser.add_argument('--save-path', type=str, default='models/genconvit_v2_ddp.pth',
                       help='Path to save the best model (default: models/genconvit_v2_ddp.pth)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--save-every', type=int, default=5,
                       help='Save checkpoint every N epochs (default: 5)')

    # Other arguments
    parser.add_argument('--log-interval', type=int, default=50,
                       help='Batch interval for logging (default: 50)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')

    args = parser.parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Determine world size
    if args.world_size == -1:
        args.world_size = torch.cuda.device_count()

    if args.world_size == 0:
        raise RuntimeError("No CUDA devices available!")

    print(f"Starting improved distributed training on {args.world_size} GPUs")
    print(f"Total batch size: {args.batch_size * args.world_size}")

    # Create output directory
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    # Launch distributed training
    if args.world_size == 1:
        # Single GPU training
        train_ddp(0, 1, args)
    else:
        # Multi-GPU training
        mp.spawn(train_ddp, args=(args.world_size, args), nprocs=args.world_size, join=True)

if __name__ == '__main__':
    main()
