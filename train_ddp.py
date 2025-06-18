import argparse
import os
import json
import logging
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import numpy as np
import timm

from video_dataset import VideoFrameDataset, create_video_transforms

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_logging(rank: int):
    """Setup logging for distributed training"""
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('training_ddp.log'),
                logging.StreamHandler()
            ]
        )
    else:
        logging.basicConfig(level=logging.WARNING)

# ====================
# Model Definitions
# ====================
class AutoEncoder(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        # encoder for 224x224 input
        self.enc = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),  # 224x224 -> 112x112
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),  # 112 -> 56
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),  # 56 -> 28
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256*28*28, latent_dim)  # Fixed for 224x224 input
        )
        # decoder for 224x224 output
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, 256*28*28),  # Fixed for 224x224 output
            nn.Unflatten(1, (256, 28, 28)),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 28 -> 56
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 56 -> 112
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),     # 112 -> 224
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.enc(x)
        return self.dec(z)


class VariationalAutoEncoder(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        # encoder for 224x224 input
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),   # 224 -> 112
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), # 112 -> 56
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1), # 56 -> 28
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc_mu = nn.Linear(256*28*28, latent_dim)      # Fixed for 224x224 input
        self.fc_logvar = nn.Linear(256*28*28, latent_dim)  # Fixed for 224x224 input
        # decoder for 224x224 output
        self.dec_fc = nn.Linear(latent_dim, 256*28*28)     # Fixed for 224x224 output
        self.dec_conv = nn.Sequential(
            nn.Unflatten(1, (256, 28, 28)),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 28 -> 56
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 56 -> 112
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),     # 112 -> 224
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.conv(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        out = self.dec_conv(self.dec_fc(z))
        return out, mu, logvar


class GenConViT(nn.Module):
    def __init__(self, ae_latent=256, vae_latent=256, num_classes=2):
        super().__init__()
        self.ae = AutoEncoder(ae_latent)
        self.vae = VariationalAutoEncoder(vae_latent)
        # pretrained backbones
        self.convnext = timm.create_model('convnext_tiny', pretrained=True, num_classes=0)
        self.swin = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=0)
        feat_dim = self.convnext.num_features
        # classification heads
        self.head_a = nn.Sequential(
            nn.Linear(feat_dim + ae_latent, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        self.head_b = nn.Sequential(
            nn.Linear(feat_dim + vae_latent, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # Path A: AutoEncoder + ConvNeXt + Swin
        ia = self.ae(x)
        fa1 = self.convnext(ia)
        fa2 = self.swin(ia)
        la = self.head_a(torch.cat([fa1, self.ae.enc(x)], dim=1))
        
        # Path B: VAE + ConvNeXt + Swin
        ib, mu, logvar = self.vae(x)
        fb1 = self.convnext(ib)
        fb2 = self.swin(ib)
        lb = self.head_b(torch.cat([fb1, self.vae.fc_mu(self.vae.conv(x))], dim=1))
        
        # Combined logits
        logits = la + lb
        
        return logits, la, lb, mu, logvar


def vae_loss(recon_x, x, mu, logvar):
    """VAE loss function"""
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


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
        frames_per_video=1,  # Single frame per sample
        random_frame_selection=True
    )
    
    val_dataset = VideoFrameDataset(
        root_dir=data_dir,
        split='val',
        transform=val_transform,
        frames_per_video=1,  # Single frame per sample
        random_frame_selection=False
    )
    
    return train_dataset, val_dataset


def train_epoch(model: DDP, train_loader: DataLoader, optimizer: optim.Optimizer, 
                criterion: nn.Module, device: torch.device, rank: int, beta: float = 1.0) -> Dict[str, float]:
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_vae_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (imgs, labels) in enumerate(train_loader):
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        
        # Handle video frames if needed
        if len(imgs.shape) == 5:
            batch_size, num_frames = imgs.shape[0], imgs.shape[1]
            imgs = imgs.view(-1, imgs.shape[2], imgs.shape[3], imgs.shape[4])
            labels = labels.repeat_interleave(num_frames)
        
        optimizer.zero_grad()
        
        # Forward pass
        logits, la, lb, mu, logvar = model(imgs)
        
        # Calculate losses
        loss_cls = criterion(logits, labels)
        vae_out, vae_mu, vae_logvar = model.module.vae(imgs)
        loss_vae = vae_loss(vae_out, imgs, vae_mu, vae_logvar) / imgs.size(0)  # Normalize by batch size
        loss = loss_cls + beta * loss_vae
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        total_cls_loss += loss_cls.item()
        total_vae_loss += loss_vae.item()
        
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        if rank == 0 and batch_idx % 100 == 0:
            logger.info(f'Batch {batch_idx}/{len(train_loader)}, '
                       f'Loss: {loss.item():.4f}, '
                       f'Acc: {100. * correct / total:.2f}%')
    
    return {
        'loss': total_loss / len(train_loader),
        'cls_loss': total_cls_loss / len(train_loader),
        'vae_loss': total_vae_loss / len(train_loader),
        'accuracy': correct / total
    }


def validate_epoch(model: DDP, val_loader: DataLoader, criterion: nn.Module, 
                  device: torch.device, beta: float = 1.0) -> Dict[str, float]:
    """Validate for one epoch"""
    model.eval()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_vae_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            # Handle video frames if needed
            if len(imgs.shape) == 5:
                batch_size, num_frames = imgs.shape[0], imgs.shape[1]
                imgs = imgs.view(-1, imgs.shape[2], imgs.shape[3], imgs.shape[4])
                labels = labels.repeat_interleave(num_frames)
            
            # Forward pass
            logits, la, lb, mu, logvar = model(imgs)
            
            # Calculate losses
            loss_cls = criterion(logits, labels)
            vae_out, vae_mu, vae_logvar = model.module.vae(imgs)
            loss_vae = vae_loss(vae_out, imgs, vae_mu, vae_logvar) / imgs.size(0)
            loss = loss_cls + beta * loss_vae
            
            # Statistics
            total_loss += loss.item()
            total_cls_loss += loss_cls.item()
            total_vae_loss += loss_vae.item()
            
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    return {
        'loss': total_loss / len(val_loader),
        'cls_loss': total_cls_loss / len(val_loader),
        'vae_loss': total_vae_loss / len(val_loader),
        'accuracy': correct / total
    }


def save_checkpoint(model: DDP, optimizer: optim.Optimizer, scheduler: optim.lr_scheduler._LRScheduler,
                   epoch: int, best_acc: float, save_path: str, class_names: list):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_acc': best_acc,
        'class_names': class_names
    }
    torch.save(checkpoint, save_path)


def train_ddp(rank: int, world_size: int, args):
    """Main DDP training function"""
    # Setup DDP
    setup_ddp(rank, world_size)
    setup_logging(rank)
    
    # Set device
    device = torch.device(f'cuda:{rank}')
    
    if rank == 0:
        logger.info(f"Starting distributed training on {world_size} GPUs")
        logger.info(f"Arguments: {vars(args)}")
    
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
        logger.info(f"Training samples: {len(train_dataset)}")
        logger.info(f"Validation samples: {len(val_dataset)}")
        logger.info(f"Classes: {train_dataset.class_names}")
    
    # Create model
    num_classes = len(train_dataset.class_names)
    model = GenConViT(num_classes=num_classes)
    model = model.to(device)
    
    # Wrap model with DDP
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    
    # Create optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
    criterion = nn.CrossEntropyLoss()
    
    # Load checkpoint if resuming
    start_epoch = 1
    best_acc = 0.0
    if args.resume and os.path.exists(args.resume):
        if rank == 0:
            logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']
    
    # Training loop
    for epoch in range(start_epoch, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device, rank, args.beta)
        
        # Validate
        val_metrics = validate_epoch(model, val_loader, criterion, device, args.beta)
        
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
        
        # Step scheduler
        scheduler.step()
        
        # Log and save
        if rank == 0:
            logger.info(f'Epoch {epoch}/{args.epochs}: '
                       f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                       f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, '
                       f'LR: {scheduler.get_last_lr()[0]:.6f}')
            
            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                save_checkpoint(model, optimizer, scheduler, epoch, best_acc, 
                              args.save_path, train_dataset.class_names)
                logger.info(f"New best model saved with validation accuracy: {best_acc:.4f}")
            
            # Save periodic checkpoint
            if epoch % args.save_every == 0:
                checkpoint_path = f"{args.save_path.rsplit('.', 1)[0]}_epoch_{epoch}.pth"
                save_checkpoint(model, optimizer, scheduler, epoch, best_acc, 
                              checkpoint_path, train_dataset.class_names)
    
    if rank == 0:
        logger.info(f"Training completed! Best validation accuracy: {best_acc:.4f}")
    
    cleanup_ddp()


def main():
    parser = argparse.ArgumentParser(description='GenConViT Distributed Training')
    
    # Data arguments
    parser.add_argument('--data', type=str, required=True,
                       help='Data root directory with train/val/test folders')
    parser.add_argument('--input-size', type=int, default=224,
                       help='Input image size (default: 224)')
    
    # Training arguments
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size per GPU (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate (default: 1e-4)')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                       help='Weight decay (default: 1e-5)')
    parser.add_argument('--lr-step', type=int, default=30,
                       help='Learning rate decay step (default: 30)')
    parser.add_argument('--lr-gamma', type=float, default=0.1,
                       help='Learning rate decay factor (default: 0.1)')
    parser.add_argument('--beta', type=float, default=1.0,
                       help='Weight for VAE loss (default: 1.0)')
    
    # System arguments
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers per GPU (default: 4)')
    parser.add_argument('--world-size', type=int, default=-1,
                       help='Number of GPUs to use (default: all available)')
    
    # Checkpoint arguments
    parser.add_argument('--save-path', type=str, default='genconvit_ddp_best.pth',
                       help='Path to save the best model (default: genconvit_ddp_best.pth)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--save-every', type=int, default=10,
                       help='Save checkpoint every N epochs (default: 10)')
    
    # Other arguments
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
    
    print(f"Starting distributed training on {args.world_size} GPUs")
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