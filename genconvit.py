import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import timm

from video_dataset import VideoFrameDataModule, create_video_transforms, analyze_dataset

# ====================
# Model Definitions
# ====================
class AutoEncoder(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        # encoder for 128x128 input
        self.enc = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),  # 128x128 -> 64x64
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),  # 64 -> 32
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),  # 32 -> 16
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256*16*16, latent_dim)  # Fixed for 128x128 input
        )
        # decoder for 128x128 output
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, 256*16*16),  # Fixed for 128x128 output
            nn.Unflatten(1, (256, 16, 16)),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 16 -> 32
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 32 -> 64
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),     # 64 -> 128
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.enc(x)
        return self.dec(z)


class VariationalAutoEncoder(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        # encoder for 128x128 input
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),   # 128 -> 64
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), # 64 -> 32
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1), # 32 -> 16
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc_mu = nn.Linear(256*16*16, latent_dim)      # Fixed for 128x128 input
        self.fc_logvar = nn.Linear(256*16*16, latent_dim)  # Fixed for 128x128 input
        # decoder for 128x128 output
        self.dec_fc = nn.Linear(latent_dim, 256*16*16)     # Fixed for 128x128 output
        self.dec_conv = nn.Sequential(
            nn.Unflatten(1, (256, 16, 16)),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 16 -> 32
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 32 -> 64
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),     # 64 -> 128
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
        self.fc_a = nn.Sequential(
            nn.GELU(),
            nn.Linear(feat_dim, num_classes)
        )
        self.fc_b = nn.Sequential(
            nn.ReLU(),
            nn.Linear(feat_dim, num_classes)
        )

    def forward(self, x):
        # path A: AE
        ia = self.ae(x)
        fa1 = self.convnext(ia)
        fa2 = self.swin(ia)
        la = self.fc_a(fa1 + fa2)
        # path B: VAE
        ib, mu, logvar = self.vae(x)
        fb1 = self.convnext(ib)
        fb2 = self.swin(ib)
        lb = self.fc_b(fb1 + fb2)
        # fused logits
        logits = la + lb
        return logits, la, lb, mu, logvar

# ====================
# Loss and Metrics
# ====================
def vae_loss(recon_x, x, mu, logvar):
    bce = nn.functional.mse_loss(recon_x, x, reduction='mean')
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return bce + kld

# ====================
# Training & Evaluation
# ====================
def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data module with video frame support
    train_transform = create_video_transforms(input_size=128, augment=True)
    val_transform = create_video_transforms(input_size=128, augment=False)
    
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
        print(f"Training dataset: {train_stats['total_samples']} samples, {train_stats['num_classes']} classes")
        print(f"Class distribution: {train_stats['class_distribution']}")
        print(f"Average frames per video: {train_stats['avg_frames_per_video']:.2f}")
    
    if data_module.val_dataset:
        val_stats = analyze_dataset(data_module.val_dataset)
        print(f"Validation dataset: {val_stats['total_samples']} samples")

    # Get number of classes from data module
    num_classes = data_module.get_num_classes()
    print(f"Number of classes: {num_classes}")
    
    model = GenConViT(num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)

    best_acc = 0.0
    for epoch in range(1, args.epochs+1):
        model.train()
        total_loss = 0.0
        total = 0
        correct = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            
            # Handle video frames: reshape from [batch, frames, channels, height, width] to [batch*frames, channels, height, width]
            if len(imgs.shape) == 5:
                batch_size, num_frames = imgs.shape[0], imgs.shape[1]
                imgs = imgs.view(-1, imgs.shape[2], imgs.shape[3], imgs.shape[4])
                # Repeat labels for each frame
                labels = labels.repeat_interleave(num_frames)
            
            optimizer.zero_grad()
            logits, la, lb, mu, logvar = model(imgs)
            loss_cls = criterion(logits, labels)
            vae_out, vae_mu, vae_logvar = model.vae(imgs)
            loss_vae = vae_loss(vae_out, imgs, vae_mu, vae_logvar)
            loss = loss_cls + args.beta * loss_vae
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
        train_acc = correct / total
        avg_loss = total_loss / total
        val_acc = evaluate(model, val_loader, device)
        
        # Step scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        print(f"Epoch {epoch}: Loss {avg_loss:.4f}, Train Acc {train_acc:.4f}, Val Acc {val_acc:.4f}, LR {current_lr:.6f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_acc': best_acc,
                'class_names': data_module.get_class_names()
            }, args.save_path)
            print(f"New best model saved with validation accuracy: {best_acc:.4f}")
    
    print(f"Training completed! Best Val Acc: {best_acc:.4f}")


def evaluate(model, loader, device):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            
            # Handle video frames: reshape from [batch, frames, channels, height, width] to [batch*frames, channels, height, width]
            if len(imgs.shape) == 5:
                batch_size, num_frames = imgs.shape[0], imgs.shape[1]
                imgs = imgs.view(-1, imgs.shape[2], imgs.shape[3], imgs.shape[4])
                # Repeat labels for each frame
                labels = labels.repeat_interleave(num_frames)
            
            logits, *_ = model(imgs)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total


# ====================
# CLI
# ====================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GenConViT Training for Video Frame DeepFake Detection')
    parser.add_argument('--data', type=str, required=True, 
                       help='Data root directory with train/val/test folders or extracted frames')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs (default: 20)')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate (default: 1e-4)')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                       help='Weight decay for regularization (default: 1e-5)')
    parser.add_argument('--lr-step', type=int, default=7,
                       help='Learning rate decay step size (default: 7)')
    parser.add_argument('--lr-gamma', type=float, default=0.5,
                       help='Learning rate decay factor (default: 0.5)')
    parser.add_argument('--beta', type=float, default=1.0, 
                       help='Weight for VAE loss (default: 1.0)')
    parser.add_argument('--save-path', type=str, default='genconvit_best.pth',
                       help='Path to save the best model (default: genconvit_best.pth)')
    parser.add_argument('--mode', choices=['train', 'eval'], default='train',
                       help='Mode: train or eval (default: train)')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers (default: 4)')
    parser.add_argument('--frames-per-video', type=int, default=None,
                       help='Number of frames to sample per video (default: None - use all)')
    parser.add_argument('--balanced-sampling', action='store_true',
                       help='Use balanced sampling for imbalanced datasets')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train(args)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model checkpoint
        checkpoint = torch.load(args.save_path, map_location=device)
        
        # Create data module for evaluation
        val_transform = create_video_transforms(input_size=128, augment=False)
        data_module = VideoFrameDataModule(
            data_dir=args.data,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            val_transform=val_transform,
            frames_per_video=args.frames_per_video
        )
        
        data_module.setup('test')
        
        # Use test data if available, otherwise use validation data
        if data_module.test_dataset:
            loader = data_module.test_dataloader()
            print("Using test dataset for evaluation")
        else:
            data_module.setup('fit')  # Setup validation data
            loader = data_module.val_dataloader()
            print("Using validation dataset for evaluation")
        
        # Create model with correct number of classes
        num_classes = len(checkpoint.get('class_names', ['fake', 'real']))
        model = GenConViT(num_classes=num_classes).to(device)
        
        # Load model weights
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
        else:
            model.load_state_dict(checkpoint)
        
        acc = evaluate(model, loader, device)
        print(f"Evaluation Accuracy: {acc:.4f}")
