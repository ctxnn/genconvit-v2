import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader\ Pluginsestion
from torchvision import transforms, datasets
import timm

# ====================
# Model Definitions
# ====================
class AutoEncoder(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        # encoder
        self.enc = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),  # 128x128 -> 64x64
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),  # 64 -> 32
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256*32*32, latent_dim)
        )
        # decoder
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, 256*32*32),
            nn.Unflatten(1, (256, 32, 32)),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 32 -> 64
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.enc(x)
        return self.dec(z)


class VariationalAutoEncoder(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        # encoder
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc_mu = nn.Linear(256*32*32, latent_dim)
        self.fc_logvar = nn.Linear(256*32*32, latent_dim)
        # decoder
        self.dec_fc = nn.Linear(latent_dim, 256*32*32)
        self.dec_conv = nn.Sequential(
            nn.Unflatten(1, (256, 32, 32)),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
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
    # data
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    train_ds = datasets.ImageFolder(os.path.join(args.data, 'train'), transform)
    val_ds = datasets.ImageFolder(os.path.join(args.data, 'val'), transform)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = GenConViT().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(1, args.epochs+1):
        model.train()
        total_loss = 0.0
        total = 0
        correct = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits, la, lb, mu, logvar = model(imgs)
            loss_cls = criterion(logits, labels)
            loss_vae = vae_loss( model.vae(imgs)[0], imgs, mu, logvar)
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
        print(f"Epoch {epoch}: Loss {avg_loss:.4f}, Train Acc {train_acc:.4f}, Val Acc {val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), args.save_path)
    print(f"Best Val Acc: {best_acc:.4f}")


def evaluate(model, loader, device):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits, *_ = model(imgs)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total


# ====================
# CLI
# ====================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='data root with train/val folders')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--beta', type=float, default=1.0, help='weight for VAE loss')
    parser.add_argument('--save-path', type=str, default='genconvit.pth')
    parser.add_argument('--mode', choices=['train', 'eval'], default='train')
    args = parser.parse_args()
    if args.mode == 'train':
        train(args)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = GenConViT().to(device)
        model.load_state_dict(torch.load(args.save_path, map_location=device))
        # assume eval uses val split
        transform = transforms.Compose([
            transforms.Resize((128,128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
        val_ds = datasets.ImageFolder(os.path.join(args.data, 'val'), transform)
        loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=4)
        acc = evaluate(model, loader, device)
        print(f"Evaluation Accuracy: {acc:.4f}")
