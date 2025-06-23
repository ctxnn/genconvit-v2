"""
GenConViT Models Module
======================

This module contains the improved GenConViT architecture with fixes for:
1. Proper feature fusion between ConvNeXt and Swin Transformer
2. Enhanced regularization with dropout
3. Better loss computation
4. Modular design for easy experimentation

Classes:
    AutoEncoder: Autoencoder component for reconstruction path
    VariationalAutoEncoder: VAE component with reparameterization
    GenConViT: Main model combining AE, VAE, and dual transformer backbones
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Tuple, Optional


class AutoEncoder(nn.Module):
    """
    Autoencoder for learning compressed representations.
    Fixed for 224x224 input size with proper dimension calculations.
    """

    def __init__(self, latent_dim: int = 256, input_size: int = 224):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_size = input_size

        # Calculate the feature map size after convolutions
        # 224 -> 112 -> 56 -> 28
        self.feature_size = input_size // 8
        self.flat_features = 256 * self.feature_size * self.feature_size

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),  # 224x224 -> 112x112
            #nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),  # 112x112 -> 56x56
            #nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),  # 56x56 -> 28x28
            #nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(self.flat_features, latent_dim),
            nn.ReLU(inplace=True)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, self.flat_features),
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (256, self.feature_size, self.feature_size)),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 28x28 -> 56x56
            #nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 56x56 -> 112x112
            #nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),     # 112x112 -> 224x224
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Get latent representation"""
        return self.encoder(x)


class VariationalAutoEncoder(nn.Module):
    """
    Variational Autoencoder with proper reparameterization trick.
    Enhanced with batch normalization for better training stability.
    """

    def __init__(self, latent_dim: int = 256, input_size: int = 224):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_size = input_size

        # Calculate the feature map size after convolutions
        self.feature_size = input_size // 8
        self.flat_features = 256 * self.feature_size * self.feature_size

        # Encoder (shared part)
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),   # 224 -> 112
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1), # 112 -> 56
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1), # 56 -> 28
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Flatten()
        )

        # Mean and log variance layers
        self.fc_mu = nn.Linear(self.flat_features, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_features, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, self.flat_features),
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (256, self.feature_size, self.feature_size)),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 28 -> 56
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 56 -> 112
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),     # 112 -> 224
            nn.Sigmoid()
        )

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for VAE"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning reconstruction, mu, and logvar"""
        h = self.encoder_conv(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decoder(z)
        return reconstructed, mu, logvar

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get mean and log variance"""
        h = self.encoder_conv(x)
        return self.fc_mu(h), self.fc_logvar(h)


class GenConViT(nn.Module):
    """
    FIXED GenConViT Architecture with proper feature fusion.

    This model combines:
    1. AutoEncoder path with ConvNeXt + Swin Transformer feature fusion
    2. VAE path with ConvNeXt + Swin Transformer feature fusion
    3. Enhanced regularization with dropout
    4. Proper loss computation combining classification and reconstruction

    Key fixes:
    - Swin Transformer features are now properly used (was major bug before)
    - Added dropout for better regularization
    - Improved feature fusion strategy
    - Better initialization and architecture design
    """

    def __init__(self,
                 ae_latent: int = 256,
                 vae_latent: int = 256,
                 num_classes: int = 2,
                 dropout_rate: float = 0.5,
                 input_size: int = 224,
                 pretrained: bool = True):
        super().__init__()

        self.ae_latent = ae_latent
        self.vae_latent = vae_latent
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        # Autoencoder and VAE components
        self.ae = AutoEncoder(ae_latent, input_size)
        self.vae = VariationalAutoEncoder(vae_latent, input_size)

        # Pretrained backbone networks
        self.convnext = timm.create_model('convnext_tiny', pretrained=pretrained, num_classes=0)
        self.swin = timm.create_model('swin_tiny_patch4_window7_224', pretrained=pretrained, num_classes=0)

        # Get feature dimensions
        backbone_dim = self.convnext.num_features

        # FIXED: Classification heads that properly use fused features
        # Path A: Uses fused ConvNeXt + Swin features from AE reconstructed images
        self.classifier_a = nn.Sequential(
            nn.Linear(backbone_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            #nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )

        # Path B: Uses fused ConvNeXt + Swin features from VAE reconstructed images
        self.classifier_b = nn.Sequential(
            nn.Linear(backbone_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            #nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )

        # Feature fusion layer for combining different modalities
        self.fusion_layer = nn.Sequential(
            nn.Linear(backbone_dim * 2, backbone_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5)  # Lower dropout for fusion
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights for better training stability"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        FIXED Forward pass with proper feature fusion.

        Returns:
            logits: Combined classification logits
            logits_a: Path A classification logits
            logits_b: Path B classification logits
            mu: VAE mean
            logvar: VAE log variance
        """
        batch_size = x.size(0)

        # Path A: AutoEncoder reconstruction + Feature extraction
        ae_reconstructed = self.ae(x)

        # Extract features from both backbones using AE reconstructed images
        convnext_features_a = self.convnext(ae_reconstructed)
        swin_features_a = self.swin(ae_reconstructed)

        # FIXED: Properly fuse features (this was the major bug!)
        fused_features_a = convnext_features_a + swin_features_a  # Element-wise addition
        logits_a = self.classifier_a(fused_features_a)

        # Path B: VAE reconstruction + Feature extraction
        vae_reconstructed, mu, logvar = self.vae(x)

        # Extract features from both backbones using VAE reconstructed images
        convnext_features_b = self.convnext(vae_reconstructed)
        swin_features_b = self.swin(vae_reconstructed)

        # FIXED: Properly fuse features (this was the major bug!)
        fused_features_b = convnext_features_b + swin_features_b  # Element-wise addition
        logits_b = self.classifier_b(fused_features_b)

        # Final logits combination
        logits = logits_a + logits_b

        return logits, logits_a, logits_b, mu, logvar

    def get_reconstructions(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get reconstructed images from both paths"""
        ae_reconstructed = self.ae(x)
        vae_reconstructed, _, _ = self.vae(x)
        return ae_reconstructed, vae_reconstructed

    def get_latent_representations(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get latent representations from both paths"""
        ae_latent = self.ae.encode(x)
        vae_mu, vae_logvar = self.vae.encode(x)
        return ae_latent, vae_mu, vae_logvar


# Loss functions
def vae_loss(reconstructed: torch.Tensor,
             original: torch.Tensor,
             mu: torch.Tensor,
             logvar: torch.Tensor,
             beta: float = 1.0) -> torch.Tensor:
    """
    Improved VAE loss with configurable beta for KL divergence weighting.

    Args:
        reconstructed: Reconstructed images
        original: Original images
        mu: Mean from encoder
        logvar: Log variance from encoder
        beta: Weight for KL divergence term

    Returns:
        Combined VAE loss
    """
    # Reconstruction loss (MSE works better than BCE for continuous outputs)
    recon_loss = F.mse_loss(reconstructed, original, reduction='mean')

    # KL divergence loss
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + beta * kl_loss


def ae_loss(reconstructed: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
    """Simple autoencoder reconstruction loss"""
    return F.mse_loss(reconstructed, original, reduction='mean')


def combined_loss(logits: torch.Tensor,
                  targets: torch.Tensor,
                  ae_reconstructed: torch.Tensor,
                  vae_reconstructed: torch.Tensor,
                  original: torch.Tensor,
                  mu: torch.Tensor,
                  logvar: torch.Tensor,
                  classification_weight: float = 1.0,
                  ae_weight: float = 0.1,
                  vae_weight: float = 0.1,
                  vae_beta: float = 1.0) -> Tuple[torch.Tensor, dict]:
    """
    Combined loss function for the complete GenConViT model.

    Args:
        logits: Classification logits
        targets: True class labels
        ae_reconstructed: Autoencoder reconstructed images
        vae_reconstructed: VAE reconstructed images
        original: Original input images
        mu: VAE encoder mean
        logvar: VAE encoder log variance
        classification_weight: Weight for classification loss
        ae_weight: Weight for AE reconstruction loss
        vae_weight: Weight for VAE loss
        vae_beta: Beta parameter for VAE KL divergence

    Returns:
        total_loss: Combined weighted loss
        loss_dict: Dictionary with individual loss components
    """
    # Classification loss
    cls_loss = F.cross_entropy(logits, targets)

    # Reconstruction losses
    ae_recon_loss = ae_loss(ae_reconstructed, original)
    vae_total_loss = vae_loss(vae_reconstructed, original, mu, logvar, vae_beta)

    # Combined loss
    total_loss = (classification_weight * cls_loss +
                  ae_weight * ae_recon_loss +
                  vae_weight * vae_total_loss)

    # Loss dictionary for monitoring
    loss_dict = {
        'total_loss': total_loss.item(),
        'classification_loss': cls_loss.item(),
        'ae_reconstruction_loss': ae_recon_loss.item(),
        'vae_loss': vae_total_loss.item()
    }

    return total_loss, loss_dict


# Model factory functions
def create_genconvit(num_classes: int = 2,
                     pretrained: bool = True,
                     dropout_rate: float = 0.5,
                     **kwargs) -> GenConViT:
    """Factory function to create GenConViT model with sensible defaults"""
    return GenConViT(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout_rate=dropout_rate,
        **kwargs
    )


def load_genconvit_from_checkpoint(checkpoint_path: str,
                                   num_classes: int = 2,
                                   map_location: str = 'cpu',
                                   **model_kwargs) -> GenConViT:
    """Load GenConViT model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=map_location)

    # Handle different checkpoint formats
    if 'class_names' in checkpoint:
        num_classes = len(checkpoint['class_names'])
    elif 'num_classes' in checkpoint:
        num_classes = checkpoint['num_classes']

    model = create_genconvit(num_classes=num_classes, **model_kwargs)

    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    return model
