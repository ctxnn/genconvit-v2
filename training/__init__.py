"""
GenConViT Training Module
========================

This module contains training scripts and utilities for the GenConViT deepfake detection system.

Scripts:
    train_improved: Enhanced single-GPU training with ReduceLROnPlateau and early stopping
    train_ddp: Improved distributed training for multi-GPU setups
    train_ddp_old: Legacy DDP training script (for reference)
"""

from .train_improved import ImprovedTrainer, EarlyStopping

__all__ = [
    'ImprovedTrainer',
    'EarlyStopping'
]