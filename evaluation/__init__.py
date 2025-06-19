"""
GenConViT Evaluation Module
===========================

This module contains evaluation scripts and utilities for testing trained GenConViT models
on video datasets and generating comprehensive evaluation reports.

Scripts:
    predict_and_evaluate: Main evaluation script for video-based deepfake detection
    evaluate_model: Model evaluation utilities
"""

from .predict_and_evaluate import (
    VideoFrameExtractor,
    ModelPredictor,
    EvaluationReporter
)

__all__ = [
    'VideoFrameExtractor',
    'ModelPredictor', 
    'EvaluationReporter'
]