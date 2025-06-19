#!/usr/bin/env python3
"""
GenConViT Video Prediction and Evaluation Script

This script can:
1. Extract frames from videos
2. Run predictions on extracted frames
3. Generate comprehensive evaluation reports
4. Create visualization graphs and charts
5. Handle both individual videos and batch processing

Usage:
    python predict_and_evaluate.py --model path/to/model.pth --video-dir path/to/videos --output-dir results
"""

import argparse
import os
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score
)
import timm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Model definitions (same as training script)
class AutoEncoder(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256*28*28, latent_dim)
        )
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, 256*28*28),
            nn.Unflatten(1, (256, 28, 28)),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
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
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc_mu = nn.Linear(256*28*28, latent_dim)
        self.fc_logvar = nn.Linear(256*28*28, latent_dim)
        self.dec_fc = nn.Linear(latent_dim, 256*28*28)
        self.dec_conv = nn.Sequential(
            nn.Unflatten(1, (256, 28, 28)),
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
        self.convnext = timm.create_model('convnext_tiny', pretrained=True, num_classes=0)
        self.swin = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=0)
        feat_dim = self.convnext.num_features
        
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
        ia = self.ae(x)
        fa1 = self.convnext(ia)
        fa2 = self.swin(ia)
        la = self.head_a(torch.cat([fa1, self.ae.enc(x)], dim=1))
        
        ib, mu, logvar = self.vae(x)
        fb1 = self.convnext(ib)
        fb2 = self.swin(ib)
        lb = self.head_b(torch.cat([fb1, self.vae.fc_mu(self.vae.conv(x))], dim=1))
        
        logits = la + lb
        return logits, la, lb, mu, logvar

class VideoFrameExtractor:
    """Extract frames from video files"""
    
    def __init__(self, frame_interval: int = 5, max_frames: Optional[int] = None, target_size: Tuple[int, int] = (224, 224)):
        self.frame_interval = frame_interval
        self.max_frames = max_frames
        self.target_size = target_size
        
    def extract_frames_from_video(self, video_path: str, output_dir: str) -> List[str]:
        """Extract frames from a single video"""
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return []
        
        frame_paths = []
        frame_count = 0
        extracted_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % self.frame_interval == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Resize frame
                frame_resized = cv2.resize(frame_rgb, self.target_size)
                
                # Save frame
                frame_filename = f"{video_path.stem}_frame_{extracted_count:06d}.png"
                frame_path = output_dir / frame_filename
                
                # Convert to PIL and save
                pil_image = Image.fromarray(frame_resized)
                pil_image.save(frame_path)
                
                frame_paths.append(str(frame_path))
                extracted_count += 1
                
                if self.max_frames and extracted_count >= self.max_frames:
                    break
                    
            frame_count += 1
        
        cap.release()
        logger.info(f"Extracted {extracted_count} frames from {video_path.name}")
        return frame_paths

class FrameDataset(Dataset):
    """Dataset for loading extracted frames"""
    
    def __init__(self, frame_paths: List[str], transform=None):
        self.frame_paths = frame_paths
        self.transform = transform
    
    def __len__(self):
        return len(self.frame_paths)
    
    def __getitem__(self, idx):
        frame_path = self.frame_paths[idx]
        image = Image.open(frame_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, frame_path

class ModelPredictor:
    """Handle model loading and predictions"""
    
    def __init__(self, model_path: str, device: str = 'auto'):
        self.device = self._setup_device(device)
        self.model = self._load_model(model_path)
        self.transform = self._create_transform()
        
    def _setup_device(self, device: str) -> torch.device:
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device)
    
    def _load_model(self, model_path: str) -> nn.Module:
        logger.info(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Determine number of classes
        if 'class_names' in checkpoint:
            num_classes = len(checkpoint['class_names'])
            self.class_names = checkpoint['class_names']
        else:
            num_classes = 2
            self.class_names = ['fake', 'real']
        
        # Create and load model
        model = GenConViT(num_classes=num_classes)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        model = model.to(self.device)
        model.eval()
        
        logger.info(f"Model loaded successfully on {self.device}")
        logger.info(f"Classes: {self.class_names}")
        
        return model
    
    def _create_transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def predict_frames(self, frame_paths: List[str], batch_size: int = 32) -> Dict[str, Any]:
        """Predict on a list of frame paths"""
        dataset = FrameDataset(frame_paths, self.transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        all_predictions = []
        all_probabilities = []
        frame_paths_ordered = []
        
        with torch.no_grad():
            for images, paths in dataloader:
                images = images.to(self.device)
                
                logits, _, _, _, _ = self.model(images)
                probabilities = F.softmax(logits, dim=1)
                predictions = torch.argmax(logits, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                frame_paths_ordered.extend(paths)
        
        return {
            'frame_paths': frame_paths_ordered,
            'predictions': np.array(all_predictions),
            'probabilities': np.array(all_probabilities),
            'class_names': self.class_names
        }
    
    def predict_video(self, video_path: str, extractor: VideoFrameExtractor, temp_dir: str) -> Dict[str, Any]:
        """Predict on a single video"""
        video_name = Path(video_path).stem
        frame_output_dir = Path(temp_dir) / video_name
        
        # Extract frames
        frame_paths = extractor.extract_frames_from_video(video_path, frame_output_dir)
        
        if not frame_paths:
            return {
                'video_path': video_path,
                'video_name': video_name,
                'error': 'No frames extracted',
                'frame_count': 0
            }
        
        # Predict on frames
        frame_results = self.predict_frames(frame_paths)
        
        # Aggregate video-level prediction
        probabilities = frame_results['probabilities']
        avg_probabilities = np.mean(probabilities, axis=0)
        video_prediction = np.argmax(avg_probabilities)
        confidence = np.max(avg_probabilities)
        
        return {
            'video_path': video_path,
            'video_name': video_name,
            'frame_count': len(frame_paths),
            'frame_predictions': frame_results['predictions'],
            'frame_probabilities': probabilities,
            'video_prediction': video_prediction,
            'video_probabilities': avg_probabilities,
            'confidence': confidence,
            'predicted_class': self.class_names[video_prediction]
        }

class EvaluationReporter:
    """Generate evaluation reports and visualizations"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set matplotlib style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def generate_video_report(self, results: List[Dict[str, Any]], ground_truth: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
        """Generate comprehensive video-level evaluation report"""
        
        # Collect predictions
        video_names = [r['video_name'] for r in results if 'error' not in r]
        predictions = [r['video_prediction'] for r in results if 'error' not in r]
        confidences = [r['confidence'] for r in results if 'error' not in r]
        predicted_classes = [r['predicted_class'] for r in results if 'error' not in r]
        
        # Create results DataFrame
        df = pd.DataFrame({
            'video_name': video_names,
            'prediction': predictions,
            'confidence': confidences,
            'predicted_class': predicted_classes
        })
        
        report = {
            'total_videos': len(results),
            'successful_predictions': len(video_names),
            'failed_predictions': len(results) - len(video_names),
            'prediction_distribution': df['predicted_class'].value_counts().to_dict(),
            'average_confidence': float(np.mean(confidences)),
            'confidence_std': float(np.std(confidences))
        }
        
        # If ground truth is provided, calculate metrics
        if ground_truth:
            true_labels = [ground_truth.get(name, -1) for name in video_names]
            valid_indices = [i for i, label in enumerate(true_labels) if label != -1]
            
            if valid_indices:
                y_true = [true_labels[i] for i in valid_indices]
                y_pred = [predictions[i] for i in valid_indices]
                y_proba = np.array([results[i]['video_probabilities'] for i in valid_indices])
                
                # Calculate metrics
                report.update({
                    'accuracy': float(accuracy_score(y_true, y_pred)),
                    'precision': float(precision_score(y_true, y_pred, average='weighted')),
                    'recall': float(recall_score(y_true, y_pred, average='weighted')),
                    'f1_score': float(f1_score(y_true, y_pred, average='weighted')),
                    'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
                    'classification_report': classification_report(y_true, y_pred, target_names=['fake', 'real'], output_dict=True)
                })
                
                # Generate visualizations
                self._plot_confusion_matrix(y_true, y_pred)
                self._plot_roc_curve(y_true, y_proba)
                self._plot_precision_recall_curve(y_true, y_proba)
        
        # Generate general visualizations
        self._plot_confidence_distribution(confidences, predicted_classes)
        self._plot_prediction_distribution(predicted_classes)
        
        # Save detailed results
        df.to_csv(self.output_dir / 'detailed_results.csv', index=False)
        
        return report
    
    def _plot_confusion_matrix(self, y_true: List[int], y_pred: List[int]):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_roc_curve(self, y_true: List[int], y_proba: np.ndarray):
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'roc_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_precision_recall_curve(self, y_true: List[int], y_proba: np.ndarray):
        """Plot Precision-Recall curve"""
        precision, recall, _ = precision_recall_curve(y_true, y_proba[:, 1])
        avg_precision = average_precision_score(y_true, y_proba[:, 1])
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {avg_precision:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'precision_recall_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_confidence_distribution(self, confidences: List[float], predicted_classes: List[str]):
        """Plot confidence distribution"""
        df = pd.DataFrame({
            'confidence': confidences,
            'predicted_class': predicted_classes
        })
        
        plt.figure(figsize=(12, 5))
        
        # Overall confidence distribution
        plt.subplot(1, 2, 1)
        plt.hist(confidences, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')
        plt.title('Overall Confidence Distribution')
        plt.grid(True, alpha=0.3)
        
        # Confidence by class
        plt.subplot(1, 2, 2)
        for class_name in df['predicted_class'].unique():
            class_confidences = df[df['predicted_class'] == class_name]['confidence']
            plt.hist(class_confidences, bins=15, alpha=0.7, label=class_name, edgecolor='black')
        
        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')
        plt.title('Confidence Distribution by Class')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'confidence_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_prediction_distribution(self, predicted_classes: List[str]):
        """Plot prediction distribution"""
        class_counts = pd.Series(predicted_classes).value_counts()
        
        plt.figure(figsize=(10, 6))
        
        # Bar plot
        plt.subplot(1, 2, 1)
        bars = plt.bar(class_counts.index, class_counts.values, color=['#ff9999', '#66b3ff'])
        plt.xlabel('Predicted Class')
        plt.ylabel('Count')
        plt.title('Prediction Distribution')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
        
        # Pie chart
        plt.subplot(1, 2, 2)
        plt.pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%',
               colors=['#ff9999', '#66b3ff'], startangle=90)
        plt.title('Prediction Distribution')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'prediction_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_report(self, report: Dict[str, Any], filename: str = 'evaluation_report.json'):
        """Save evaluation report to JSON"""
        with open(self.output_dir / filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Evaluation report saved to {self.output_dir / filename}")

def parse_ground_truth(ground_truth_path: str) -> Dict[str, int]:
    """Parse ground truth file (CSV or JSON format)"""
    ground_truth_path = Path(ground_truth_path)
    
    if ground_truth_path.suffix.lower() == '.csv':
        df = pd.read_csv(ground_truth_path)
        # Assume columns: video_name, label (0 for fake, 1 for real)
        return dict(zip(df['video_name'], df['label']))
    
    elif ground_truth_path.suffix.lower() == '.json':
        with open(ground_truth_path, 'r') as f:
            return json.load(f)
    
    else:
        raise ValueError(f"Unsupported ground truth format: {ground_truth_path.suffix}")

def main():
    parser = argparse.ArgumentParser(description='GenConViT Video Prediction and Evaluation')
    
    # Required arguments
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--video-dir', type=str, required=True,
                       help='Directory containing video files to evaluate')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Directory to save results and visualizations')
    
    # Optional arguments
    parser.add_argument('--ground-truth', type=str, default=None,
                       help='Path to ground truth file (CSV or JSON)')
    parser.add_argument('--frame-interval', type=int, default=5,
                       help='Extract every Nth frame (default: 5)')
    parser.add_argument('--max-frames', type=int, default=50,
                       help='Maximum frames per video (default: 50)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for prediction (default: 32)')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use for inference (default: auto)')
    parser.add_argument('--temp-dir', type=str, default='./temp_frames',
                       help='Temporary directory for extracted frames (default: ./temp_frames)')
    parser.add_argument('--keep-frames', action='store_true',
                       help='Keep extracted frames after evaluation')
    parser.add_argument('--video-extensions', nargs='+', 
                       default=['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'],
                       help='Video file extensions to process')
    
    args = parser.parse_args()
    
    # Setup
    logger.info("Starting GenConViT Video Evaluation")
    logger.info(f"Model: {args.model}")
    logger.info(f"Video directory: {args.video_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find video files
    video_dir = Path(args.video_dir)
    video_files = []
    for ext in args.video_extensions:
        video_files.extend(list(video_dir.rglob(f"*{ext}")))
        video_files.extend(list(video_dir.rglob(f"*{ext.upper()}")))
    
    logger.info(f"Found {len(video_files)} video files")
    
    if len(video_files) == 0:
        logger.error("No video files found!")
        return
    
    # Initialize components
    extractor = VideoFrameExtractor(
        frame_interval=args.frame_interval,
        max_frames=args.max_frames,
        target_size=(224, 224)
    )
    
    predictor = ModelPredictor(args.model, args.device)
    reporter = EvaluationReporter(args.output_dir)
    
    # Load ground truth if provided
    ground_truth = None
    if args.ground_truth:
        try:
            ground_truth = parse_ground_truth(args.ground_truth)
            logger.info(f"Loaded ground truth for {len(ground_truth)} videos")
        except Exception as e:
            logger.error(f"Error loading ground truth: {e}")
    
    # Process videos
    results = []
    temp_dir = Path(args.temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    
    for i, video_path in enumerate(video_files):
        logger.info(f"Processing video {i+1}/{len(video_files)}: {video_path.name}")
        
        try:
            result = predictor.predict_video(str(video_path), extractor, str(temp_dir))
            results.append(result)
            
            logger.info(f"  Prediction: {result.get('predicted_class', 'Error')} "
                       f"(Confidence: {result.get('confidence', 0):.3f})")
            
        except Exception as e:
            logger.error(f"Error processing {video_path.name}: {e}")
            results.append({
                'video_path': str(video_path),
                'video_name': video_path.stem,
                'error': str(e)
            })
    
    # Generate report
    logger.info("Generating evaluation report...")
    report = reporter.generate_video_report(results, ground_truth)
    
    # Save results
    reporter.save_report(report)
    
    # Save detailed results
    with open(output_dir / 'detailed_predictions.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Clean up temporary frames if requested
    if not args.keep_frames:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        logger.info("Temporary frames cleaned up")
    
    # Print summary
    total_time = time.time() - start_time
    logger.info("=" * 60)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total videos processed: {report['total_videos']}")
    logger.info(f"Successful predictions: {report['successful_predictions']}")
    logger.info(f"Failed predictions: {report['failed_predictions']}")
    logger.info(f"Average confidence: {report['average_confidence']:.3f}")
    
    if 'accuracy' in report:
        logger.info(f"Accuracy: {report['accuracy']:.3f}")
        logger.info(f"F1 Score: {report['f1_score']:.3f}")
    
    logger.info(f"Processing time: {total_time:.2f} seconds")
    logger.info(f"Results saved to: {args.output_dir}")
    logger.info("=" * 60)

if __name__ == '__main__':
    main()