# GenConViT-v2: Video-Based DeepFake Detection

A comprehensive deep learning pipeline for detecting deepfake videos using the GenConViT (Generative Convolutional Vision Transformer) architecture. This implementation supports end-to-end training from raw video files to a fully trained deepfake detection model.

## 🚀 Features

- **End-to-end pipeline**: From raw videos to trained model
- **Automatic frame extraction**: Supports multiple video formats
- **Flexible data organization**: Handles various directory structures
- **Advanced model architecture**: GenConViT with dual-path processing
- **Data augmentation**: Comprehensive augmentation pipeline
- **Balanced sampling**: Handles imbalanced datasets
- **Face detection**: Optional face cropping for better performance
- **Resume capability**: Continue training from checkpoints
- **Comprehensive logging**: Detailed progress tracking

## 📋 Requirements

- Python 3.7+
- CUDA-compatible GPU (recommended)
- OpenCV
- PyTorch 1.9+
- See `requirements.txt` for full dependencies

## 🛠️ Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd genconvit-v2
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Verify installation:
```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import cv2; print(f'OpenCV version: {cv2.__version__}')"
```

## 📁 Data Organization

### Option 1: Pre-organized by Class
```
videos/
├── real/
│   ├── video1.mp4
│   ├── video2.avi
│   └── ...
└── fake/
    ├── video1.mp4
    ├── video2.mov
    └── ...
```

### Option 2: Flat Structure (Auto-detection)
```
videos/
├── real_video1.mp4
├── fake_video1.mp4
├── original_video2.avi
├── deepfake_video2.mp4
└── ...
```

The system will automatically detect class names from:
- Directory structure (Option 1)
- Filename patterns: `real`, `original`, `genuine` → "real" class
- Filename patterns: `fake`, `deepfake`, `synthetic` → "fake" class

## 🚀 Quick Start

### Complete Pipeline (Recommended)

Train a model from raw videos in one command:

```bash
python train_from_videos.py \
    --video_dir ./videos \
    --output_dir ./training_output \
    --epochs 30 \
    --batch_size 16
```

### Step-by-Step Approach

#### 1. Extract Frames from Videos

```bash
python extract_frames.py \
    --input_dir ./videos \
    --output_dir ./extracted_frames \
    --frame_interval 5 \
    --width 224 \
    --height 224
```

#### 2. Train the Model

```bash
python genconvit.py \
    --data ./extracted_frames \
    --batch_size 32 \
    --epochs 20 \
    --lr 0.0001 \
    --mode train
```

#### 3. Evaluate the Model

```bash
python genconvit.py \
    --data ./extracted_frames \
    --save_path ./genconvit_best.pth \
    --mode eval
```

## 🔧 Advanced Usage

### Frame Extraction Options

```bash
# Extract every 10th frame with face detection
python extract_frames.py \
    --input_dir ./videos \
    --output_dir ./frames \
    --frame_interval 10 \
    --face_detection \
    --max_frames 100

# Custom resolution and video extensions
python extract_frames.py \
    --input_dir ./videos \
    --output_dir ./frames \
    --width 128 \
    --height 128 \
    --extensions .mp4 .avi .mov
```

### Training with Custom Parameters

```bash
# Training with balanced sampling and custom splits
python train_from_videos.py \
    --video_dir ./videos \
    --output_dir ./training_output \
    --train_split 0.8 \
    --val_split 0.1 \
    --test_split 0.1 \
    --balanced_sampling \
    --face_detection \
    --epochs 50 \
    --batch_size 16 \
    --lr 0.0001 \
    --weight_decay 1e-5
```

### Resume Training from Checkpoint

```bash
python train_from_videos.py \
    --video_dir ./videos \
    --output_dir ./training_output \
    --resume_from_checkpoint ./training_output/models/checkpoint_epoch_20.pth \
    --epochs 50
```

## 📊 Model Architecture

GenConViT uses a dual-path architecture:

- **Path A**: AutoEncoder + ConvNeXt + Swin Transformer
- **Path B**: Variational AutoEncoder + ConvNeXt + Swin Transformer
- **Fusion**: Combined logits from both paths

### Key Components:
- **AutoEncoder**: Learns compressed representations
- **Variational AutoEncoder**: Learns probabilistic representations
- **ConvNeXt**: Modern CNN backbone
- **Swin Transformer**: Hierarchical vision transformer
- **Dual Classification Heads**: Separate processing paths

## 📈 Training Configuration

### Default Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 32 | Training batch size |
| `epochs` | 20 | Number of training epochs |
| `learning_rate` | 1e-4 | Initial learning rate |
| `weight_decay` | 1e-5 | L2 regularization |
| `frame_interval` | 5 | Extract every Nth frame |
| `target_size` | 224x224 | Frame resolution |
| `beta` | 1.0 | VAE loss weight |

### Recommended Settings

**For limited GPU memory:**
```bash
--batch_size 8 --frames_per_video 10
```

**For large datasets:**
```bash
--balanced_sampling --max_frames 200
```

**For high accuracy:**
```bash
--face_detection --epochs 50 --lr 0.00005
```

## 📝 Output Structure

After running the complete pipeline:

```
training_output/
├── extracted_frames/
│   ├── train/
│   │   ├── real/
│   │   └── fake/
│   ├── val/
│   └── test/
├── models/
│   ├── best_model.pth
│   ├── checkpoint_epoch_5.pth
│   └── checkpoint_epoch_10.pth
├── logs/
│   ├── training_pipeline.log
│   └── frame_extraction.log
└── pipeline_results.json
```

## 🎯 Performance Tips

### For Better Accuracy:
1. Use face detection: `--face_detection`
2. Increase training epochs: `--epochs 50`
3. Use balanced sampling: `--balanced_sampling`
4. Extract more frames: `--frame_interval 3`

### For Faster Training:
1. Reduce batch size: `--batch_size 16`
2. Limit frames per video: `--frames_per_video 20`
3. Use fewer workers: `--num_workers 2`
4. Lower resolution: `--target_width 128 --target_height 128`

### For Large Datasets:
1. Use frame sampling: `--max_frames 200`
2. Enable balanced sampling: `--balanced_sampling`
3. Increase workers: `--num_workers 8`
4. Use multiple GPUs if available

## 🐛 Troubleshooting

### Common Issues:

**Out of Memory Error:**
```bash
# Reduce batch size and frame count
--batch_size 8 --frames_per_video 10
```

**No videos found:**
- Check video file extensions
- Verify directory structure
- Use `--extensions` to specify formats

**Low accuracy:**
- Increase training epochs
- Enable face detection
- Use data augmentation
- Check class balance

**Slow training:**
- Reduce number of workers if CPU-bound
- Use GPU if available
- Reduce frame resolution

## 📚 API Reference

### VideoFrameDataset
```python
from video_dataset import VideoFrameDataset

dataset = VideoFrameDataset(
    root_dir="./frames",
    split="train",
    frames_per_video=20,
    random_frame_selection=True
)
```

### FrameExtractor
```python
from extract_frames import FrameExtractor

extractor = FrameExtractor(
    input_dir="./videos",
    output_dir="./frames",
    frame_interval=5,
    target_size=(224, 224),
    face_detection=True
)
extractor.extract_all_frames()
```

### GenConViT Model
```python
from genconvit import GenConViT

model = GenConViT(
    ae_latent=256,
    vae_latent=256,
    num_classes=2
)
```

## 📄 Citation

If you use this code in your research, please cite:

```bibtex
@article{genconvit2024,
  title={GenConViT: Generative Convolutional Vision Transformer for DeepFake Detection},
  author={[Authors]},
  journal={[Journal]},
  year={2024}
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔗 Related Work

- [FaceForensics++](https://github.com/ondyari/FaceForensics)
- [ConvNeXt](https://github.com/facebookresearch/ConvNeXt)
- [Swin Transformer](https://github.com/microsoft/Swin-Transformer)
- [TIMM Models](https://github.com/rwightman/pytorch-image-models)

## 📞 Support

For questions and support:
- Create an issue in the repository
- Check the troubleshooting section
- Review the examples in the code

---

**Happy DeepFake Detection! 🎭**