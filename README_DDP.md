# GenConViT-v2 Distributed Training (DDP)

A high-performance distributed training implementation for GenConViT deepfake detection using PyTorch DDP (Distributed Data Parallel).

## 🚀 Features

- **Multi-GPU Training**: Automatic scaling across multiple GPUs
- **Frame-Based Processing**: Optimized for individual PNG/JPG frame files
- **Memory Efficient**: Optimized data loading and GPU memory usage
- **Checkpoint System**: Automatic saving of best models and periodic checkpoints
- **Resume Training**: Continue training from any checkpoint
- **Comprehensive Logging**: Detailed training metrics and progress tracking
- **Easy Configuration**: Simple command-line interface with sensible defaults

## 📋 Requirements

- Python 3.7+
- PyTorch 1.9+ with CUDA support
- Multiple CUDA-compatible GPUs (recommended)
- Sufficient GPU memory (8GB+ per GPU recommended)

## 🛠️ Installation

Ensure you have the required dependencies:

```bash
pip install torch torchvision timm
pip install pillow numpy
```

## 📁 Data Structure

Your data should be organized as follows:

```
your_data_directory/
├── train/
│   ├── real/
│   │   ├── frame001.png
│   │   ├── frame002.png
│   │   └── ...
│   └── fake/
│       ├── frame001.png
│       ├── frame002.png
│       └── ...
└── val/
    ├── real/
    │   ├── frame001.png
    │   ├── frame002.png
    │   └── ...
    └── fake/
        ├── frame001.png
        ├── frame002.png
        └── ...
```

**Note**: Each PNG/JPG file is treated as an individual training sample.

## 🚀 Quick Start

### Basic Training (Use All Available GPUs)

```bash
cd genconvit-v2
./launch_ddp.sh
```

This will:
- Use all available GPUs
- Train for 50 epochs
- Use batch size 16 per GPU
- Save model to `./models/genconvit_ddp_best.pth`

### Custom Training Configuration

```bash
./launch_ddp.sh \
    --data ./your_data_directory \
    --batch-size 8 \
    --epochs 30 \
    --lr 0.0001 \
    --world-size 2
```

## 📝 Command Reference

### Launch Script Options

| Option | Description | Default |
|--------|-------------|---------|
| `--data DIR` | Data directory path | `./1000_videos_combined` |
| `--batch-size N` | Batch size per GPU | `16` |
| `--epochs N` | Number of training epochs | `50` |
| `--lr RATE` | Learning rate | `0.0001` |
| `--num-workers N` | Data loading workers per GPU | `4` |
| `--save-path PATH` | Model save path | `./models/genconvit_ddp_best.pth` |
| `--world-size N` | Number of GPUs (-1 for all) | `-1` |
| `--resume PATH` | Resume from checkpoint | None |

### Direct Python Script Usage

```bash
python train_ddp.py \
    --data ./your_data \
    --batch-size 16 \
    --epochs 50 \
    --lr 0.0001 \
    --world-size 2 \
    --save-path ./models/best_model.pth
```

### Full Parameter List

```bash
python train_ddp.py \
    --data ./1000_videos_combined \
    --input-size 224 \
    --batch-size 16 \
    --epochs 100 \
    --lr 0.0001 \
    --weight-decay 1e-5 \
    --lr-step 30 \
    --lr-gamma 0.1 \
    --beta 1.0 \
    --num-workers 4 \
    --world-size -1 \
    --save-path ./models/genconvit_best.pth \
    --save-every 10 \
    --seed 42
```

## 🎯 Usage Examples

### Example 1: Small Dataset, Single GPU
```bash
./launch_ddp.sh \
    --data ./small_dataset \
    --batch-size 32 \
    --epochs 20 \
    --world-size 1
```

### Example 2: Large Dataset, Multi-GPU
```bash
./launch_ddp.sh \
    --data ./large_dataset \
    --batch-size 8 \
    --epochs 100 \
    --lr 0.0005 \
    --world-size 4 \
    --num-workers 8
```

### Example 3: Resume Training
```bash
./launch_ddp.sh \
    --data ./1000_videos_combined \
    --resume ./models/genconvit_ddp_best.pth \
    --epochs 100
```

### Example 4: Memory-Constrained Training
```bash
./launch_ddp.sh \
    --data ./your_data \
    --batch-size 4 \
    --num-workers 2 \
    --world-size 1
```

### Example 5: High-Performance Training
```bash
./launch_ddp.sh \
    --data ./your_data \
    --batch-size 32 \
    --epochs 200 \
    --lr 0.0002 \
    --world-size 8 \
    --num-workers 8 \
    --lr-step 50 \
    --lr-gamma 0.5
```

## 📊 Performance Optimization

### Batch Size Guidelines

| GPU Memory | Recommended Batch Size per GPU |
|------------|--------------------------------|
| 8GB        | 4-8                           |
| 11GB       | 8-16                          |
| 16GB+      | 16-32                         |

### Multi-GPU Scaling

| Number of GPUs | Total Batch Size | Expected Speedup |
|----------------|------------------|------------------|
| 1              | 16               | 1x               |
| 2              | 32               | ~1.8x            |
| 4              | 64               | ~3.5x            |
| 8              | 128              | ~6.5x            |

### Memory Optimization Tips

1. **Reduce batch size** if you encounter OOM errors
2. **Reduce num-workers** if CPU memory is limited
3. **Use gradient checkpointing** for very large models
4. **Mixed precision training** (automatically handled)

## 📈 Monitoring Training

### Real-time Monitoring

Training progress is displayed in the terminal:

```
Epoch 1/50: Train Loss: 1.2345, Train Acc: 0.6789, Val Loss: 1.1234, Val Acc: 0.7123, LR: 0.000100
New best model saved with validation accuracy: 0.7123
Batch 100/500, Loss: 1.1234, Acc: 72.50%
```

### Log Files

- `training_ddp.log`: Comprehensive training log
- Checkpoints saved every N epochs (configurable)

### Tensorboard Integration (Optional)

To add Tensorboard logging, modify the training script or use external monitoring tools.

## 🔧 Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
```bash
# Reduce batch size
./launch_ddp.sh --batch-size 4

# Reduce number of workers
./launch_ddp.sh --num-workers 2

# Use single GPU
./launch_ddp.sh --world-size 1
```

#### 2. No CUDA Devices Found
```bash
# Check CUDA installation
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.cuda.device_count())"
```

#### 3. Data Loading Errors
```bash
# Check data structure
find ./your_data -name "*.png" | head -10
find ./your_data -name "*.jpg" | head -10

# Verify directory structure
ls -la ./your_data/train/
ls -la ./your_data/val/
```

#### 4. Port Already in Use
```bash
# Kill existing processes
pkill -f "train_ddp.py"

# Or change port in train_ddp.py (line 164):
os.environ['MASTER_PORT'] = '12356'  # Change port number
```

#### 5. Slow Training
```bash
# Increase number of workers
./launch_ddp.sh --num-workers 8

# Increase batch size (if memory allows)
./launch_ddp.sh --batch-size 32

# Use more GPUs
./launch_ddp.sh --world-size 4
```

## 📋 Model Evaluation

### Load and Evaluate Trained Model

```python
import torch
from genconvit import GenConViT

# Load checkpoint
checkpoint = torch.load('models/genconvit_ddp_best.pth')
model = GenConViT(num_classes=2)
model.load_state_dict(checkpoint['model_state_dict'])

print(f"Best accuracy: {checkpoint['best_acc']:.4f}")
print(f"Trained for {checkpoint['epoch']} epochs")
```

### Inference on New Data

```python
import torch
from torchvision import transforms
from PIL import Image

# Load model
model = GenConViT(num_classes=2)
checkpoint = torch.load('models/genconvit_ddp_best.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Prepare image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Inference
image = Image.open('path/to/image.png').convert('RGB')
input_tensor = transform(image).unsqueeze(0)

with torch.no_grad():
    logits, _, _, _, _ = model(input_tensor)
    prediction = torch.softmax(logits, dim=1)
    
print(f"Real: {prediction[0][1]:.4f}, Fake: {prediction[0][0]:.4f}")
```

## 🔬 Advanced Configuration

### Custom Learning Rate Scheduling

```bash
./launch_ddp.sh \
    --lr 0.001 \
    --lr-step 20 \
    --lr-gamma 0.5 \
    --epochs 100
```

### Adjust VAE Loss Weight

```bash
./launch_ddp.sh \
    --beta 0.5  # Reduce VAE loss contribution
```

### Custom Model Architecture

Modify `train_ddp.py` to adjust:
- Latent dimensions
- Backbone models
- Classification heads

## 📝 Best Practices

### 1. Data Preparation
- Ensure balanced dataset (equal real/fake samples)
- Use high-quality images (224x224 or higher)
- Validate data integrity before training

### 2. Training Strategy
- Start with lower learning rate (1e-4)
- Use learning rate scheduling
- Monitor both training and validation metrics
- Save checkpoints regularly

### 3. Resource Management
- Monitor GPU memory usage
- Use appropriate batch sizes
- Balance number of workers with CPU cores

### 4. Hyperparameter Tuning
- Start with default parameters
- Adjust batch size based on GPU memory
- Tune learning rate based on convergence
- Experiment with VAE loss weight (beta)

## 📚 Additional Resources

- [PyTorch DDP Documentation](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [GenConViT Paper](https://arxiv.org/abs/paper-link)
- [DeepFake Detection Best Practices](https://github.com/deepfakes/faceswap)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with different GPU configurations
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section above
2. Verify your environment meets requirements
3. Create an issue with detailed error logs
4. Include system specifications and command used

---

**Happy Distributed Training! 🚀**