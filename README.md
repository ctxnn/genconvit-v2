# GenConViT-v2: Complete Video-Based DeepFake Detection System

A comprehensive deep learning pipeline for detecting deepfake videos using the GenConViT (Generative Convolutional Vision Transformer) architecture with distributed training and advanced evaluation capabilities.

## 🚀 **Key Features**

### **Training Options**
- **🔥 Distributed Training (DDP)**: Multi-GPU training with PyTorch DDP
- **🖥️ Single GPU Training**: Standard single GPU training
- **💻 CPU Training**: CPU-only training for systems without GPU
- **📸 Frame-Based Processing**: Direct training on PNG/JPG frame files
- **⚡ High Performance**: Optimized data loading and memory usage
- **💾 Smart Checkpointing**: Auto-save best models and resume training

### **Evaluation System**
- **🎬 Video Processing**: Automatic frame extraction from videos
- **📈 Advanced Metrics**: ROC curves, confusion matrices, PR curves
- **📊 Beautiful Visualizations**: Professional graphs and charts
- **🎯 No-Label Support**: Works with or without ground truth labels
- **📋 Detailed Reports**: JSON, CSV, and visual outputs

### **Model Architecture**
- **🧠 Dual-Path Design**: AutoEncoder + VAE pathways
- **🔍 Modern Backbones**: ConvNeXt + Swin Transformer
- **🎛️ Flexible Configuration**: Customizable architecture parameters

## 📋 **Quick Setup**

### **Requirements**
```bash
# Install dependencies
pip install torch torchvision timm opencv-python pillow numpy pandas matplotlib seaborn scikit-learn

# For CPU-only systems (optional)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### **Data Structure**
```
your_data/
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
    └── fake/
```

## 🚀 **Training Options**

### **Option 1: Distributed Training (Multi-GPU) - RECOMMENDED**

#### **Quick Start - Use All GPUs**
```bash
cd genconvit-v2
./launch_ddp.sh --data ./your_data --epochs 50
```

#### **Custom Multi-GPU Training**
```bash
./launch_ddp.sh \
    --data ./your_data \
    --batch-size 16 \
    --epochs 100 \
    --lr 0.0001 \
    --world-size 2
```

#### **Memory-Efficient Multi-GPU**
```bash
./launch_ddp.sh \
    --data ./your_data \
    --batch-size 4 \
    --world-size 2 \
    --num-workers 2
```

### **Option 2: Single GPU Training**

#### **Basic Single GPU**
```bash
python genconvit.py \
    --data ./your_data \
    --batch-size 16 \
    --epochs 30 \
    --lr 0.0001 \
    --mode train
```

#### **Single GPU with Custom Settings**
```bash
python genconvit.py \
    --data ./your_data \
    --batch-size 8 \
    --epochs 50 \
    --lr 0.0001 \
    --weight-decay 1e-5 \
    --balanced-sampling \
    --mode train \
    --save-path ./single_gpu_model.pth
```

#### **Memory-Constrained Single GPU**
```bash
python genconvit.py \
    --data ./your_data \
    --batch-size 4 \
    --epochs 30 \
    --lr 0.0001 \
    --num-workers 2 \
    --mode train
```

### **Option 3: CPU Training (No GPU Required)**

#### **Basic CPU Training**
```bash
# Force CPU usage
CUDA_VISIBLE_DEVICES="" python genconvit.py \
    --data ./your_data \
    --batch-size 2 \
    --epochs 20 \
    --lr 0.001 \
    --num-workers 1 \
    --mode train
```

#### **CPU Training with Optimizations**
```bash
# Set CPU threads for better performance
export OMP_NUM_THREADS=4
CUDA_VISIBLE_DEVICES="" python genconvit.py \
    --data ./your_data \
    --batch-size 1 \
    --epochs 15 \
    --lr 0.001 \
    --num-workers 1 \
    --mode train \
    --save-path ./cpu_model.pth
```

### **Training Performance Comparison**

| Setup | Batch Size | Typical Speed | Memory Usage | Recommended For |
|-------|------------|---------------|--------------|-----------------|
| **Multi-GPU (4x)** | 32 per GPU | ~4x faster | High | Large datasets, fast training |
| **Single GPU** | 16-32 | Baseline | Medium | Most use cases |
| **CPU Only** | 1-4 | ~10x slower | Low | No GPU available, small datasets |

## 📊 **Evaluation & Prediction**

### **Evaluate on Videos (With Labels)**
```bash
./evaluate.sh \
    --model ./models/genconvit_best.pth \
    --video-dir ./test_videos \
    --ground-truth ./labels.csv
```

### **Evaluate on Videos (No Labels)**
```bash
./evaluate.sh \
    --model ./models/genconvit_best.pth \
    --video-dir ./unknown_videos
```

### **CPU Evaluation**
```bash
./evaluate.sh \
    --model ./models/model.pth \
    --video-dir ./videos \
    --device cpu \
    --batch-size 1 \
    --num-workers 1
```

### **Custom Evaluation**
```bash
./evaluate.sh \
    --model ./models/model.pth \
    --video-dir ./videos \
    --output-dir ./my_results \
    --frame-interval 10 \
    --max-frames 30 \
    --batch-size 16
```

## 📝 **Complete Command Reference**

### **Training Commands**

| Command | Description | Best For |
|---------|-------------|----------|
| `./launch_ddp.sh` | Multi-GPU distributed training | Multiple GPUs available |
| `python genconvit.py` | Single GPU/CPU training | Single GPU or CPU only |
| `./launch_ddp.sh --help` | Show distributed training options | - |
| `python genconvit.py --help` | Show single GPU/CPU options | - |

### **Key Training Parameters**

| Parameter | Multi-GPU Default | Single GPU Default | CPU Default | Description |
|-----------|------------------|-------------------|-------------|-------------|
| `--batch-size` | 16 | 16 | 2 | Batch size (per GPU for multi-GPU) |
| `--epochs` | 50 | 20 | 15 | Number of training epochs |
| `--lr` | 0.0001 | 0.0001 | 0.001 | Learning rate |
| `--num-workers` | 4 | 4 | 1 | Data loading workers |
| `--world-size` | -1 (all GPUs) | N/A | N/A | Number of GPUs to use |

### **Evaluation Parameters**

| Parameter | Description | Default | CPU Recommended |
|-----------|-------------|---------|-----------------|
| `--model PATH` | Trained model path | Required | Required |
| `--video-dir DIR` | Video directory | Required | Required |
| `--batch-size N` | Prediction batch size | 32 | 1-4 |
| `--device` | cpu/cuda/auto | auto | cpu |
| `--frame-interval N` | Extract every Nth frame | 5 | 10 |
| `--max-frames N` | Max frames per video | 50 | 20 |

## 🎯 **Usage Examples by System Type**

### **🖥️ Multi-GPU System (Recommended)**
```bash
# Training
./launch_ddp.sh \
    --data ./my_data \
    --batch-size 32 \
    --epochs 100 \
    --world-size 4

# Evaluation
./evaluate.sh \
    --model ./models/genconvit_ddp_best.pth \
    --video-dir ./test_videos \
    --batch-size 32
```

### **🔥 Single GPU System**
```bash
# Training
python genconvit.py \
    --data ./my_data \
    --batch-size 16 \
    --epochs 50 \
    --lr 0.0001 \
    --balanced-sampling \
    --mode train

# Evaluation
./evaluate.sh \
    --model ./genconvit_best.pth \
    --video-dir ./test_videos \
    --batch-size 16
```

### **💻 CPU-Only System**
```bash
# Training (be patient - this takes time!)
export OMP_NUM_THREADS=4
CUDA_VISIBLE_DEVICES="" python genconvit.py \
    --data ./small_dataset \
    --batch-size 1 \
    --epochs 10 \
    --lr 0.001 \
    --num-workers 1 \
    --mode train

# Evaluation
./evaluate.sh \
    --model ./genconvit_best.pth \
    --video-dir ./test_videos \
    --device cpu \
    --batch-size 1 \
    --frame-interval 10 \
    --max-frames 10
```

### **🚀 Quick Start by Experience Level**

#### **Beginner (Just want it to work)**
```bash
# If you have GPU
python genconvit.py --data ./your_data --mode train

# If you don't have GPU
CUDA_VISIBLE_DEVICES="" python genconvit.py --data ./your_data --batch-size 1 --mode train

# Evaluate
./evaluate.sh --model ./genconvit_best.pth --video-dir ./videos
```

#### **Intermediate (Want good performance)**
```bash
# Multi-GPU if available
./launch_ddp.sh --data ./your_data --batch-size 16 --epochs 50

# Single GPU
python genconvit.py --data ./your_data --batch-size 16 --epochs 30 --balanced-sampling --mode train

# Evaluate with labels
./evaluate.sh --model ./models/best.pth --video-dir ./videos --ground-truth ./labels.csv
```

#### **Advanced (Want full control)**
```bash
# Custom multi-GPU training
./launch_ddp.sh \
    --data ./data \
    --batch-size 32 \
    --epochs 200 \
    --lr 0.0001 \
    --weight-decay 1e-5 \
    --lr-step 50 \
    --world-size 4 \
    --save-every 10

# Detailed evaluation
./evaluate.sh \
    --model ./models/model.pth \
    --video-dir ./videos \
    --ground-truth ./labels.csv \
    --frame-interval 3 \
    --max-frames 100 \
    --output-dir ./detailed_results
```

## 📊 **Output Files**

### **Training Outputs**
```
# Multi-GPU Training
models/
├── genconvit_ddp_best.pth          # Best model
├── checkpoint_epoch_10.pth         # Periodic checkpoints
└── training_ddp.log                # Training log

# Single GPU Training
├── genconvit_best.pth              # Best model
└── training.log                    # Training log
```

### **Evaluation Outputs**
```
evaluation_results/
├── evaluation_report.json          # Main metrics and statistics
├── detailed_results.csv            # Per-video predictions
├── detailed_predictions.json       # Complete prediction data
├── confusion_matrix.png            # Confusion matrix (if labels provided)
├── roc_curve.png                   # ROC curve (if labels provided)
├── precision_recall_curve.png      # PR curve (if labels provided)
├── confidence_distribution.png     # Confidence score distribution
└── prediction_distribution.png     # Prediction class distribution
```

## 🔧 **Troubleshooting**

### **Training Issues**

#### **CUDA Out of Memory**
```bash
# Reduce batch size
--batch-size 4

# Reduce workers
--num-workers 2

# Use CPU instead
CUDA_VISIBLE_DEVICES="" python genconvit.py --data ./data --batch-size 1 --mode train
```

#### **No GPU Available**
```bash
# Check GPU
python -c "import torch; print(torch.cuda.is_available())"

# Force CPU training
CUDA_VISIBLE_DEVICES="" python genconvit.py --data ./data --batch-size 1 --epochs 10 --mode train
```

#### **Slow Training on CPU**
```bash
# Optimize CPU training
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# Use smaller batch size and fewer epochs
python genconvit.py --data ./data --batch-size 1 --epochs 5 --mode train
```

#### **Dataset Loading Errors**
```bash
# Check data structure
find ./your_data -name "*.png" | head -10
ls -la ./your_data/train/real/
ls -la ./your_data/train/fake/
```

### **Evaluation Issues**

#### **Model Loading Errors**
```bash
# Check model file
ls -la ./models/
python -c "import torch; print(torch.load('./model.pth', map_location='cpu').keys())"
```

#### **Video Processing Errors**
```bash
# Check video files
find ./videos -name "*.mp4" | head -5
file ./videos/*.mp4

# Reduce frame extraction if needed
./evaluate.sh --model ./model.pth --video-dir ./videos --max-frames 10
```

## 📈 **Performance Guidelines**

### **Training Performance by System**

| System Type | Batch Size | Epochs | Expected Time (1000 samples) |
|-------------|------------|--------|-------------------------------|
| **4x RTX 4090** | 128 total | 50 | 30 minutes |
| **2x RTX 3080** | 32 total | 50 | 2 hours |
| **1x RTX 3060** | 16 | 30 | 4 hours |
| **1x GTX 1660** | 8 | 20 | 8 hours |
| **CPU (8 cores)** | 2 | 10 | 24+ hours |

### **Memory Requirements**

| Setup | GPU Memory | System RAM | Disk Space |
|-------|------------|------------|------------|
| **Multi-GPU** | 8GB+ per GPU | 16GB+ | 10GB+ |
| **Single GPU** | 6GB+ | 8GB+ | 5GB+ |
| **CPU Only** | N/A | 4GB+ | 2GB+ |

## 🎛️ **Advanced Configuration**

### **Custom Model Architecture**
```python
# Modify in train_ddp.py or genconvit.py
model = GenConViT(
    ae_latent=512,      # AutoEncoder latent dimension
    vae_latent=512,     # VAE latent dimension
    num_classes=2       # Number of classes
)
```

### **Environment Variables for Optimization**
```bash
# CPU optimization
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# GPU optimization
export CUDA_LAUNCH_BLOCKING=0
```

## 📚 **Ground Truth Formats**

### **CSV Format**
```csv
video_name,label
video001,0
video002,1
video003,0
```

### **JSON Format**
```json
{
    "video001": 0,
    "video002": 1,
    "video003": 0
}
```

Where: `0 = fake`, `1 = real`

## 🎯 **Best Practices**

### **For Multi-GPU Systems**
1. Use distributed training with `./launch_ddp.sh`
2. Start with batch size 16-32 per GPU
3. Monitor GPU memory usage
4. Use multiple workers (4-8 per GPU)

### **For Single GPU Systems**
1. Use `python genconvit.py` directly
2. Start with batch size 8-16
3. Enable balanced sampling
4. Monitor memory usage

### **For CPU-Only Systems**
1. Use very small batch sizes (1-2)
2. Reduce number of epochs (5-15)
3. Use fewer workers (1-2)
4. Consider using a subset of data for testing
5. Be patient - CPU training is slow!

### **Data Preparation**
1. **Balance Dataset**: Equal fake/real samples
2. **Quality Control**: Remove corrupted frames
3. **Consistent Naming**: Use clear naming conventions
4. **Proper Splits**: 70% train, 15% val, 15% test

## 🚀 **Quick Start Checklist**

### **Before Training:**
- [ ] Data organized in `train/val/{real,fake}` structure
- [ ] PNG/JPG frame files in appropriate directories
- [ ] Check if GPU available: `python -c "import torch; print(torch.cuda.is_available())"`
- [ ] Choose appropriate training command based on your system

### **For Multi-GPU Systems:**
- [ ] Run: `./launch_ddp.sh --data ./your_data`

### **For Single GPU Systems:**
- [ ] Run: `python genconvit.py --data ./your_data --mode train`

### **For CPU-Only Systems:**
- [ ] Run: `CUDA_VISIBLE_DEVICES="" python genconvit.py --data ./your_data --batch-size 1 --mode train`

### **For Evaluation:**
- [ ] Trained model file (`.pth`)
- [ ] Video files in a directory
- [ ] (Optional) Ground truth labels file
- [ ] Run: `./evaluate.sh --model ./model.pth --video-dir ./videos`

## 🤝 **Support & Contributing**

### **Getting Help**
1. Check troubleshooting section above
2. Verify your system meets requirements
3. Try with smaller dataset first
4. Create issue with detailed error logs and system specs

### **Contributing**
1. Fork the repository
2. Test on different hardware configurations
3. Submit pull request with examples

## 📄 **License**

This project is licensed under the MIT License.

---

**🎉 Ready to detect deepfakes?**

- **Multi-GPU?** → `./launch_ddp.sh --help`
- **Single GPU?** → `python genconvit.py --help` 
- **CPU only?** → `CUDA_VISIBLE_DEVICES="" python genconvit.py --help`
- **Evaluation?** → `./evaluate.sh --help`

---