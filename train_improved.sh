#!/bin/bash

# GenConViT Improved Single GPU Training Launch Script
# Usage: ./train_improved.sh [options]

set -e

# Default values
DATA_DIR="./1000_videos_combined"
BATCH_SIZE=16
EPOCHS=100
LR=0.0001
NUM_WORKERS=4
SAVE_PATH="./models/genconvit_improved.pth"
DROPOUT_RATE=0.5
AE_LATENT=256
VAE_LATENT=256

# Loss weights
CLASSIFICATION_WEIGHT=1.0
AE_WEIGHT=0.1
VAE_WEIGHT=0.1
VAE_BETA=1.0

# Scheduler parameters
LR_GAMMA=0.5
LR_PATIENCE=5
MIN_LR=1e-7

# Early stopping
EARLY_STOPPING_PATIENCE=10
EARLY_STOPPING_MIN_DELTA=0.001

# Other parameters
WEIGHT_DECAY=1e-4
SAVE_EVERY=5
LOG_INTERVAL=50
SEED=42

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show help
show_help() {
    echo "GenConViT Improved Single GPU Training Script"
    echo ""
    echo "This script trains the improved GenConViT model with:"
    echo "- Fixed model architecture with proper feature fusion"
    echo "- ReduceLROnPlateau scheduler for adaptive learning rate"
    echo "- Early stopping to prevent overfitting"
    echo "- Enhanced regularization and data augmentation"
    echo "- Comprehensive loss monitoring"
    echo ""
    echo "Usage: $0 [options]"
    echo ""
    echo "Data Arguments:"
    echo "  --data DIR                   Data directory (default: $DATA_DIR)"
    echo "  --batch-size N               Batch size (default: $BATCH_SIZE)"
    echo "  --num-workers N              Number of workers (default: $NUM_WORKERS)"
    echo "  --balanced-sampling          Use balanced sampling for imbalanced datasets"
    echo "  --frames-per-video N         Number of frames to sample per video"
    echo ""
    echo "Model Arguments:"
    echo "  --ae-latent N                AutoEncoder latent dimension (default: $AE_LATENT)"
    echo "  --vae-latent N               VAE latent dimension (default: $VAE_LATENT)"
    echo "  --dropout-rate RATE          Dropout rate (default: $DROPOUT_RATE)"
    echo ""
    echo "Training Arguments:"
    echo "  --epochs N                   Number of epochs (default: $EPOCHS)"
    echo "  --lr RATE                    Learning rate (default: $LR)"
    echo "  --weight-decay RATE          Weight decay (default: $WEIGHT_DECAY)"
    echo ""
    echo "Scheduler Arguments (ReduceLROnPlateau):"
    echo "  --lr-gamma FACTOR            LR reduction factor (default: $LR_GAMMA)"
    echo "  --lr-patience N              Scheduler patience (default: $LR_PATIENCE)"
    echo "  --min-lr RATE                Minimum learning rate (default: $MIN_LR)"
    echo ""
    echo "Early Stopping Arguments:"
    echo "  --early-stopping-patience N Early stopping patience (default: $EARLY_STOPPING_PATIENCE)"
    echo "  --early-stopping-min-delta  Early stopping min delta (default: $EARLY_STOPPING_MIN_DELTA)"
    echo ""
    echo "Loss Weight Arguments:"
    echo "  --classification-weight W    Classification loss weight (default: $CLASSIFICATION_WEIGHT)"
    echo "  --ae-weight W                AE reconstruction loss weight (default: $AE_WEIGHT)"
    echo "  --vae-weight W               VAE loss weight (default: $VAE_WEIGHT)"
    echo "  --vae-beta W                 VAE KL divergence weight (default: $VAE_BETA)"
    echo ""
    echo "Checkpoint Arguments:"
    echo "  --save-path PATH             Model save path (default: $SAVE_PATH)"
    echo "  --save-every N               Save every N epochs (default: $SAVE_EVERY)"
    echo "  --resume PATH                Resume from checkpoint"
    echo ""
    echo "Other Arguments:"
    echo "  --log-interval N             Logging interval (default: $LOG_INTERVAL)"
    echo "  --seed N                     Random seed (default: $SEED)"
    echo "  --help, -h                   Show this help message"
    echo ""
    echo "Examples:"
    echo "  # Basic training"
    echo "  $0 --data ./my_data"
    echo ""
    echo "  # Training with custom parameters"
    echo "  $0 --data ./my_data --batch-size 32 --epochs 50 --lr 0.0002"
    echo ""
    echo "  # Training with enhanced regularization"
    echo "  $0 --data ./my_data --dropout-rate 0.6 --weight-decay 1e-3 --early-stopping-patience 15"
    echo ""
    echo "  # Training with custom loss weights"
    echo "  $0 --data ./my_data --ae-weight 0.2 --vae-weight 0.2 --vae-beta 0.5"
    echo ""
    echo "  # Resume training"
    echo "  $0 --data ./my_data --resume ./models/checkpoint.pth"
    echo ""
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data)
            DATA_DIR="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --num-workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --save-path)
            SAVE_PATH="$2"
            shift 2
            ;;
        --dropout-rate)
            DROPOUT_RATE="$2"
            shift 2
            ;;
        --ae-latent)
            AE_LATENT="$2"
            shift 2
            ;;
        --vae-latent)
            VAE_LATENT="$2"
            shift 2
            ;;
        --weight-decay)
            WEIGHT_DECAY="$2"
            shift 2
            ;;
        --lr-gamma)
            LR_GAMMA="$2"
            shift 2
            ;;
        --lr-patience)
            LR_PATIENCE="$2"
            shift 2
            ;;
        --min-lr)
            MIN_LR="$2"
            shift 2
            ;;
        --early-stopping-patience)
            EARLY_STOPPING_PATIENCE="$2"
            shift 2
            ;;
        --early-stopping-min-delta)
            EARLY_STOPPING_MIN_DELTA="$2"
            shift 2
            ;;
        --classification-weight)
            CLASSIFICATION_WEIGHT="$2"
            shift 2
            ;;
        --ae-weight)
            AE_WEIGHT="$2"
            shift 2
            ;;
        --vae-weight)
            VAE_WEIGHT="$2"
            shift 2
            ;;
        --vae-beta)
            VAE_BETA="$2"
            shift 2
            ;;
        --save-every)
            SAVE_EVERY="$2"
            shift 2
            ;;
        --log-interval)
            LOG_INTERVAL="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --balanced-sampling)
            BALANCED_SAMPLING="--balanced-sampling"
            shift
            ;;
        --frames-per-video)
            FRAMES_PER_VIDEO="--frames-per-video $2"
            shift 2
            ;;
        --resume)
            RESUME="--resume $2"
            shift 2
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if CUDA is available
if ! python -c "import torch; print('CUDA available:', torch.cuda.is_available())" 2>/dev/null; then
    print_error "CUDA check failed. Please ensure PyTorch is installed."
    exit 1
fi

# Check GPU count
GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")
if [ "$GPU_COUNT" -eq 0 ]; then
    print_warning "No CUDA devices found! Training will use CPU (very slow)."
else
    print_success "Found $GPU_COUNT CUDA device(s)"
fi

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    print_error "Data directory '$DATA_DIR' does not exist!"
    exit 1
fi

# Check required subdirectories
for split in train val; do
    if [ ! -d "$DATA_DIR/$split" ]; then
        print_error "Required directory '$DATA_DIR/$split' does not exist!"
        echo "Expected structure:"
        echo "$DATA_DIR/"
        echo "├── train/"
        echo "│   ├── real/"
        echo "│   └── fake/"
        echo "└── val/"
        echo "    ├── real/"
        echo "    └── fake/"
        exit 1
    fi
done

# Create output directory
mkdir -p "$(dirname "$SAVE_PATH")"

# Print configuration
print_info "Starting GenConViT Improved Training"
echo "==================== Configuration ===================="
echo "Data directory:               $DATA_DIR"
echo "Batch size:                   $BATCH_SIZE"
echo "Number of epochs:             $EPOCHS"
echo "Learning rate:                $LR"
echo "Weight decay:                 $WEIGHT_DECAY"
echo "Number of workers:            $NUM_WORKERS"
echo "Model save path:              $SAVE_PATH"
echo ""
echo "Model Architecture:"
echo "  AutoEncoder latent dim:     $AE_LATENT"
echo "  VAE latent dim:             $VAE_LATENT"
echo "  Dropout rate:               $DROPOUT_RATE"
echo ""
echo "Scheduler (ReduceLROnPlateau):"
echo "  LR reduction factor:        $LR_GAMMA"
echo "  Patience:                   $LR_PATIENCE epochs"
echo "  Minimum LR:                 $MIN_LR"
echo ""
echo "Early Stopping:"
echo "  Patience:                   $EARLY_STOPPING_PATIENCE epochs"
echo "  Min delta:                  $EARLY_STOPPING_MIN_DELTA"
echo ""
echo "Loss Weights:"
echo "  Classification:             $CLASSIFICATION_WEIGHT"
echo "  AE reconstruction:          $AE_WEIGHT"
echo "  VAE loss:                   $VAE_WEIGHT"
echo "  VAE beta (KL):              $VAE_BETA"
echo ""
echo "Other:"
echo "  Save every:                 $SAVE_EVERY epochs"
echo "  Log interval:               $LOG_INTERVAL batches"
echo "  Random seed:                $SEED"
echo "=============================================================="

# Check Python environment
if ! command -v python &> /dev/null; then
    print_error "Python is not installed or not in PATH"
    exit 1
fi

# Check required Python packages
print_info "Checking Python dependencies..."
python -c "import torch, torchvision, timm, sklearn, matplotlib, seaborn, pandas, numpy" 2>/dev/null || {
    print_error "Missing required Python packages. Please install:"
    echo "pip install torch torchvision timm scikit-learn matplotlib seaborn pandas numpy"
    exit 1
}

# Build command
CMD="python genconvit-v2/training/train_improved.py"
CMD="$CMD --data '$DATA_DIR'"
CMD="$CMD --batch-size $BATCH_SIZE"
CMD="$CMD --epochs $EPOCHS"
CMD="$CMD --lr $LR"
CMD="$CMD --weight-decay $WEIGHT_DECAY"
CMD="$CMD --num-workers $NUM_WORKERS"
CMD="$CMD --save-path '$SAVE_PATH'"
CMD="$CMD --ae-latent $AE_LATENT"
CMD="$CMD --vae-latent $VAE_LATENT"
CMD="$CMD --dropout-rate $DROPOUT_RATE"
CMD="$CMD --lr-gamma $LR_GAMMA"
CMD="$CMD --lr-patience $LR_PATIENCE"
CMD="$CMD --min-lr $MIN_LR"
CMD="$CMD --early-stopping-patience $EARLY_STOPPING_PATIENCE"
CMD="$CMD --early-stopping-min-delta $EARLY_STOPPING_MIN_DELTA"
CMD="$CMD --classification-weight $CLASSIFICATION_WEIGHT"
CMD="$CMD --ae-weight $AE_WEIGHT"
CMD="$CMD --vae-weight $VAE_WEIGHT"
CMD="$CMD --vae-beta $VAE_BETA"
CMD="$CMD --save-every $SAVE_EVERY"
CMD="$CMD --log-interval $LOG_INTERVAL"
CMD="$CMD --seed $SEED"

# Add optional arguments
if [ ! -z "$BALANCED_SAMPLING" ]; then
    CMD="$CMD $BALANCED_SAMPLING"
fi

if [ ! -z "$FRAMES_PER_VIDEO" ]; then
    CMD="$CMD $FRAMES_PER_VIDEO"
fi

if [ ! -z "$RESUME" ]; then
    CMD="$CMD $RESUME"
fi

echo ""
print_info "Starting training..."
echo "Command: $CMD"
echo ""

# Set environment variables for better performance
export OMP_NUM_THREADS=1
export CUDA_LAUNCH_BLOCKING=0

# Run training
start_time=$(date +%s)
eval $CMD
exit_code=$?
end_time=$(date +%s)
duration=$((end_time - start_time))

echo ""
if [ $exit_code -eq 0 ]; then
    print_success "Training completed successfully!"
    print_info "Training time: ${duration} seconds"
    if [ -f "$SAVE_PATH" ]; then
        print_success "Best model saved at: $SAVE_PATH"
    fi

    # Check for results file
    RESULTS_FILE="${SAVE_PATH//.pth/_results.json}"
    if [ -f "$RESULTS_FILE" ]; then
        print_info "Training results saved to: $RESULTS_FILE"
    fi

    echo ""
    print_info "Training Summary:"
    echo "  - Fixed model architecture with proper feature fusion ✓"
    echo "  - ReduceLROnPlateau scheduler for adaptive learning ✓"
    echo "  - Early stopping to prevent overfitting ✓"
    echo "  - Enhanced regularization and data augmentation ✓"
    echo "  - Comprehensive loss monitoring ✓"
    echo ""
    print_success "Ready for evaluation! Use ./evaluate.sh to test your model."
else
    print_error "Training failed with exit code $exit_code"
    exit $exit_code
fi
