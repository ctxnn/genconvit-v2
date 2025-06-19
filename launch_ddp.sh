#!/bin/bash

# GenConViT Distributed Training Launch Script
# Usage: ./launch_ddp.sh [options]

set -e

# Default values
DATA_DIR="./1000_videos_combined"
BATCH_SIZE=16
EPOCHS=50
LR=0.0001
NUM_WORKERS=4
SAVE_PATH="./models/genconvit_ddp_best.pth"
WORLD_SIZE=-1  # Use all available GPUs
INPUT_SIZE=224
WEIGHT_DECAY=1e-5
BETA=1.0
LR_STEP=15
LR_GAMMA=0.5
SAVE_EVERY=5
SEED=42

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
        --world-size)
            WORLD_SIZE="$2"
            shift 2
            ;;
        --resume)
            RESUME="$2"
            shift 2
            ;;
        --help|-h)
            echo "GenConViT Distributed Training Launch Script"
            echo ""
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --data DIR              Data directory (default: $DATA_DIR)"
            echo "  --batch-size N          Batch size per GPU (default: $BATCH_SIZE)"
            echo "  --epochs N              Number of epochs (default: $EPOCHS)"
            echo "  --lr RATE               Learning rate (default: $LR)"
            echo "  --num-workers N         Number of workers per GPU (default: $NUM_WORKERS)"
            echo "  --save-path PATH        Model save path (default: $SAVE_PATH)"
            echo "  --world-size N          Number of GPUs (-1 for all, default: $WORLD_SIZE)"
            echo "  --resume PATH           Resume from checkpoint"
            echo "  --help, -h              Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                                          # Use all defaults"
            echo "  $0 --batch-size 8 --epochs 30              # Custom batch size and epochs"
            echo "  $0 --world-size 2 --lr 0.0005              # Use 2 GPUs with custom LR"
            echo "  $0 --resume ./models/checkpoint.pth        # Resume training"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if CUDA is available
if ! python -c "import torch; print('CUDA available:', torch.cuda.is_available())"; then
    echo "Error: CUDA is not available. Please ensure you have PyTorch with CUDA support installed."
    exit 1
fi

# Check GPU count
GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")
echo "Available GPUs: $GPU_COUNT"

if [ "$GPU_COUNT" -eq 0 ]; then
    echo "Error: No CUDA devices found!"
    exit 1
fi

# Set world size to GPU count if -1
if [ "$WORLD_SIZE" -eq -1 ]; then
    WORLD_SIZE=$GPU_COUNT
fi

if [ "$WORLD_SIZE" -gt "$GPU_COUNT" ]; then
    echo "Warning: Requested world size ($WORLD_SIZE) is greater than available GPUs ($GPU_COUNT)"
    echo "Setting world size to $GPU_COUNT"
    WORLD_SIZE=$GPU_COUNT
fi

# Create output directory
mkdir -p "$(dirname "$SAVE_PATH")"

# Print configuration
echo "==================== Training Configuration ===================="
echo "Data directory:       $DATA_DIR"
echo "Batch size per GPU:   $BATCH_SIZE"
echo "Total batch size:     $((BATCH_SIZE * WORLD_SIZE))"
echo "Number of epochs:     $EPOCHS"
echo "Learning rate:        $LR"
echo "Number of GPUs:       $WORLD_SIZE"
echo "Workers per GPU:      $NUM_WORKERS"
echo "Input size:           ${INPUT_SIZE}x${INPUT_SIZE}"
echo "Weight decay:         $WEIGHT_DECAY"
echo "VAE loss weight:      $BETA"
echo "LR step size:         $LR_STEP"
echo "LR decay factor:      $LR_GAMMA"
echo "Save every:           $SAVE_EVERY epochs"
echo "Model save path:      $SAVE_PATH"
echo "Random seed:          $SEED"
if [ ! -z "$RESUME" ]; then
    echo "Resume from:          $RESUME"
fi
echo "=============================================================="

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory '$DATA_DIR' does not exist!"
    exit 1
fi

# Check required subdirectories
for split in train val; do
    if [ ! -d "$DATA_DIR/$split" ]; then
        echo "Error: Required directory '$DATA_DIR/$split' does not exist!"
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

# Build command
CMD="python genconvit-v2/training/train_ddp.py"
CMD="$CMD --data '$DATA_DIR'"
CMD="$CMD --batch-size $BATCH_SIZE"
CMD="$CMD --epochs $EPOCHS"
CMD="$CMD --lr $LR"
CMD="$CMD --num-workers $NUM_WORKERS"
CMD="$CMD --save-path '$SAVE_PATH'"
CMD="$CMD --world-size $WORLD_SIZE"
CMD="$CMD --input-size $INPUT_SIZE"
CMD="$CMD --weight-decay $WEIGHT_DECAY"
CMD="$CMD --beta $BETA"
CMD="$CMD --lr-step $LR_STEP"
CMD="$CMD --lr-gamma $LR_GAMMA"
CMD="$CMD --save-every $SAVE_EVERY"
CMD="$CMD --seed $SEED"

if [ ! -z "$RESUME" ]; then
    if [ ! -f "$RESUME" ]; then
        echo "Error: Resume checkpoint '$RESUME' does not exist!"
        exit 1
    fi
    CMD="$CMD --resume '$RESUME'"
fi

echo ""
echo "Starting training..."
echo "Command: $CMD"
echo ""

# Set environment variables for better performance
export OMP_NUM_THREADS=1
export CUDA_LAUNCH_BLOCKING=0

# Run training
eval $CMD

echo ""
echo "Training completed!"
if [ -f "$SAVE_PATH" ]; then
    echo "Best model saved at: $SAVE_PATH"
else
    echo "Warning: Best model file not found at expected location: $SAVE_PATH"
fi
