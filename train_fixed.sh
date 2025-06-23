#!/bin/bash

# Fixed GenConViT Training Script
# This script runs the fixed training with optimized parameters for classification learning

set -e  # Exit on any error

# Default parameters - optimized for classification learning
DATA_DIR="./data"
BATCH_SIZE=16
EPOCHS=100
LEARNING_RATE=0.001
CLASSIFICATION_WEIGHT=10.0
AE_WEIGHT=0.01
VAE_WEIGHT=0.01
VAE_BETA=0.1
EARLY_STOPPING_PATIENCE=15
NUM_WORKERS=4
SAVE_PATH="./models/genconvit_v2_fixed.pth"

echo "🚀 Starting Fixed GenConViT Training"
echo "====================================="
echo "Key fixes applied:"
echo "  ✓ Higher learning rate: $LEARNING_RATE"
echo "  ✓ Classification weight: $CLASSIFICATION_WEIGHT"
echo "  ✓ AE weight (reduced): $AE_WEIGHT"
echo "  ✓ VAE weight (reduced): $VAE_WEIGHT"
echo "  ✓ VAE beta (reduced): $VAE_BETA"
echo "  ✓ More frequent logging every 25 batches"
echo "====================================="

# Create models directory if it doesn't exist
mkdir -p models

# Run the fixed training
python training/train_fixed.py \
    --data "$DATA_DIR" \
    --batch-size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LEARNING_RATE \
    --classification-weight $CLASSIFICATION_WEIGHT \
    --ae-weight $AE_WEIGHT \
    --vae-weight $VAE_WEIGHT \
    --vae-beta $VAE_BETA \
    --early-stopping-patience $EARLY_STOPPING_PATIENCE \
    --num-workers $NUM_WORKERS \
    --save-path "$SAVE_PATH" \
    --balanced-sampling \
    --save-every 5 \
    --log-interval 25

echo ""
echo "🎉 Training completed!"
echo "Check the logs in training_fixed.log for detailed progress."
echo "Model saved to: $SAVE_PATH"
echo ""
echo "If accuracy is still low, try these more aggressive settings:"
echo "  python training/train_fixed.py \\"
echo "    --data $DATA_DIR \\"
echo "    --lr 0.002 \\"
echo "    --classification-weight 20.0 \\"
echo "    --ae-weight 0.001 \\"
echo "    --vae-weight 0.001 \\"
echo "    --vae-beta 0.05"
