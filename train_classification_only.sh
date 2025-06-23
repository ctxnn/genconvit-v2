#!/bin/bash

# Classification-Only GenConViT Training Script
# This script runs pure classification training to eliminate NaN issues

set -e  # Exit on any error

# Default parameters - classification only
DATA_DIR="./data"
BATCH_SIZE=16
EPOCHS=50
LEARNING_RATE=0.001
EARLY_STOPPING_PATIENCE=10
NUM_WORKERS=4
SAVE_PATH="./models/genconvit_v2_classification_only.pth"
DROPOUT_RATE=0.3

echo "🎯 Starting Classification-Only GenConViT Training"
echo "================================================="
echo "This eliminates NaN issues by:"
echo "  ✓ No reconstruction losses"
echo "  ✓ Pure classification focus"
echo "  ✓ Stable Adam optimizer"
echo "  ✓ Lower dropout rate: $DROPOUT_RATE"
echo "  ✓ Learning rate: $LEARNING_RATE"
echo "================================================="

# Create models directory if it doesn't exist
mkdir -p models

# Run the classification-only training
python training/train_classification_only.py \
    --data "$DATA_DIR" \
    --batch-size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LEARNING_RATE \
    --dropout-rate $DROPOUT_RATE \
    --early-stopping-patience $EARLY_STOPPING_PATIENCE \
    --num-workers $NUM_WORKERS \
    --save-path "$SAVE_PATH" \
    --balanced-sampling \
    --save-every 5

echo ""
echo "🎉 Classification-only training completed!"
echo "Check the logs in training_classification_only.log"
echo "Model saved to: $SAVE_PATH"
echo ""
echo "If this works well, you can gradually add back reconstruction losses:"
echo "  python training/train_fixed.py \\"
echo "    --data $DATA_DIR \\"
echo "    --classification-weight 20.0 \\"
echo "    --ae-weight 0.001 \\"
echo "    --vae-weight 0.001"
