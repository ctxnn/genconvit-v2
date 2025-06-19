#!/bin/bash

# GenConViT Video Evaluation Launch Script
# Usage: ./evaluate.sh [options]

set -e

# Default values
MODEL_PATH=""
VIDEO_DIR=""
OUTPUT_DIR="./evaluation_results"
GROUND_TRUTH=""
FRAME_INTERVAL=5
MAX_FRAMES=50
BATCH_SIZE=32
DEVICE="auto"
TEMP_DIR="./temp_frames"
KEEP_FRAMES=false
VIDEO_EXTENSIONS=".mp4 .avi .mov .mkv .flv .wmv"

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
    echo "GenConViT Video Evaluation Script"
    echo ""
    echo "This script evaluates a trained GenConViT model on video files by:"
    echo "1. Extracting frames from videos"
    echo "2. Running predictions on frames"
    echo "3. Generating comprehensive evaluation reports with graphs"
    echo ""
    echo "Usage: $0 --model MODEL_PATH --video-dir VIDEO_DIR [options]"
    echo ""
    echo "Required Arguments:"
    echo "  --model PATH              Path to trained model checkpoint (.pth file)"
    echo "  --video-dir DIR           Directory containing video files to evaluate"
    echo ""
    echo "Optional Arguments:"
    echo "  --output-dir DIR          Output directory for results (default: $OUTPUT_DIR)"
    echo "  --ground-truth PATH       Ground truth file (CSV or JSON format)"
    echo "  --frame-interval N        Extract every Nth frame (default: $FRAME_INTERVAL)"
    echo "  --max-frames N            Maximum frames per video (default: $MAX_FRAMES)"
    echo "  --batch-size N            Batch size for prediction (default: $BATCH_SIZE)"
    echo "  --device DEVICE           Device: auto, cpu, cuda (default: $DEVICE)"
    echo "  --temp-dir DIR            Temporary directory for frames (default: $TEMP_DIR)"
    echo "  --keep-frames             Keep extracted frames after evaluation"
    echo "  --video-extensions EXTS   Video extensions to process (default: $VIDEO_EXTENSIONS)"
    echo "  --help, -h                Show this help message"
    echo ""
    echo "Ground Truth File Format:"
    echo "  CSV: video_name,label (0=fake, 1=real)"
    echo "  JSON: {\"video1\": 0, \"video2\": 1, ...}"
    echo ""
    echo "Examples:"
    echo "  # Basic evaluation"
    echo "  $0 --model ./models/best_model.pth --video-dir ./test_videos"
    echo ""
    echo "  # With ground truth for metrics"
    echo "  $0 --model ./models/best_model.pth --video-dir ./test_videos --ground-truth ./labels.csv"
    echo ""
    echo "  # Custom frame extraction"
    echo "  $0 --model ./models/best_model.pth --video-dir ./videos --frame-interval 10 --max-frames 30"
    echo ""
    echo "  # Memory-efficient evaluation"
    echo "  $0 --model ./models/best_model.pth --video-dir ./videos --batch-size 8 --device cpu"
    echo ""
    echo "Output Files:"
    echo "  - evaluation_report.json: Comprehensive evaluation metrics"
    echo "  - detailed_results.csv: Per-video prediction results"
    echo "  - detailed_predictions.json: Complete prediction data"
    echo "  - confusion_matrix.png: Confusion matrix (if ground truth provided)"
    echo "  - roc_curve.png: ROC curve (if ground truth provided)"
    echo "  - precision_recall_curve.png: PR curve (if ground truth provided)"
    echo "  - confidence_distribution.png: Confidence score distribution"
    echo "  - prediction_distribution.png: Prediction class distribution"
    echo ""
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --video-dir)
            VIDEO_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --ground-truth)
            GROUND_TRUTH="$2"
            shift 2
            ;;
        --frame-interval)
            FRAME_INTERVAL="$2"
            shift 2
            ;;
        --max-frames)
            MAX_FRAMES="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --temp-dir)
            TEMP_DIR="$2"
            shift 2
            ;;
        --keep-frames)
            KEEP_FRAMES=true
            shift
            ;;
        --video-extensions)
            VIDEO_EXTENSIONS="$2"
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

# Validate required arguments
if [ -z "$MODEL_PATH" ]; then
    print_error "Model path is required. Use --model to specify the model checkpoint."
    echo "Use --help for usage information"
    exit 1
fi

if [ -z "$VIDEO_DIR" ]; then
    print_error "Video directory is required. Use --video-dir to specify the directory."
    echo "Use --help for usage information"
    exit 1
fi

# Validate paths
if [ ! -f "$MODEL_PATH" ]; then
    print_error "Model file not found: $MODEL_PATH"
    exit 1
fi

if [ ! -d "$VIDEO_DIR" ]; then
    print_error "Video directory not found: $VIDEO_DIR"
    exit 1
fi

if [ ! -z "$GROUND_TRUTH" ] && [ ! -f "$GROUND_TRUTH" ]; then
    print_error "Ground truth file not found: $GROUND_TRUTH"
    exit 1
fi

# Check Python environment
if ! command -v python &> /dev/null; then
    print_error "Python is not installed or not in PATH"
    exit 1
fi

# Check required Python packages
print_info "Checking Python dependencies..."
python -c "import torch, torchvision, timm, cv2, PIL, numpy, pandas, matplotlib, seaborn, sklearn" 2>/dev/null || {
    print_error "Missing required Python packages. Please install:"
    echo "pip install torch torchvision timm opencv-python pillow numpy pandas matplotlib seaborn scikit-learn"
    exit 1
}

# Check CUDA availability if requested
if [ "$DEVICE" = "cuda" ]; then
    python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" 2>/dev/null || {
        print_warning "CUDA requested but not available. Falling back to CPU."
        DEVICE="cpu"
    }
fi

# Count video files
print_info "Scanning for video files..."
VIDEO_COUNT=0
for ext in $VIDEO_EXTENSIONS; do
    count=$(find "$VIDEO_DIR" -name "*$ext" -o -name "*${ext^^}" 2>/dev/null | wc -l)
    VIDEO_COUNT=$((VIDEO_COUNT + count))
done

if [ $VIDEO_COUNT -eq 0 ]; then
    print_error "No video files found in $VIDEO_DIR"
    print_info "Supported extensions: $VIDEO_EXTENSIONS"
    exit 1
fi

print_success "Found $VIDEO_COUNT video files"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Print configuration
print_info "Starting GenConViT Video Evaluation"
echo "==================== Configuration ===================="
echo "Model path:           $MODEL_PATH"
echo "Video directory:      $VIDEO_DIR"
echo "Output directory:     $OUTPUT_DIR"
echo "Ground truth:         ${GROUND_TRUTH:-"Not provided"}"
echo "Frame interval:       $FRAME_INTERVAL"
echo "Max frames per video: $MAX_FRAMES"
echo "Batch size:           $BATCH_SIZE"
echo "Device:               $DEVICE"
echo "Temporary directory:  $TEMP_DIR"
echo "Keep frames:          $KEEP_FRAMES"
echo "Video extensions:     $VIDEO_EXTENSIONS"
echo "Videos to process:    $VIDEO_COUNT"
echo "========================================================"

# Build command
CMD="python genconvit-v2/evaluation/predict_and_evaluate.py"
CMD="$CMD --model '$MODEL_PATH'"
CMD="$CMD --video-dir '$VIDEO_DIR'"
CMD="$CMD --output-dir '$OUTPUT_DIR'"
CMD="$CMD --frame-interval $FRAME_INTERVAL"
CMD="$CMD --max-frames $MAX_FRAMES"
CMD="$CMD --batch-size $BATCH_SIZE"
CMD="$CMD --device $DEVICE"
CMD="$CMD --temp-dir '$TEMP_DIR'"

if [ ! -z "$GROUND_TRUTH" ]; then
    CMD="$CMD --ground-truth '$GROUND_TRUTH'"
fi

if [ "$KEEP_FRAMES" = true ]; then
    CMD="$CMD --keep-frames"
fi

CMD="$CMD --video-extensions"
for ext in $VIDEO_EXTENSIONS; do
    CMD="$CMD $ext"
done

print_info "Running evaluation..."
echo "Command: $CMD"
echo ""

# Set environment variables for better performance
export OMP_NUM_THREADS=1

# Run evaluation
start_time=$(date +%s)
eval $CMD
exit_code=$?
end_time=$(date +%s)
duration=$((end_time - start_time))

echo ""
if [ $exit_code -eq 0 ]; then
    print_success "Evaluation completed successfully!"
    print_info "Processing time: ${duration} seconds"
    print_info "Results saved to: $OUTPUT_DIR"
    echo ""
    print_info "Generated files:"
    if [ -f "$OUTPUT_DIR/evaluation_report.json" ]; then
        echo "  ✓ evaluation_report.json - Main evaluation metrics"
    fi
    if [ -f "$OUTPUT_DIR/detailed_results.csv" ]; then
        echo "  ✓ detailed_results.csv - Per-video results"
    fi
    if [ -f "$OUTPUT_DIR/confusion_matrix.png" ]; then
        echo "  ✓ confusion_matrix.png - Confusion matrix"
    fi
    if [ -f "$OUTPUT_DIR/roc_curve.png" ]; then
        echo "  ✓ roc_curve.png - ROC curve"
    fi
    if [ -f "$OUTPUT_DIR/precision_recall_curve.png" ]; then
        echo "  ✓ precision_recall_curve.png - Precision-Recall curve"
    fi
    if [ -f "$OUTPUT_DIR/confidence_distribution.png" ]; then
        echo "  ✓ confidence_distribution.png - Confidence distribution"
    fi
    if [ -f "$OUTPUT_DIR/prediction_distribution.png" ]; then
        echo "  ✓ prediction_distribution.png - Prediction distribution"
    fi
    echo ""
    print_info "Open the output directory to view all results and visualizations."
else
    print_error "Evaluation failed with exit code $exit_code"
    exit $exit_code
fi
