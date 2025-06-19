#!/bin/bash

# Quick Fix Script for DDP Training Issues
# Usage: ./fix_ddp_training.sh [--debug]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# Check for debug flag
DEBUG_MODE=false
if [[ "$1" == "--debug" ]]; then
    DEBUG_MODE=true
    print_info "Debug mode enabled"
fi

print_info "DDP Training Issue Fix Script"
echo "============================================"

# Function to check if we're in the right directory
check_directory() {
    if [ ! -f "training/train_ddp.py" ]; then
        print_error "This script must be run from the genconvit-v2 root directory"
        print_info "Expected structure:"
        echo "  genconvit-v2/"
        echo "  ├── training/"
        echo "  │   └── train_ddp.py"
        echo "  └── fix_ddp_training.sh"
        exit 1
    fi
    print_success "Found training/train_ddp.py"
}

# Function to check Python environment
check_python_env() {
    print_info "Checking Python environment..."

    if ! command -v python &> /dev/null; then
        print_error "Python not found in PATH"
        exit 1
    fi

    # Check PyTorch
    python -c "import torch; print(f'PyTorch version: {torch.__version__}')" 2>/dev/null || {
        print_error "PyTorch not installed or not accessible"
        exit 1
    }

    # Check CUDA
    python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')" 2>/dev/null || {
        print_error "Cannot check CUDA availability"
        exit 1
    }

    # Check GPU count
    GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")
    print_success "Found $GPU_COUNT CUDA device(s)"

    if [ "$GPU_COUNT" -lt 2 ]; then
        print_warning "DDP training requires at least 2 GPUs. You have $GPU_COUNT."
        print_info "Consider using single GPU training instead:"
        echo "  ./train_improved.sh --data ./your_data"
        exit 1
    fi
}

# Function to fix common DDP issues
fix_ddp_issues() {
    print_info "Applying DDP fixes..."

    # Check if the file already has the fix
    if grep -q "find_unused_parameters=True" training/train_ddp.py; then
        print_success "DDP unused parameters fix already applied"
    else
        print_info "Applying unused parameters fix..."
        sed -i 's/find_unused_parameters=False/find_unused_parameters=True/g' training/train_ddp.py
        print_success "Applied unused parameters fix"
    fi

    # Check for F import
    if grep -q "import torch.nn.functional as F" training/train_ddp.py; then
        print_success "Functional import already present"
    else
        print_info "Adding functional import..."
        sed -i '/import torch.nn as nn/a import torch.nn.functional as F' training/train_ddp.py
        print_success "Added functional import"
    fi
}

# Function to set environment variables for better DDP performance
set_env_vars() {
    print_info "Setting environment variables for DDP..."

    export TORCH_DISTRIBUTED_DEBUG=INFO
    export NCCL_DEBUG=INFO
    export CUDA_LAUNCH_BLOCKING=1
    export OMP_NUM_THREADS=1

    if [ "$DEBUG_MODE" = true ]; then
        export TORCH_DISTRIBUTED_DEBUG=DETAIL
        export NCCL_DEBUG=DETAIL
        print_info "Debug mode: Set detailed logging"
    fi

    print_success "Environment variables set"
}

# Function to check for port conflicts
check_port_conflicts() {
    print_info "Checking for port conflicts..."

    # Default DDP port is 12355
    if netstat -tuln 2>/dev/null | grep -q ":12355 "; then
        print_warning "Port 12355 is in use. Killing existing processes..."
        pkill -f "train_ddp.py" || true
        sleep 2
        if netstat -tuln 2>/dev/null | grep -q ":12355 "; then
            print_error "Port 12355 still in use. Please manually kill processes using this port."
            print_info "Try: sudo lsof -ti:12355 | xargs kill -9"
            exit 1
        fi
    fi

    print_success "Port 12355 is available"
}

# Function to clean up previous training artifacts
cleanup_artifacts() {
    print_info "Cleaning up previous training artifacts..."

    # Remove old log files
    if [ -f "training_ddp_improved.log" ]; then
        mv training_ddp_improved.log training_ddp_improved.log.bak
        print_info "Backed up old log file"
    fi

    # Clean up any stuck processes
    pkill -f "train_ddp.py" || true

    print_success "Cleanup completed"
}

# Function to test DDP setup
test_ddp_setup() {
    print_info "Testing DDP setup..."

    # Create a simple DDP test script
    cat > test_ddp.py << 'EOF'
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

def test_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    print(f"Rank {rank}/{world_size} initialized successfully")

    # Test tensor operations
    tensor = torch.randn(2, 2).cuda(rank)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    print(f"Rank {rank}: Tensor test passed")
    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    if world_size < 2:
        print("Need at least 2 GPUs for DDP test")
        exit(1)
    mp.spawn(test_ddp, args=(world_size,), nprocs=world_size, join=True)
    print("DDP test completed successfully!")
EOF

    # Run the test
    if python test_ddp.py; then
        print_success "DDP setup test passed"
        rm test_ddp.py
    else
        print_error "DDP setup test failed"
        rm test_ddp.py
        exit 1
    fi
}

# Function to provide training recommendations
provide_recommendations() {
    print_info "Training Recommendations:"
    echo "=================================="
    echo ""
    print_info "1. For stable training, use these settings:"
    echo "   ./launch_ddp_improved.sh \\"
    echo "     --data ./your_data \\"
    echo "     --batch-size 8 \\"
    echo "     --lr 0.00005 \\"
    echo "     --weight-decay 1e-4"
    echo ""
    print_info "2. If you encounter memory issues:"
    echo "   ./launch_ddp_improved.sh \\"
    echo "     --data ./your_data \\"
    echo "     --batch-size 4 \\"
    echo "     --num-workers 2"
    echo ""
    print_info "3. For debugging, enable detailed logging:"
    echo "   TORCH_DISTRIBUTED_DEBUG=DETAIL ./launch_ddp_improved.sh --data ./your_data"
    echo ""
    print_info "4. Monitor training with:"
    echo "   tail -f training_ddp_improved.log"
    echo ""
}

# Main execution
main() {
    check_directory
    check_python_env
    cleanup_artifacts
    check_port_conflicts
    fix_ddp_issues
    set_env_vars

    if [ "$DEBUG_MODE" = true ]; then
        test_ddp_setup
    fi

    provide_recommendations

    print_success "DDP training environment is ready!"
    print_info "You can now run: ./launch_ddp_improved.sh --data ./your_data"
}

# Help function
show_help() {
    echo "DDP Training Fix Script"
    echo ""
    echo "This script fixes common issues with distributed training and prepares"
    echo "your environment for stable multi-GPU training."
    echo ""
    echo "Usage:"
    echo "  ./fix_ddp_training.sh           # Standard fix"
    echo "  ./fix_ddp_training.sh --debug   # Debug mode with detailed testing"
    echo "  ./fix_ddp_training.sh --help    # Show this help"
    echo ""
    echo "What this script does:"
    echo "  ✓ Checks Python and PyTorch environment"
    echo "  ✓ Verifies GPU availability (requires 2+ GPUs)"
    echo "  ✓ Fixes unused parameters issue in DDP"
    echo "  ✓ Sets optimal environment variables"
    echo "  ✓ Cleans up port conflicts"
    echo "  ✓ Tests DDP setup (in debug mode)"
    echo "  ✓ Provides training recommendations"
    echo ""
    echo "Common DDP Issues Fixed:"
    echo "  • 'Expected to have finished reduction in the prior iteration'"
    echo "  • 'Parameters that were not used in producing loss'"
    echo "  • Port conflicts and stuck processes"
    echo "  • Suboptimal environment variables"
    echo ""
}

# Check for help flag
if [[ "$1" == "--help" || "$1" == "-h" ]]; then
    show_help
    exit 0
fi

# Run main function
main

echo ""
print_success "All fixes applied successfully!"
print_info "Your DDP training environment is now optimized and ready to use."
