#!/bin/bash
# =============================================================================
# Quick test runner for CLIP-ViT TTNN (ttsim / Wormhole simulator)
# =============================================================================
# Usage:
#   ./run_tests.sh              # Run all tests
#   ./run_tests.sh verify       # Just open/close simulated device
#   ./run_tests.sh quick        # Config + weight loading only
#   ./run_tests.sh stage1       # Stage 1 tests only
#   ./run_tests.sh demo         # Run the demo
#   ./run_tests.sh bench        # Run benchmark
# =============================================================================

set -e

# -------------------------------------------------------------------
# Environment setup for ttsim
# -------------------------------------------------------------------
if [ -z "$TT_METAL_HOME" ]; then
    echo "ERROR: TT_METAL_HOME is not set. Run setup_ttsim.sh first."
    exit 1
fi

export TT_METAL_SLOW_DISPATCH_MODE=1
export TT_METAL_DISABLE_SFPLOADMACRO=1
export ARCH_NAME=wormhole_b0

# Set SoC descriptor if available
SOC_DESC="$TT_METAL_HOME/tt_metal/soc_descriptors/wormhole_b0_80_arch.yaml"
if [ -f "$SOC_DESC" ]; then
    export TT_METAL_DEVICE_INFO="$SOC_DESC"
fi

# Activate tt-metal venv if not already active
if [ -z "$VIRTUAL_ENV" ]; then
    TT_VENV="$TT_METAL_HOME/python_env/bin/activate"
    if [ -f "$TT_VENV" ]; then
        source "$TT_VENV"
    else
        echo "WARNING: tt-metal venv not found. Using current Python."
    fi
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case "${1:-all}" in
    verify)
        echo "Verifying simulated device open/close..."
        python -c "
import ttnn
print('Opening simulated device...')
device = ttnn.open_device(device_id=0)
print(f'  Device: {device}')
ttnn.close_device(device)
print('Device opened and closed successfully.')
"
        ;;
    quick)
        echo "Running config and weight loading tests..."
        cd "$SCRIPT_DIR/clip_vit_ttnn"
        pytest tests/test_clip_ttnn.py -v -k "TestConfig or TestWeightLoading"
        ;;
    stage1)
        echo "Running Stage 1 tests..."
        cd "$SCRIPT_DIR/clip_vit_ttnn"
        pytest tests/test_clip_ttnn.py -v -k "stage1 or TestConfig or TestWeightLoading or TestEndToEnd"
        ;;
    stage2)
        echo "Running Stage 2 tests..."
        cd "$SCRIPT_DIR/clip_vit_ttnn"
        pytest tests/test_clip_ttnn.py -v -k "stage2"
        ;;
    stage3)
        echo "Running Stage 3 tests..."
        cd "$SCRIPT_DIR/clip_vit_ttnn"
        pytest tests/test_clip_ttnn.py -v -k "stage3"
        ;;
    demo)
        echo "Running demo (Stage 1)..."
        cd "$SCRIPT_DIR/clip_vit_ttnn"
        python demo/demo_clip.py --stage 1
        ;;
    demo2)
        echo "Running demo (Stage 2)..."
        cd "$SCRIPT_DIR/clip_vit_ttnn"
        python demo/demo_clip.py --stage 2
        ;;
    demo3)
        echo "Running demo (Stage 3)..."
        cd "$SCRIPT_DIR/clip_vit_ttnn"
        python demo/demo_clip.py --stage 3
        ;;
    bench)
        echo "Running full benchmark..."
        cd "$SCRIPT_DIR/clip_vit_ttnn"
        python demo/benchmark.py
        ;;
    all)
        echo "Running ALL tests..."
        cd "$SCRIPT_DIR/clip_vit_ttnn"
        pytest tests/test_clip_ttnn.py -v
        echo ""
        echo "Running demo..."
        python demo/demo_clip.py --stage 1
        ;;
    *)
        echo "Usage: $0 {verify|quick|stage1|stage2|stage3|demo|demo2|demo3|bench|all}"
        exit 1
        ;;
esac
