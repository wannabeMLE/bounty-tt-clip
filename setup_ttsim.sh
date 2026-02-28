#!/bin/bash
# =============================================================================
# CLIP-ViT TTNN - ttsim (Wormhole Simulator) Environment Setup
# =============================================================================
# Prerequisites:
#   1. tt-metal built from source (https://github.com/tenstorrent/tt-metal)
#   2. ttsim v1.4.0+ installed (https://github.com/tenstorrent/ttsim)
#   3. TT_METAL_HOME set to your tt-metal build directory
#   4. TT_METAL_SIMULATOR pointing to libttsim_wh.so
#
# Usage:
#   chmod +x setup_ttsim.sh
#   ./setup_ttsim.sh
# =============================================================================

set -e

echo "============================================"
echo "  CLIP-ViT TTNN - ttsim Environment Setup"
echo "  Target: Wormhole B0 (simulated)"
echo "============================================"

# -------------------------------------------------------------------
# 1. Check TT_METAL_HOME
# -------------------------------------------------------------------
echo ""
echo "[1/6] Checking TT_METAL_HOME..."
if [ -z "$TT_METAL_HOME" ]; then
    echo "  ERROR: TT_METAL_HOME is not set."
    echo "  Set it to your tt-metal source/build directory, e.g.:"
    echo "    export TT_METAL_HOME=~/tt-metal"
    exit 1
fi

if [ ! -d "$TT_METAL_HOME" ]; then
    echo "  ERROR: TT_METAL_HOME=$TT_METAL_HOME does not exist."
    exit 1
fi

if [ ! -f "$TT_METAL_HOME/build/lib/libtt_metal.so" ] && [ ! -f "$TT_METAL_HOME/build/lib/libdevice.so" ]; then
    echo "  WARNING: tt-metal build artifacts not found in $TT_METAL_HOME/build/lib/"
    echo "  Make sure tt-metal is fully built (./build_metal.sh)."
fi

echo "  TT_METAL_HOME=$TT_METAL_HOME"

# -------------------------------------------------------------------
# 2. Check TT_METAL_SIMULATOR (ttsim)
# -------------------------------------------------------------------
echo ""
echo "[2/6] Checking ttsim simulator library..."
if [ -z "$TT_METAL_SIMULATOR" ]; then
    echo "  ERROR: TT_METAL_SIMULATOR is not set."
    echo "  Set it to the path of libttsim_wh.so, e.g.:"
    echo "    export TT_METAL_SIMULATOR=/path/to/ttsim/lib/libttsim_wh.so"
    exit 1
fi

if [ ! -f "$TT_METAL_SIMULATOR" ]; then
    echo "  ERROR: TT_METAL_SIMULATOR=$TT_METAL_SIMULATOR does not exist."
    echo "  Download ttsim from: https://github.com/tenstorrent/ttsim/releases"
    exit 1
fi

echo "  TT_METAL_SIMULATOR=$TT_METAL_SIMULATOR"

# -------------------------------------------------------------------
# 3. Set required environment variables
# -------------------------------------------------------------------
echo ""
echo "[3/6] Setting environment variables..."

export TT_METAL_SLOW_DISPATCH_MODE=1
export TT_METAL_DISABLE_SFPLOADMACRO=1
export ARCH_NAME=wormhole_b0

SOC_DESC="$TT_METAL_HOME/tt_metal/soc_descriptors/wormhole_b0_80_arch.yaml"
if [ -f "$SOC_DESC" ]; then
    export TT_METAL_DEVICE_INFO="$SOC_DESC"
    echo "  TT_METAL_DEVICE_INFO=$TT_METAL_DEVICE_INFO"
else
    echo "  WARNING: SoC descriptor not found at $SOC_DESC"
    echo "  TT_METAL_DEVICE_INFO not set — tt-metal may use a default."
fi

echo "  TT_METAL_SLOW_DISPATCH_MODE=1"
echo "  ARCH_NAME=wormhole_b0"

# -------------------------------------------------------------------
# 4. Activate tt-metal Python environment
# -------------------------------------------------------------------
echo ""
echo "[4/6] Activating tt-metal Python environment..."

TT_VENV="$TT_METAL_HOME/python_env/bin/activate"
if [ -f "$TT_VENV" ]; then
    source "$TT_VENV"
    echo "  Activated: $(which python)"
else
    echo "  WARNING: tt-metal venv not found at $TT_VENV"
    echo "  Using current Python: $(which python)"
fi

# -------------------------------------------------------------------
# 5. Install project dependencies
# -------------------------------------------------------------------
echo ""
echo "[5/6] Installing project dependencies..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pip install -r "$SCRIPT_DIR/requirements.txt" -q
echo "  Done."

# -------------------------------------------------------------------
# 6. Verify installation
# -------------------------------------------------------------------
echo ""
echo "[6/6] Verifying installation..."

python -c "
import sys
print('Python:', sys.version)

checks = []

# torch
try:
    import torch
    checks.append(('torch', torch.__version__, True))
except ImportError:
    checks.append(('torch', 'MISSING', False))

# transformers
try:
    import transformers
    checks.append(('transformers', transformers.__version__, True))
except ImportError:
    checks.append(('transformers', 'MISSING', False))

# ttnn
try:
    import ttnn
    ver = getattr(ttnn, '__version__', 'installed')
    checks.append(('ttnn', ver, True))
except ImportError:
    checks.append(('ttnn', 'MISSING', False))

# PIL
try:
    from PIL import Image
    import PIL
    checks.append(('Pillow', PIL.__version__, True))
except ImportError:
    checks.append(('Pillow', 'MISSING', False))

# datasets
try:
    import datasets
    checks.append(('datasets', datasets.__version__, True))
except ImportError:
    checks.append(('datasets', 'MISSING', False))

# pytest
try:
    import pytest
    checks.append(('pytest', pytest.__version__, True))
except ImportError:
    checks.append(('pytest', 'MISSING', False))

# scipy
try:
    import scipy
    checks.append(('scipy', scipy.__version__, True))
except ImportError:
    checks.append(('scipy', 'MISSING', False))

print()
print('Package Versions:')
print('-' * 40)
all_ok = True
for name, ver, ok in checks:
    status = 'OK' if ok else 'MISSING'
    print(f'  [{status:7s}] {name:20s} {ver}')
    if not ok:
        all_ok = False

print()
if all_ok:
    print('All packages installed successfully!')
else:
    print('WARNING: Some packages are missing. Check errors above.')
"

# -------------------------------------------------------------------
# Try to open a simulated TT device
# -------------------------------------------------------------------
echo ""
echo "Attempting to open simulated Wormhole device..."
python -c "
try:
    import ttnn
    device = ttnn.open_device(device_id=0)
    print('  SUCCESS: Simulated TT device opened!')
    print(f'  Device: {device}')
    ttnn.close_device(device)
    print('  Device closed.')
except Exception as e:
    print(f'  FAILED: Could not open simulated device: {e}')
    print('  Check that TT_METAL_SIMULATOR and TT_METAL_SLOW_DISPATCH_MODE are set.')
" 2>&1

echo ""
echo "============================================"
echo "  Setup complete!"
echo ""
echo "  Required env vars (add to your shell profile):"
echo "    export TT_METAL_HOME=$TT_METAL_HOME"
echo "    export TT_METAL_SIMULATOR=$TT_METAL_SIMULATOR"
echo "    export TT_METAL_SLOW_DISPATCH_MODE=1"
echo "    export TT_METAL_DISABLE_SFPLOADMACRO=1"
echo "    export ARCH_NAME=wormhole_b0"
echo ""
echo "  To run tests:"
echo "    ./run_tests.sh verify   # Quick device open/close"
echo "    ./run_tests.sh quick    # Config + weight tests"
echo "    ./run_tests.sh stage1   # Stage 1 validation"
echo ""
echo "  NOTE: ttsim is much slower than real hardware."
echo "  Full model inference may take several minutes."
echo "============================================"
