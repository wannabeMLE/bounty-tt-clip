# Running CLIP-ViT TTNN on Koyeb (N300 Hardware)

## Hardware

- **Instance:** Koyeb GPU instance with Tenstorrent N300
- **Device:** Wormhole B0, 2 chips, 8x7 = 56 compute cores per chip
- **Kernel:** `6.1.62-tenstorrent-gpu`
- **CPU:** AMD EPYC 9254 24-Core, 32GB RAM

## Connecting

```bash
# 1. Start the Koyeb instance from the Koyeb dashboard

# 2. SSH into the instance
ssh root@tt-on-koyeb
ssh <koyeb-ssh-connection-string>
# (or use Koyeb CLI / web terminal)
```

## Environment

The Koyeb instance comes pre-configured:

- **Python:** `/opt/venv/bin/python` (Python 3.10.19)
- **ttnn:** Pre-installed in `/opt/venv/lib/python3.10/site-packages/ttnn/`
- **torch:** 2.10.0+cpu (CPU-only, used for reference/preprocessing)
- **No `TT_METAL_HOME` needed** — ttnn is pip-installed, not built from source
- **No `TT_METAL_SIMULATOR`** — this is real hardware, not ttsim
- **Fast dispatch mode** (default, no env vars needed)

## First-Time Setup

```bash
# Install project dependencies (into system dist-packages)
pip install -r requirements.txt

# Link system packages so the venv Python can see them
echo "/usr/local/lib/python3.10/dist-packages" > /opt/venv/lib/python3.10/site-packages/system-packages.pth

# Verify both ttnn and transformers work together
python -c "import ttnn; from transformers import CLIPModel; print('OK')"
```

## Running Tests

```bash
# Generate golden reference (CPU only, ~5 seconds)
python tests/generate_golden.py --skip_coco --output golden_reference.pt

# Validate Stage 1 against golden reference (~30 seconds)
python tests/validate_golden.py --stage 1 --golden golden_reference.pt

# Multi-image validation — 5 COCO images, PyTorch vs TTNN (~10 seconds)
python tests/validate_multi_image.py --stage 1
```

## Key Differences from ttsim

| | ttsim | Koyeb N300 |
|---|---|---|
| `TT_METAL_SIMULATOR` | Set to libttsim_wh.so | **Not set** (real HW) |
| `TT_METAL_SLOW_DISPATCH_MODE` | 1 | **Not set** (fast dispatch) |
| `ttnn.softmax` | Broken (SFPLOADMACRO) | **Works natively** |
| `_ON_TTSIM` flag in code | `True` | **`False`** |
| Subtile broadcast | Not supported | **Supported** |
| bfloat8_b dtype | May not work | **Supported** |
| fold op | May not work | **Supported** |
| Performance | Not meaningful | **Real timings** |

## Stage 1 Results on N300

```
Phase 1 (patch embeddings):    PCC = 1.000000  PASS
Phase 2 (block 0 internals):   PCC = 0.990871  PASS
Phase 5 (full pipeline logits): PCC = 0.998843  PASS
Multi-image predictions:        5/5 correct
Top-1: "a photo of a cat" @ 94.5% (matches PyTorch)
```

## Troubleshooting

### `ModuleNotFoundError: No module named 'transformers'`
The venv Python doesn't see system packages. Fix:
```bash
echo "/usr/local/lib/python3.10/dist-packages" > /opt/venv/lib/python3.10/site-packages/system-packages.pth
```

### Device not found
Check device exists:
```bash
ls /dev/tenstorrent*
python -c "import ttnn; print(ttnn.GetNumAvailableDevices())"  # Should print 2
```
