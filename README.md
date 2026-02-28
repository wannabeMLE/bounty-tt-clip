# CLIP-ViT TTNN Implementation

Implementation of [CLIP-ViT-B/32](https://huggingface.co/openai/clip-vit-base-patch32) on Tenstorrent hardware using the TTNN API.

Built for [Tenstorrent Bounty #30870](https://github.com/tenstorrent/tt-metal/issues/30870).

## Overview

This project implements the full CLIP (Contrastive Language-Image Pre-training) model in TTNN, including:
- **Vision Encoder** — 12-layer ViT transformer (patch embeddings, multi-head attention, MLP, layer norms)
- **Text Encoder** — 12-layer transformer with causal masking
- **Similarity** — Cosine similarity with learned temperature for zero-shot classification

### Three Optimization Stages

| Stage | Memory | Math | Features | PCC Target |
|-------|--------|------|----------|------------|
| **Stage 1** | DRAM | HiFi4 | Baseline functional | > 0.99 |
| **Stage 2** | L1 | LoFi | GELU fusion, all 56 cores, program cache | > 0.98 |
| **Stage 3** | L1 | LoFi | SDPA, full fusion, program configs | > 0.97 |

## Project Structure

```
bounty-tt-clip/
├── README.md
├── requirements.txt
├── setup_ttsim.sh              # ttsim environment setup
│
├── clip_vit_ttnn/              # Core implementation
│   ├── tt/
│   │   ├── clip_model.py       # TTNN model (all 3 stages)
│   │   └── weight_loader.py    # Weight loading + config
│   ├── reference/
│   │   ├── torch_clip.py       # PyTorch reference implementation
│   │   └── modeling_clip_hf.py # HuggingFace model reference
│   └── demo/
│       ├── demo_clip.py        # Interactive demo
│       └── benchmark.py        # Performance benchmark
│
├── tests/                      # Validation & testing
│   ├── generate_golden.py      # Generate golden reference tensors
│   ├── validate_golden.py      # Per-layer validation against golden ref
│   ├── validate_multi_image.py # Multi-image prediction test
│   └── test_cpu.py             # CPU-only PyTorch reference test
│
```

## Current Status

**Stage 1: Complete and validated on ttsim**

| Test | Result |
|------|--------|
| Vision patch embeddings | PCC = 0.999999 |
| Text encoder (avg 3 texts) | PCC = 0.990 |
| Full pipeline logits | PCC = 0.999 |
| Multi-image predictions | 4/4 correct |
| Top-1 prediction | "a photo of a cat" @ 99.2% (matches PyTorch) |

> Vision encoder per-layer PCC degrades to ~0.965 due to manual softmax decomposition
> required by ttsim. Native `ttnn.softmax` on real hardware should resolve this.

## Quick Start

### Prerequisites

- [tt-metal](https://github.com/tenstorrent/tt-metal) built from source
- [ttsim](https://github.com/tenstorrent/ttsim) v1.4.0+ (or real Tenstorrent hardware)
- Python 3.10+ with tt-metal's virtual environment

### Setup

```bash
# Install dependencies into tt-metal's python_env
source $TT_METAL_HOME/python_env/bin/activate
pip install -r requirements.txt
```

### Run Validation

```bash
# Set environment
export TT_METAL_HOME=/path/to/tt-metal
export TT_METAL_SIMULATOR=/path/to/libttsim_wh.so  # only for ttsim
export TT_METAL_SLOW_DISPATCH_MODE=1
export ARCH_NAME=wormhole_b0

# Generate golden reference (CPU only, no hardware needed)
python tests/generate_golden.py

# Validate against golden reference
python tests/validate_golden.py --stage 1

# Test on multiple images
python tests/validate_multi_image.py
```

## Technical Notes

### ttsim Limitations
- `ttnn.softmax` not supported (uses SFPLOADMACRO) — manual decomposition used instead
- `ttnn.add` does not support subtile broadcast — causal mask pre-expanded to `[1, num_heads, S, S]`
- Performance benchmarks are not meaningful on the simulator

See [docs/validation_report.md](docs/validation_report.md) for detailed analysis.
