# CLIP-ViT TTNN Implementation

Implementation of [CLIP-ViT-B/32](https://huggingface.co/openai/clip-vit-base-patch32) on Tenstorrent hardware using the TTNN API.

Built for [Tenstorrent Bounty #30870](https://github.com/tenstorrent/tt-metal/issues/30870).

## Overview

This project implements the full CLIP (Contrastive Language-Image Pre-training) model in TTNN, including:
- **Vision Encoder** — 12-layer ViT transformer (patch embeddings, multi-head attention, MLP, layer norms)
- **Text Encoder** — 12-layer transformer with causal masking
- **Similarity** — Cosine similarity with learned temperature for zero-shot classification

### Three Optimization Stages

| Stage | Memory | Math | Features |
|-------|--------|------|----------|
| **Stage 1** | DRAM | HiFi4 | Baseline functional |
| **Stage 2** | L1 | LoFi | GELU fusion, sharding, program cache |
| **Stage 3** | L1 | LoFi | SDPA, full fusion, program configs |

## Project Structure

```
bounty-tt-clip/
├── README.md
├── requirements.txt
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

**Stage 1: Complete and validated on N300 hardware**

| Test | Result |
|------|--------|
| Vision patch embeddings | PCC = 1.000000 |
| Vision block 0 output | PCC = 0.990871 |
| Full pipeline logits | PCC = 0.998843 |
| Multi-image predictions | 5/5 correct |
| Top-1 prediction | "a photo of a cat" @ 94.5% (matches PyTorch) |

## Quick Start

### Prerequisites

- Tenstorrent N150 or N300 hardware (Wormhole B0)
- Python 3.10+ with ttnn installed
- Dependencies: `pip install -r requirements.txt`

### Run Validation

```bash
# Generate golden reference (CPU only, no hardware needed)
python tests/generate_golden.py --skip_coco

# Validate against golden reference
python tests/validate_golden.py --stage 1 --golden golden_reference.pt

# Test on multiple images
python tests/validate_multi_image.py --stage 1
```

### Run Demo

```bash
python clip_vit_ttnn/demo/demo_clip.py --stage 1
```

## Hardware

- **Device:** Tenstorrent Wormhole B0 (N300, 2 chips)
- **Compute grid:** 8x7 = 56 cores
- **Architecture:** `Arch.WORMHOLE_B0`
