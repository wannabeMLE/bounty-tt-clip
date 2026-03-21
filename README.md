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
| **Stage 1** | DRAM | HiFi4 | Baseline functional (bfloat16 weights) |
| **Stage 2** | L1 interleaved | LoFi | bfloat8_b weights, softmax_in_place, program cache |
| **Stage 3** | L1 interleaved | LoFi | SDPA, fused QKV, LayerNorm fusion, program configs |

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
│       └── demo_clip.py        # Interactive demo
│
├── tests/                      # Validation & testing
│   ├── generate_golden.py      # Generate golden reference tensors
│   ├── validate_golden.py      # Per-layer validation against golden ref
│   ├── validate_multi_image.py # Multi-image prediction test
│   └── test_cpu.py             # CPU-only PyTorch reference test
│
```

## Current Status

**Stage 2: Complete and validated on N300 hardware**

### Accuracy (PCC vs PyTorch reference)

| Stage | Vision PCC | Text PCC | Logits PCC | Threshold |
|-------|-----------|----------|------------|-----------|
| Stage 1 | 0.9990 | 0.9999 | 0.9996 | >= 0.99 |
| Stage 2 | 0.9956 | 0.9993 | 0.9944 | >= 0.98 |

### Performance

| Metric | Stage 1 | Stage 2 |
|--------|---------|---------|
| Vision encoder (avg) | 7.7 ms | 6.9 ms |
| Text encoder (1 seq, avg) | 4.2 ms | 4.4 ms |
| Full pipeline | 19.1 ms | 20.6 ms |
| Vision speedup vs CPU | 6.85x | 8.06x |
| Throughput (vision) | 130.7 img/s | 145.9 img/s |
| Vision vs Stage 1 | — | 1.12x faster |

> **Note:** Benchmark uses 20 timed runs with 3 warmup iterations to ensure program cache is fully populated. Results are sensitive to warmup — insufficient warmup causes Stage 2 to appear slower due to additional L1 kernel compilation paths.

### Stage 2 Optimizations Applied

- **L1 interleaved memory** — activations and intermediates in L1 SRAM (vs DRAM in Stage 1)
- **LoFi math fidelity** — reduced-precision compute kernels
- **bfloat8_b weights** — 1 byte/param (vs 2 bytes in Stage 1)
- **softmax_in_place** — in-place softmax saves one tensor allocation per attention layer
- **Program cache** — enabled across all stages, eliminates recompilation

### Sharding vs Interleaving Analysis

We systematically evaluated all viable sharding strategies for CLIP-ViT-B/32's tensor geometry and determined that **L1 interleaved is optimal**.

**Why sharding doesn't help here:** CLIP-ViT-B/32 operates on small sequences — vision has 50 tokens (2 tile-rows after padding to 64), text has 77 tokens (3 tile-rows after padding to 96). The L1 interleaved matmul kernel already distributes work across all 56 cores internally. Explicit sharding restricts parallelism to the shard grid, which is limited by the number of tile-rows.

| Strategy | Result | Reason |
|----------|--------|--------|
| HEIGHT_SHARDED | 14x slower | Only 2-3 cores (limited by tile-rows) vs 56 cores interleaved |
| BLOCK_SHARDED (output) | Cannot run | Output tiles too wide (96 cols vision, 64 cols text) exceeds 8-col grid |
| BLOCK_SHARDED (input) | 1.2x–5.6x slower | Reshard overhead + reduced parallelism vs interleaved kernel |
| L1 INTERLEAVED | **Fastest** | Kernel uses full 56-core grid internally |

Full benchmark data with all grid configurations: [`results/stage2_sharding_investigation.md`](results/stage2_sharding_investigation.md)

### Fusion Analysis

| Fusion | Status | Detail |
|--------|--------|--------|
| Linear + bias | Done | Built-in `ttnn.linear` bias parameter |
| QuickGELU | Cannot fuse | CLIP uses `x * sigmoid(1.702x)` — no fused TTNN kernel exists. Standard GELU is mathematically different and causes PCC regression through 12 layers (0.999 per-op compounds to 0.964 end-to-end). |
| LayerNorm + residual add | Cannot fuse | No `ttnn.add_and_norm` or residual parameter in `ttnn.layer_norm`. These remain separate ops. |
| softmax_in_place | Done | Replaces `ttnn.softmax` — saves one allocation per attention layer |
| On-device patch embedding | Deferred to Stage 3 | fold+linear implementation exists but produces incorrect output (PCC -0.02). CPU conv2d used for Stages 1-2. |

### Performance Profiling

Per-layer timing is collected via host-side `time.perf_counter()` with `ttnn.synchronize_device()` barriers. Results in [`results/stage2_profile.txt`](results/stage2_profile.txt).

Tracy-based device profiling (`TT_METAL_DEVICE_PROFILER=1`) requires a Tracy-enabled build of tt-metal which is not available in this environment. The standard TT perf sheet format requires Tracy kernel-level profiling data.

**Per-layer breakdown (Stage 2 Vision, avg of 12 layers):**

| Component | Time | % of Layer |
|-----------|------|------------|
| Attention (total) | 0.29 ms | 43.1% |
| MLP (fc1 + QuickGELU + fc2) | 0.21 ms | 31.7% |
| LayerNorm (×2) | 0.11 ms | 16.9% |
| Residual adds (×2) | 0.06 ms | 8.3% |
| **Total per layer** | **0.67 ms** | 100% |

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

### Run Benchmark

```bash
python benchmark.py --stage 1 --output results/stage1_benchmark.md
python benchmark.py --stage 2 --output results/stage2_benchmark.md
```

### COCO Zero-Shot Validation

```bash
# 20 images (quick sanity check)
python tests/validate_coco.py --stage 2 --num_images 20

# Full COCO val2017 set (5000 images)
python tests/validate_coco.py --stage 2 --num_images 5000
```

### Run Demo

```bash
python clip_vit_ttnn/demo/demo_clip.py --stage 2
```

## Hardware

- **Device:** Tenstorrent Wormhole B0 (N300, 2 chips)
- **Compute grid:** 8x7 = 56 cores
- **Architecture:** `Arch.WORMHOLE_B0`
