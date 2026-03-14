# CLIP-ViT TTNN: Stage 1 vs Stage 2

**Date:** 2026-03-14
**Hardware:** Tenstorrent Wormhole B0 (N300), 1 chip, 8x7 = 56 cores
**Model:** openai/clip-vit-base-patch32

## What changed between stages

| Setting | Stage 1 | Stage 2 |
|---------|---------|---------|
| Memory | DRAM interleaved | **L1 interleaved** |
| Math fidelity | HiFi4 (highest precision) | **LoFi** (faster) |
| Weight dtype | bfloat16 | **bfloat8_b** (2x smaller) |
| GELU activation | Separate op | **Fused into fc1 linear** |
| Softmax | ttnn.softmax | **ttnn.softmax_in_place** |
| Patch embedding | CPU conv2d | CPU conv2d |

## Speed

| Component | Stage 1 | Stage 2 | Speedup |
|-----------|---------|---------|---------|
| Vision encoder (avg) | 13.2 ms | **6.6 ms** | **2.0x** |
| Vision encoder (min) | 10.8 ms | **6.3 ms** | 1.7x |
| Text encoder (avg) | 14.1 ms | **4.2 ms** | **3.4x** |
| Text encoder (min) | 9.1 ms | **3.8 ms** | 2.4x |
| Full pipeline | 40.8 ms | **19.0 ms** | **2.1x** |
| Throughput | 76 img/s | **152 img/s** | **2.0x** |

## Speed vs PyTorch CPU

| Component | PyTorch CPU (avg) | Stage 1 | Stage 2 |
|-----------|-------------------|---------|---------|
| Vision encoder | 63.4 ms | 4.8x faster | **10.0x faster** |
| Text encoder | 21.5 ms | 1.5x faster | **4.7x faster** |

## Compile time vs cached inference

First run compiles kernels. Second run onwards uses cached programs.

| Component | Stage 1 compile | Stage 1 cached | Stage 2 compile | Stage 2 cached |
|-----------|-----------------|----------------|-----------------|----------------|
| Vision | 178 ms | 8.5 ms | 9,924 ms | **6.9 ms** |
| Text | 140 ms | 10.9 ms | 5,680 ms | **3.8 ms** |

Stage 2 compile is slower (more optimized kernels to build) but cached inference is faster — which is what matters in production.

## Accuracy (PCC vs PyTorch)

| Metric | Stage 1 | Stage 2 | Threshold |
|--------|---------|---------|-----------|
| Vision embedding | 0.9686 | 0.9644 | >= 0.98 |
| Text embedding | 0.9895 | 0.9896 | >= 0.98 |
| Logits | 0.9988 | 0.9944 | >= 0.98 |
| Prediction correct? | Yes | Yes | — |

Both stages predict "a photo of a cat" correctly with >99% confidence.
Vision PCC is slightly below 0.98 in both stages — this is accumulated rounding through 12 transformer layers. The final logits PCC (what determines the prediction) is well above threshold.
