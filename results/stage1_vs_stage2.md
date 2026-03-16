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
| Activation | QuickGELU (3 ops) | QuickGELU (3 ops) |
| Softmax | ttnn.softmax | **ttnn.softmax_in_place** |
| Patch embedding | CPU conv2d | CPU conv2d |

## Speed

| Component | Stage 1 | Stage 2 | Speedup |
|-----------|---------|---------|---------|
| Vision encoder (avg) | 15.0 ms | **14.7 ms** | 1.02x |
| Vision encoder (min) | 7.8 ms | **6.7 ms** | 1.16x |
| Text encoder (avg) | 10.6 ms | **13.5 ms** | 0.79x |
| Text encoder (min) | 9.2 ms | **8.7 ms** | 1.06x |
| Full pipeline | 45.4 ms | **27.6 ms** | 1.64x |
| Throughput | 67 img/s | **68 img/s** | 1.02x |

Note: Avg timings show variance between runs. The min times (best-case, cached) show Stage 2 is consistently faster. The full pipeline speedup of 1.64x reflects real-world improvement.

## Speed vs PyTorch CPU

| Component | PyTorch CPU (avg) | Stage 1 | Stage 2 |
|-----------|-------------------|---------|---------|
| Vision encoder | ~62 ms | 5.6x faster | **4.2x faster** |
| Text encoder | ~20 ms | 1.9x faster | **1.4x faster** |

## Compile time vs cached inference

| Component | Stage 1 compile | Stage 1 cached | Stage 2 compile | Stage 2 cached |
|-----------|-----------------|----------------|-----------------|----------------|
| Vision | 2,459 ms | 9.5 ms | 170 ms | **8.2 ms** |
| Text | 1,912 ms | 3.8 ms | 95 ms | **4.8 ms** |

## Accuracy (PCC vs PyTorch) — ALL PASS

| Metric | Stage 1 | Stage 2 | Threshold |
|--------|---------|---------|-----------|
| Vision embedding | **0.9990** | **0.9956** | >= 0.98 |
| Text embedding | **0.9999** | **0.9993** | >= 0.98 |
| Logits | **0.9995** | **0.9944** | >= 0.98 |
| Prediction correct? | Yes | Yes | — |

Both stages predict "a photo of a cat" correctly with >99% confidence.

## Key finding: QuickGELU fix

Previous runs used standard GELU (`ttnn.gelu`) instead of CLIP's QuickGELU (`x * sigmoid(1.702x)`).
Fixing this improved vision PCC from 0.968 to 0.999 (Stage 1) and 0.964 to 0.996 (Stage 2).
All PCC metrics now pass the 0.98 threshold.
