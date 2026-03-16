# CLIP-ViT TTNN Benchmark — Stage 1

**Date:** 2026-03-16T19:51:08.020148
**Hardware:** Tenstorrent Wormhole B0 (N300)
**Model:** openai/clip-vit-base-patch32
**Runs:** 10 (after compile warmup)

## Latency

| Component | PyTorch CPU (avg) | TTNN (avg) | TTNN (min) | TTNN (stddev) | Speedup vs CPU |
|-----------|-------------------|------------|------------|---------------|----------------|
| Vision encoder | 49.3 ms | 13.7 ms | 11.6 ms | 3.92 ms | 3.59x |
| Text encoder (1 seq) | 16.7 ms | 10.0 ms | 9.9 ms | 0.08 ms | 1.67x |
| Full pipeline | — | 42.8 ms | — | — | — |

**Throughput:** 72.9 images/sec (vision encoder)

> **Note:** Text encoder benchmarked per single sequence. Full pipeline encodes 3 texts serially.

## Compile vs Cached (Program Cache)

| Component | First run (compile) | Cached run | Compile overhead |
|-----------|--------------------:|----------:|-----------------:|
| Vision encoder | 176.7 ms | 12.1 ms | 164.6 ms |
| Text encoder | 132.1 ms | 10.6 ms | 121.5 ms |

## Accuracy (PCC >= 0.99)

| Metric | PCC | Status |
|--------|-----|--------|
| Vision embedding | 0.999014 | PASS |
| Text embedding | 0.999933 | PASS |
| Logits | 0.999548 | PASS |

## Prediction

| | Value |
|--|-------|
| Predicted | "a photo of a cat" |
| Correct | Yes |
| TTNN logits | [[24.375, 19.0, 18.375]] |
| PyTorch logits | [[24.570056915283203, 19.304899215698242, 18.48160743713379]] |

## Configuration

| Setting | Value |
|---------|-------|
| Memory | DRAM interleaved |
| Math fidelity | HiFi4 |
| Weight dtype | bfloat16 |
| Activation | QuickGELU (3 ops) |
| Compute grid | 8x7 (56 cores) |
| Dispatch | WORKER |
