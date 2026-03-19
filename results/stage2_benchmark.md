# CLIP-ViT TTNN Benchmark — Stage 2

**Date:** 2026-03-19T13:45:25.334482
**Hardware:** Tenstorrent Wormhole B0 (N300)
**Model:** openai/clip-vit-base-patch32
**Runs:** 10 (after compile warmup)

## Latency

| Component | PyTorch CPU (avg) | TTNN (avg) | TTNN (min) | TTNN (stddev) | Speedup vs CPU |
|-----------|-------------------|------------|------------|---------------|----------------|
| Vision encoder | 58.5 ms | 7.6 ms | 6.3 ms | 2.24 ms | 7.66x |
| Text encoder (1 seq) | 18.1 ms | 4.4 ms | 4.1 ms | 0.33 ms | 4.15x |
| Full pipeline | — | 23.1 ms | — | — | — |

**Throughput:** 131.0 images/sec (vision encoder)

> **Note:** Text encoder benchmarked per single sequence. Full pipeline encodes 3 texts serially.

## Speedup vs Stage 1

| Component | Stage 1 (avg) | Stage 2 (avg) | Speedup |
|-----------|---------------|-------------|---------|
| Vision encoder | 13.7 ms | 7.6 ms | 1.80x |
| Text encoder | 10.0 ms | 4.4 ms | 2.29x |

## Compile vs Cached (Program Cache)

| Component | First run (compile) | Cached run | Compile overhead |
|-----------|--------------------:|----------:|-----------------:|
| Vision encoder | 176.2 ms | 15.9 ms | 160.2 ms |
| Text encoder | 98.4 ms | 4.1 ms | 94.2 ms |

## Accuracy (PCC >= 0.98)

| Metric | PCC | Status |
|--------|-----|--------|
| Vision embedding | 0.995625 | PASS |
| Text embedding | 0.999292 | PASS |
| Logits | 0.994353 | PASS |

## Prediction

| | Value |
|--|-------|
| Predicted | "a photo of a cat" |
| Correct | Yes |
| TTNN logits | [[24.25, 18.5, 18.375]] |
| PyTorch logits | [[24.570056915283203, 19.304899215698242, 18.48160743713379]] |

## Configuration

| Setting | Value |
|---------|-------|
| Memory | L1 interleaved |
| Math fidelity | LoFi |
| Weight dtype | bfloat8_b |
| Activation | QuickGELU (3 ops) |
| Compute grid | 8x7 (56 cores) |
| Dispatch | WORKER |
