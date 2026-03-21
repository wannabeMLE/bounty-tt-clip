# CLIP-ViT TTNN Benchmark — Stage 2

**Date:** 2026-03-21T00:22:57.514915
**Hardware:** Tenstorrent Wormhole B0 (N300)
**Model:** openai/clip-vit-base-patch32
**Runs:** 20 (after compile warmup)

## Latency

| Component | PyTorch CPU (avg) | TTNN (avg) | TTNN (min) | TTNN (stddev) | Speedup vs CPU |
|-----------|-------------------|------------|------------|---------------|----------------|
| Vision encoder | 55.3 ms | 6.9 ms | 6.4 ms | 1.01 ms | 8.06x |
| Text encoder (1 seq) | 28.5 ms | 4.4 ms | 4.1 ms | 0.19 ms | 6.48x |
| Full pipeline | — | 20.6 ms | — | — | — |

**Throughput:** 145.9 images/sec (vision encoder)

> **Note:** Text encoder benchmarked per single sequence. Full pipeline encodes 3 texts serially.

## Speedup vs Stage 1

| Component | Stage 1 (avg) | Stage 2 (avg) | Speedup |
|-----------|---------------|-------------|---------|
| Vision encoder | 7.7 ms | 6.9 ms | 1.12x |
| Text encoder | 4.2 ms | 4.4 ms | 0.97x |

## Compile vs Cached (Program Cache)

| Component | First run (compile) | Cached run | Compile overhead |
|-----------|--------------------:|----------:|-----------------:|
| Vision encoder | 172.7 ms | 6.7 ms | 166.0 ms |
| Text encoder | 98.5 ms | 4.3 ms | 94.2 ms |

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
