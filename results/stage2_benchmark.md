# CLIP-ViT TTNN Benchmark — Stage 2

**Date:** 2026-03-16T19:52:02.018553
**Hardware:** Tenstorrent Wormhole B0 (N300)
**Model:** openai/clip-vit-base-patch32
**Runs:** 10 (after compile warmup)

## Latency

| Component | PyTorch CPU (avg) | TTNN (avg) | TTNN (min) | TTNN (stddev) | Speedup vs CPU |
|-----------|-------------------|------------|------------|---------------|----------------|
| Vision encoder | 47.4 ms | 6.5 ms | 6.3 ms | 0.34 ms | 7.28x |
| Text encoder (1 seq) | 16.3 ms | 4.2 ms | 4.1 ms | 0.27 ms | 3.85x |
| Full pipeline | — | 19.8 ms | — | — | — |

**Throughput:** 153.4 images/sec (vision encoder)

> **Note:** Text encoder benchmarked per single sequence. Full pipeline encodes 3 texts serially.

## Speedup vs Stage 1

| Component | Stage 1 (avg) | Stage 2 (avg) | Speedup |
|-----------|---------------|-------------|---------|
| Vision encoder | 13.7 ms | 6.5 ms | 2.10x |
| Text encoder | 10.0 ms | 4.2 ms | 2.35x |

## Compile vs Cached (Program Cache)

| Component | First run (compile) | Cached run | Compile overhead |
|-----------|--------------------:|----------:|-----------------:|
| Vision encoder | 172.2 ms | 6.6 ms | 165.6 ms |
| Text encoder | 95.2 ms | 4.7 ms | 90.6 ms |

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
