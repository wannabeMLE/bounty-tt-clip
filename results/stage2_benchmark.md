# CLIP-ViT TTNN Benchmark — Stage 2

**Date:** 2026-03-21T18:23:14.096925
**Hardware:** Tenstorrent Wormhole B0 (N300)
**Model:** openai/clip-vit-base-patch32
**Runs:** 20 (after compile warmup)

## Latency

| Component | PyTorch CPU (median) | TTNN (median) | TTNN (min) | TTNN (stddev) | Speedup vs CPU |
|-----------|----------------------|---------------|------------|---------------|----------------|
| Vision encoder | 50.3 ms | 6.6 ms | 6.3 ms | 12.45 ms | 7.68x |
| Text encoder (1 seq) | 16.9 ms | 4.1 ms | 4.0 ms | 0.80 ms | 4.08x |
| Full pipeline | — | 19.0 ms | — | — | — |

**Throughput:** 88.6 images/sec (vision encoder)

> **Note:** Text encoder benchmarked per single sequence. Full pipeline encodes 3 texts serially.

## Speedup vs Stage 1

| Component | Stage 1 (median) | Stage 2 (median) | Speedup |
|-----------|---------------|-------------|---------|
| Vision encoder | 7.6 ms | 6.6 ms | 1.16x |
| Text encoder | 3.8 ms | 4.1 ms | 0.91x |

## Compile vs Cached (Program Cache)

| Component | First run (compile) | Cached run | Compile overhead |
|-----------|--------------------:|----------:|-----------------:|
| Vision encoder | 177.8 ms | 25.3 ms | 152.5 ms |
| Text encoder | 110.7 ms | 6.6 ms | 104.1 ms |

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
