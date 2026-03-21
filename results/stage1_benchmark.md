# CLIP-ViT TTNN Benchmark — Stage 1

**Date:** 2026-03-21T00:23:28.807209
**Hardware:** Tenstorrent Wormhole B0 (N300)
**Model:** openai/clip-vit-base-patch32
**Runs:** 20 (after compile warmup)

## Latency

| Component | PyTorch CPU (avg) | TTNN (avg) | TTNN (min) | TTNN (stddev) | Speedup vs CPU |
|-----------|-------------------|------------|------------|---------------|----------------|
| Vision encoder | 53.3 ms | 12.8 ms | 11.9 ms | 1.59 ms | 4.17x |
| Text encoder (1 seq) | 26.6 ms | 10.9 ms | 7.9 ms | 1.91 ms | 2.43x |
| Full pipeline | — | 43.1 ms | — | — | — |

**Throughput:** 78.2 images/sec (vision encoder)

> **Note:** Text encoder benchmarked per single sequence. Full pipeline encodes 3 texts serially.

## Compile vs Cached (Program Cache)

| Component | First run (compile) | Cached run | Compile overhead |
|-----------|--------------------:|----------:|-----------------:|
| Vision encoder | 180.2 ms | 23.8 ms | 156.4 ms |
| Text encoder | 141.4 ms | 10.4 ms | 130.9 ms |

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
