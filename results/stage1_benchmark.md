# CLIP-ViT TTNN Benchmark — Stage 1

**Date:** 2026-03-21T18:22:42.009744
**Hardware:** Tenstorrent Wormhole B0 (N300)
**Model:** openai/clip-vit-base-patch32
**Runs:** 20 (after compile warmup)

## Latency

| Component | PyTorch CPU (median) | TTNN (median) | TTNN (min) | TTNN (stddev) | Speedup vs CPU |
|-----------|----------------------|---------------|------------|---------------|----------------|
| Vision encoder | 49.1 ms | 7.6 ms | 7.5 ms | 1.16 ms | 6.45x |
| Text encoder (1 seq) | 17.1 ms | 3.8 ms | 3.7 ms | 0.37 ms | 4.54x |
| Full pipeline | — | 22.4 ms | — | — | — |

**Throughput:** 124.1 images/sec (vision encoder)

> **Note:** Text encoder benchmarked per single sequence. Full pipeline encodes 3 texts serially.

## Compile vs Cached (Program Cache)

| Component | First run (compile) | Cached run | Compile overhead |
|-----------|--------------------:|----------:|-----------------:|
| Vision encoder | 170.1 ms | 8.8 ms | 161.3 ms |
| Text encoder | 140.6 ms | 4.2 ms | 136.4 ms |

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
