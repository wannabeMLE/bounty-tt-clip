# CLIP-ViT TTNN Benchmark — Stage 1

**Date:** 2026-03-14T00:26:52.409897
**Hardware:** Tenstorrent Wormhole B0 (N300)
**Model:** openai/clip-vit-base-patch32
**Runs:** 10 (after compile warmup)

## Latency

| Component | PyTorch CPU (avg) | PyTorch CPU (min) | TTNN (avg) | TTNN (min) | Speedup vs CPU |
|-----------|-------------------|-------------------|------------|------------|----------------|
| Vision encoder | 63.4 ms | 53.9 ms | 13.2 ms | 10.8 ms | 4.82x |
| Text encoder | 21.5 ms | 18.4 ms | 14.1 ms | 9.1 ms | 1.52x |
| Full pipeline | — | — | 40.8 ms | — | — |

**Throughput:** 76.0 images/sec (vision encoder)

## Compile vs Cached (Program Cache)

| Component | First run (compile) | Cached run | Compile overhead |
|-----------|--------------------:|----------:|-----------------:|
| Vision encoder | 177.9 ms | 8.5 ms | 169.3 ms |
| Text encoder | 139.5 ms | 10.9 ms | 128.6 ms |

## Accuracy (PCC)

| Metric | PCC | Status |
|--------|-----|--------|
| Vision embedding | 0.968555 | FAIL (>= 0.98) |
| Text embedding | 0.989511 | PASS (>= 0.98) |
| Logits | 0.998843 | PASS (>= 0.98) |

## Prediction

| | Value |
|--|-------|
| Predicted | "a photo of a cat" |
| Correct | Yes |
| TTNN logits | [[25.75, 20.375, 19.875]] |
| PyTorch logits | [[24.570056915283203, 19.304899215698242, 18.48160743713379]] |

## Configuration

| Setting | Value |
|---------|-------|
| Memory | DRAM interleaved |
| Math fidelity | HiFi4 |
| Weight dtype | bfloat16 |
| GELU | separate |
| Compute grid | 8x7 (56 cores) |
| Dispatch | WORKER |
