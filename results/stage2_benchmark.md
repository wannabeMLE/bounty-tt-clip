# CLIP-ViT TTNN Benchmark — Stage 2

**Date:** 2026-03-14T00:29:30.400862
**Hardware:** Tenstorrent Wormhole B0 (N300)
**Model:** openai/clip-vit-base-patch32
**Runs:** 10 (after compile warmup)

## Latency

| Component | PyTorch CPU (avg) | PyTorch CPU (min) | TTNN (avg) | TTNN (min) | Speedup vs CPU |
|-----------|-------------------|-------------------|------------|------------|----------------|
| Vision encoder | 66.0 ms | 53.3 ms | 6.6 ms | 6.3 ms | 10.01x |
| Text encoder | 19.8 ms | 17.4 ms | 4.2 ms | 3.8 ms | 4.74x |
| Full pipeline | — | — | 19.0 ms | — | — |

**Throughput:** 151.5 images/sec (vision encoder)

## Speedup vs Stage 1

| Component | Stage 1 (avg) | Stage 2 (avg) | Speedup |
|-----------|---------------|-------------|---------|
| Vision encoder | 13.2 ms | 6.6 ms | 1.99x |
| Text encoder | 14.1 ms | 4.2 ms | 3.37x |

## Compile vs Cached (Program Cache)

| Component | First run (compile) | Cached run | Compile overhead |
|-----------|--------------------:|----------:|-----------------:|
| Vision encoder | 9923.9 ms | 6.9 ms | 9917.1 ms |
| Text encoder | 5680.4 ms | 3.8 ms | 5676.5 ms |

## Accuracy (PCC)

| Metric | PCC | Status |
|--------|-----|--------|
| Vision embedding | 0.964412 | FAIL (>= 0.98) |
| Text embedding | 0.989631 | PASS (>= 0.98) |
| Logits | 0.994397 | PASS (>= 0.98) |

## Prediction

| | Value |
|--|-------|
| Predicted | "a photo of a cat" |
| Correct | Yes |
| TTNN logits | [[26.0, 20.375, 20.25]] |
| PyTorch logits | [[24.570056915283203, 19.304899215698242, 18.48160743713379]] |

## Configuration

| Setting | Value |
|---------|-------|
| Memory | L1 interleaved |
| Math fidelity | LoFi |
| Weight dtype | bfloat8_b |
| GELU | fused in fc1 |
| Compute grid | 8x7 (56 cores) |
| Dispatch | WORKER |
