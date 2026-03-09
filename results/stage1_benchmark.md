# CLIP-ViT TTNN Benchmark — Stage 1

**Date:** 2026-03-09T22:12:10.495294
**Hardware:** Tenstorrent Wormhole B0 (N300)
**Model:** openai/clip-vit-base-patch32
**Runs:** 10 (after 3 warmup)

## Timing

| Component | PyTorch CPU (avg) | PyTorch CPU (min) | TTNN (avg) | TTNN (min) | Speedup |
|-----------|-------------------|-------------------|------------|------------|---------|
| Vision encoder | 53.3 ms | 51.4 ms | 7.8 ms | 7.2 ms | 6.85x |
| Text encoder | 19.6 ms | 16.4 ms | 7.6 ms | 3.7 ms | 2.58x |
| Full pipeline | — | — | 39.3 ms | — | — |

## Accuracy (PCC)

| Metric | PCC |
|--------|-----|
| Vision embedding | 0.968555 |
| Text embedding | 0.989511 |
| Logits | 0.998843 |

## Prediction

| | Value |
|--|-------|
| Predicted | "a photo of a cat" |
| Correct | Yes |
| TTNN logits | [[25.75, 20.375, 19.875]] |
| PyTorch logits | [[24.570056915283203, 19.304899215698242, 18.48160743713379]] |
| TTNN probs | [[0.9921875, 0.004608154296875, 0.0027923583984375]] |
| PyTorch probs | [[0.9926174283027649, 0.005130420438945293, 0.0022521736100316048]] |

## Configuration

- Memory: DRAM interleaved
- Math fidelity: HiFi4
- Weight dtype: bfloat16
- GELU: separate
- Softmax: native ttnn.softmax
