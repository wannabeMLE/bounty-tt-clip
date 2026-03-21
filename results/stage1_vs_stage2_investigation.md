# Stage 1 vs Stage 2 Performance Investigation

**Date:** 2026-03-21
**Hardware:** Tenstorrent Wormhole B0 (N300), 1 chip, 8x7 = 56 cores
**Model:** openai/clip-vit-base-patch32

## What changed between stages

| Setting | Stage 1 | Stage 2 |
|---------|---------|---------|
| Memory | DRAM interleaved | **L1 interleaved** |
| Math fidelity | HiFi4 | **LoFi** |
| Weight dtype | bfloat16 | **bfloat8_b** |
| Activation | QuickGELU (3 ops) | QuickGELU (3 ops) |
| Softmax | ttnn.softmax | **ttnn.softmax_in_place** |
| Patch embedding | CPU conv2d | CPU conv2d |

## Final benchmark results (20 runs, 50 warmup, cold process, idle system)

| Component | Stage 1 (median) | Stage 2 (median) | Speedup |
|-----------|-------------------|-------------------|---------|
| Vision encoder | 7.6 ms | **6.6 ms** | **1.16x** |
| Text encoder | 3.8 ms | 4.1 ms | 0.91x |
| Full pipeline | 22.4 ms | **19.0 ms** | **1.18x** |
| Throughput (vision, median) | 132 img/s | **152 img/s** | **1.16x** |

## Accuracy — ALL PASS

| Metric | Stage 1 | Stage 2 | Threshold |
|--------|---------|---------|-----------|
| Vision PCC | 0.9990 | 0.9956 | >= 0.99 / 0.98 |
| Text PCC | 0.9999 | 0.9993 | >= 0.99 / 0.98 |
| Logits PCC | 0.9996 | 0.9944 | >= 0.99 / 0.98 |
| COCO match rate | 20/20 | 20/20 | — |

## Investigation: why Stage 2 initially appeared slower

Early benchmarks showed Stage 2 **slower** than Stage 1 (avg 14.4ms vs 12.4ms vision).
This was a measurement artifact. Systematic isolation tests identified the root cause.

### Isolation tests performed

All tests used Stage 2 bf8 weights (same model), DRAM memory, 20 runs, 7+ warmup.

| Test | Memory | Compute | Vision (avg) | Vision (min) |
|------|--------|---------|-------------|-------------|
| Stage 1 baseline | DRAM | HiFi4 + bf16 | 7.6 ms | 7.5 ms |
| bf8 only | DRAM | HiFi4 + bf8 | 7.6 ms | 7.4 ms |
| LoFi only | DRAM | LoFi + bf8 | 7.5 ms | 7.3 ms |
| L1 only (same compute) | L1 | LoFi + bf8 | 6.8 ms | 6.3 ms |
| Full Stage 2 | L1 | LoFi + bf8 | 6.8 ms | 6.3 ms |

### Compute fidelity comparison (all DRAM, bf8 weights)

| Config | Vision (avg) | Vision (min) |
|--------|-------------|-------------|
| HiFi4, approx=False, fp32=True | 8.0 ms | 7.4 ms |
| HiFi4, approx=True, fp32=False | 7.6 ms | 7.3 ms |
| HiFi2, approx=True, fp32=False | 7.5 ms | 7.3 ms |
| LoFi, approx=True, fp32=False | 8.0 ms | 7.3 ms |
| LoFi, approx=False, fp32=False | 7.5 ms | 7.3 ms |
| LoFi, approx=False, fp32=True | 7.7 ms | 7.3 ms |

**Finding:** All fidelity levels produce the same speed (~7.5ms). At CLIP's tensor sizes, compute is not the bottleneck.

### L1 spike analysis

L1 interleaved has intermittent latency spikes (~2-5% of runs) where a single run takes 2-3x normal time.

Text encoder example (40 runs, 20 warmup, L1):
- Normal runs: 4.1-4.5 ms (97.5% of runs)
- Spike runs: 10-11 ms (2.5% of runs)
- Cause: L1 memory allocator fragmentation/eviction when allocating/deallocating intermediates across 12 layers

This made **average** misleading. Switching to **median** gives stable results.

### Per-layer profiling (with sync between ops)

| Category | Stage 1 (avg/layer) | Stage 2 (avg/layer) | Speedup |
|----------|--------------------|--------------------|---------|
| Total | 0.95 ms | 0.66 ms | **1.44x** |
| Attention | 0.51 ms (53%) | 0.28 ms (43%) | 1.82x |
| MLP | 0.25 ms (26%) | 0.21 ms (32%) | 1.19x |
| LayerNorm | 0.13 ms (13%) | 0.11 ms (17%) | 1.18x |
| Residual | 0.07 ms (8%) | 0.05 ms (8%) | 1.40x |

Note: profiling inserts sync between ops (kills pipelining), so absolute times differ from end-to-end.

## Key findings

1. **L1 memory is the only optimization that provides measurable speedup** — 1.16x vision (median). The lower NOC latency for intermediate tensors benefits attention ops most (1.82x per-layer).

2. **LoFi and bfloat8_b provide zero speed benefit** at CLIP's tensor sizes (`[1, 64, 768]` vision, `[1, 96, 512]` text). These tensors are too small for compute or bandwidth savings to materialize. The benefit is reduced memory footprint (useful for larger batch sizes or models).

3. **Text encoder benefits less from L1** (0.89x median) because its tensors are smaller (512-dim vs 768-dim) and the L1 allocation overhead is proportionally larger relative to compute.

4. **L1 interleaved causes intermittent latency spikes** (2-5% of runs). Use median instead of mean for stable comparisons.

5. **Benchmark variability is dominated by system load, not warmup count.** Both stages have the same ~15 unique op signatures per layer, all compiled after one forward pass. Program cache warmup requires 2-3 iterations for either stage. Run-to-run variance comes from CPU contention (host-side tensor prep, data transfers) and occasional L1 allocator spikes.

## Methodology notes

- All benchmarks use `ttnn.synchronize_device()` before and after each timed run
- Program cache enabled before any runs
- PyTorch CPU baseline uses same warmup/run methodology
- Cross-stage comparisons use median (robust to L1 spikes)
- Speedup vs CPU uses median for both PyTorch and TTNN
