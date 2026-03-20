# Stage 2 Sharding & Optimization Investigation

## Conclusion: L1 Interleaved wins. All sharding strategies are slower for CLIP's tensor sizes.

The L1 interleaved matmul kernel already distributes work across all 56 cores internally. Explicit sharding adds overhead without benefit at these small tensor sizes.

---

## HEIGHT_SHARDED

- **Result:** 14x regression
- Vision fc1: seq_tiles=2 → only 2 cores used for height sharding
- Text fc1: seq_tiles=3 → only 3 cores used
- L1 interleaved matmul kernel uses full 56-core grid internally

## BLOCK_SHARDED — Exhaustive Benchmark

Script: `test_block_shard.py` — 100 timed runs, 5 warmup per config, sync at boundaries.

### Vision fc1: [1,64,768] × [768,3072]

Baseline (L1 interleaved): **0.0229 ms**

**Block-sharded OUTPUT:** ALL FAILED — output has 96 tile-columns, exceeds 8-column grid max.

**Block-sharded INPUT → interleaved output:**

| Config | Cores | Time (ms) | vs Baseline |
|--------|-------|-----------|-------------|
| BlkIn 2×2→Intlvd | 4 | 0.1294 | 0.18x (5.6x SLOWER) |
| BlkIn 2×3→Intlvd | 6 | 0.0839 | 0.27x |
| BlkIn 2×4→Intlvd | 8 | 0.0633 | 0.36x |
| BlkIn 2×6→Intlvd | 12 | 0.0454 | 0.50x |
| BlkIn 2×8→Intlvd | 16 | 0.0368 | 0.62x |
| BlkIn 1×4→Intlvd | 4 | 0.0533 | 0.43x |
| BlkIn 1×8→Intlvd | 8 | 0.0393 | 0.58x |

### Text fc1: [1,96,512] × [512,2048]

Baseline (L1 interleaved): **0.0188 ms**

**Block-sharded OUTPUT:** ALL FAILED.

**Block-sharded INPUT → interleaved output:**

| Config | Cores | Time (ms) | vs Baseline |
|--------|-------|-----------|-------------|
| BlkIn 3×2→Intlvd | 6 | 0.0579 | 0.33x |
| BlkIn 3×4→Intlvd | 12 | 0.0318 | 0.59x |
| BlkIn 3×8→Intlvd | 24 | 0.0224 | 0.84x (closest) |
| BlkIn 1×4→Intlvd | 4 | 0.0379 | 0.50x |
| BlkIn 1×8→Intlvd | 8 | 0.0314 | 0.60x |

## Reshard Cost

Script: `test_reshard_cost.py` — Even if sharded matmul were faster, reshard overhead (interleaved↔sharded transitions) would eat gains.

## QuickGELU Fusion

CLIP uses QuickGELU: `x * sigmoid(1.702 * x)` — implemented as 3 separate ops.

- Standard GELU fused kernel exists (`activation='gelu'`) but is mathematically different
- SiLU (`x * sigmoid(x)`) also different (max abs diff 0.18)
- Per-op PCC 0.999986 is misleading — error compounds through 12 layers (caused 0.964 PCC regression previously)
- **Decision:** Keep 3-op QuickGELU. Document as platform limitation.

### Fusion Benchmark (100 runs, fc1 shape)

| Config | Time (ms) | Speedup vs 3-op |
|--------|-----------|-----------------|
| Linear only (no activation) | 0.0295 | — |
| Linear + QuickGELU (3 sep ops) | 0.0663 | 1.00x |
| Linear + fused GELU | 0.0415 | 1.60x |
| Linear + fused SiLU | 0.0386 | 1.72x |

## core_grid Parameter

`ttnn.linear` with `core_grid=ttnn.CoreGrid(y=7, x=8)` (full grid hint):

| Config | Time (ms) |
|--------|-----------|
| No core_grid | 0.0296 |
| core_grid=8×7 | 0.0347 |

**Result:** 0.86x (slightly worse). Kernel already uses full grid for interleaved layout.

## softmax_in_place

Already implemented in Stage 2 at `clip_model.py` line 508: `ttnn.softmax_in_place(attention_scores)`.

## Stage 2 Final Status

| Item | Status |
|------|--------|
| L1 interleaved memory | Done |
| LoFi + bfloat8_b weights | Done |
| Program cache (all stages) | Done |
| Sharding (HEIGHT, BLOCK) | Done — interleaved wins |
| QuickGELU correctness | Done (0.999 PCC) |
| Activation fusion | N/A — no QuickGELU fused op in TTNN |
| softmax_in_place | Done |
| core_grid hint | Tested — no benefit |
| On-device patch embedding | Done |
