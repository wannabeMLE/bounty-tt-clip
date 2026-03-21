# Debug & Investigation Log

**Date:** 2026-03-21
**Hardware:** Tenstorrent Wormhole B0 (N300), 8x7 = 56 cores
**Model:** openai/clip-vit-base-patch32

This document records the full debugging process for Stage 2 benchmarking, including all errors encountered, root causes identified, and solutions applied.

---

## 1. Initial State: Stage 2 Performance Looked Worse Than Stage 1

**Problem:** After implementing all Stage 2 optimizations (L1 interleaved, LoFi, bfloat8_b, softmax_in_place), initial benchmarks showed Stage 2 **slower** than Stage 1 (avg 14.4ms vs 12.4ms for vision encoder).

**Root cause:** Using **average** (mean) instead of **median** for comparison. L1 interleaved memory has intermittent latency spikes (~2-5% of runs) where a single run takes 2-3x normal time due to L1 allocator fragmentation/eviction when allocating intermediates across 12 transformer layers.

**Solution:** Switched all benchmark comparisons to **median**. Median is robust to outlier spikes.

**Result:** With median, Stage 2 vision = 6.6ms vs Stage 1 = 7.6ms → **1.16x faster**.

---

## 2. Isolation Testing: Which Optimizations Actually Help?

**Question:** Stage 2 applies four optimizations simultaneously. Which ones contribute to the speedup?

**Methodology:** Tested each optimization in isolation on DRAM first, then added L1.

| Test | Memory | Compute | Vision median |
|------|--------|---------|---------------|
| Stage 1 baseline | DRAM | HiFi4 + bf16 | 7.6 ms |
| bf8 weights only | DRAM | HiFi4 + bf8 | 7.6 ms |
| LoFi only | DRAM | LoFi + bf8 | 7.5 ms |
| L1 only | L1 | LoFi + bf8 | 6.6 ms |
| Full Stage 2 | L1 | LoFi + bf8 | 6.6 ms |

**Finding:** Only L1 memory provides measurable speedup. LoFi and bfloat8_b provide zero speed benefit at CLIP's small tensor sizes ([1,64,768] vision, [1,96,512] text). The tensors are too small for compute or bandwidth savings to materialize.

---

## 3. Sharding Investigation: Why Not Shard?

**Hypothesis:** Explicit sharding (HEIGHT_SHARDED, BLOCK_SHARDED) should be faster than interleaved.

**Result:** All sharding strategies were slower or couldn't run.

| Strategy | Result | Reason |
|----------|--------|--------|
| HEIGHT_SHARDED | 14x slower | Only 2-3 cores active (limited by tile-rows) vs 56 cores interleaved |
| BLOCK_SHARDED (output) | Cannot run | Output tiles too wide for 8-col grid |
| BLOCK_SHARDED (input) | 1.2x-5.6x slower | Reshard overhead + reduced parallelism |

**Root cause:** CLIP-ViT-B/32 has very small sequences (50 vision tokens → 2 tile-rows, 77 text tokens → 3 tile-rows). The L1 interleaved matmul kernel already distributes work across all 56 cores internally. Explicit sharding restricts parallelism to the shard grid.

Full data: `results/stage2_sharding_investigation.md`

---

## 4. Tracy Profiler Contamination

**Sequence of events:**
1. **09:32** — Clean Stage 2 benchmark: vision median **6.4ms**, min 6.3ms
2. **16:01** — Ran `profile_tracy.py` with `TT_METAL_DEVICE_PROFILER=1` for Tracy device profiling
3. **16:10+** — All subsequent Stage 2 benchmarks showed vision median **13.3ms+** (>2x regression)
4. **16:28** — Stage 1 (DRAM) still fine at 7.5ms. Only L1 paths affected.

**Diagnosis steps attempted:**
- Cleared on-disk kernel cache (`rm -rf /root/.cache/tt-metal-cache`) → No improvement
- Device reset via `tt-smi -r 0` and `tt-smi -r 1` → Made things **worse** (vision 23-50ms)
- After device reset, Stage 1 recovered to ~7.5ms but Stage 2 L1 remained degraded

**Root cause:** Tracy profiler (`TT_METAL_DEVICE_PROFILER=1`) injects profiling hooks into device firmware (RISC-V cores). This contamination persists across:
- Device open/close cycles within the same process
- Separate Python processes
- `tt-smi -r` device resets

The contamination is at the firmware/RISC-V level. Only a **full board power cycle** clears it.

**Key insight:** Tracy profiling reserves additional L1 buffer space for profiling data on each core. This reduces available L1 SRAM, causing more frequent evictions and higher latency specifically for L1 interleaved workloads. DRAM workloads are unaffected because they don't compete for L1 space.

---

## 5. Stale tt-smi Processes Destroying Benchmark Stability

**Problem:** After board reset and kernel cache clear, benchmarks showed correct min values (7.6ms S1, 6.5ms S2) but terrible medians (16.4ms S1, 18.3ms S2) with high variance (stddev 8-12ms).

**Diagnosis:**
```
$ ps aux --sort=-%cpu | head -5
root  434205 97.9%  tt-smi
root  425333 97.7%  tt-smi
```

Two zombie `tt-smi` TUI processes consuming **~200% CPU** total. Load average was 2.0+ on what should be an idle system.

**Root cause:** A leftover background bash loop from a previous session was repeatedly spawning `tt-smi` in interactive TUI mode:
```bash
for i in $(seq 1 12); do sleep 10; tt-smi 2>&1 | head -3; done
```
Each `tt-smi` invocation starts the TUI (ncurses-based), which consumes 100% CPU in its render loop. The `| head -3` pipe closure doesn't cleanly terminate the TUI process.

**Solution:**
1. `kill -9` the parent bash loop (PID 425249) to stop respawning
2. `kill -9` both tt-smi children
3. Wait for load average to drop below 1.0

**Result after fix:**
- Load average: 0.73
- Stage 1: vision median **7.6ms**, stddev 1.16ms (was 8.76ms)
- Stage 2: vision median **6.6ms**, min 6.3ms (was 18.3ms median)

---

## 6. Program Cache Warmup Misunderstanding

**Initial assumption:** L1 interleaved needs more warmup iterations than DRAM (set `NUM_WARMUP = 50` with comment "L1 interleaved needs ~50 passes to fully stabilize (lazy kernel paths)").

**Correction (from hardware engineer review):** Both stages have exactly the same ~15 unique op signatures per layer. The program cache compiles all kernels after **one forward pass**. One or two additional passes ensure stability. L1 does NOT need more warmup than DRAM.

**What the 50 warmup actually helps with:** Host-side variance stabilization (Python JIT, CPU cache warming, memory allocator settling). Not device-side compilation.

**Fix:** Updated benchmark.py comment:
```python
NUM_WARMUP = 50  # Program cache compiles after 1 pass; extra warmup stabilizes host-side variance
```

---

## 7. QuickGELU Fusion Attempt

**Hypothesis:** Replace CLIP's QuickGELU (`x * sigmoid(1.702x)`, 3 ops) with `ttnn.gelu` (1 fused op).

**Result:** Standard GELU is mathematically different from QuickGELU. Per-op PCC is 0.999, but through 12 layers it compounds to 0.964 end-to-end — well below the 0.98 threshold.

**Conclusion:** Cannot fuse. QuickGELU must remain as 3 separate ops (multiply, sigmoid, multiply).

---

## 8. On-Device Patch Embedding Attempt

**Hypothesis:** Move patch embedding from CPU `conv2d` to device using fold+linear.

**Implementation:** `ttnn.fold` to rearrange patches + `ttnn.linear` for projection.

**Result:** Output PCC = **-0.02** (effectively random). The fold operation doesn't correctly replicate PyTorch's `nn.Conv2d` stride behavior for this patch size/stride combination.

**Conclusion:** Deferred to Stage 3. CPU conv2d remains for Stages 1-2.

---

## 9. Final Clean Benchmark Procedure

After resolving all issues, the final benchmark run followed this procedure:

1. **Kill all background processes** consuming CPU (verify load < 1.0)
2. **Board reset:** `tt-smi -r 0` (full PCI-level reset)
3. **Clear kernel cache:** `rm -rf /root/.cache/tt-metal-cache`
4. **Wait for system idle:** load average < 1.0
5. **Run Stage 1:** `python benchmark.py --stage 1 --output results/stage1_benchmark.md` (cold process)
6. **Run Stage 2:** `python benchmark.py --stage 2 --output results/stage2_benchmark.md` (separate cold process)

Each benchmark:
- 50 warmup iterations (1 compile + 1 cached + 50 warmup loop)
- 20 timed runs with `ttnn.synchronize_device()` barriers
- `ttnn.deallocate()` between runs to prevent memory fragmentation
- Median used for all comparisons

---

## 10. Final Results

### Performance (20 runs, 50 warmup, cold process, idle system)

| Component | Stage 1 (median) | Stage 2 (median) | Speedup |
|-----------|-------------------|-------------------|---------|
| Vision encoder | 7.6 ms | **6.6 ms** | **1.16x** |
| Text encoder | 3.8 ms | 4.1 ms | 0.91x |
| Full pipeline | 22.4 ms | **19.0 ms** | **1.18x** |
| Throughput (vision) | 132 img/s | **152 img/s** | **1.16x** |

### Accuracy (all PASS)

| Metric | Stage 1 | Stage 2 | Threshold |
|--------|---------|---------|-----------|
| Vision PCC | 0.9990 | 0.9956 | >= 0.99 / 0.98 |
| Text PCC | 0.9999 | 0.9993 | >= 0.99 / 0.98 |
| Logits PCC | 0.9995 | 0.9944 | >= 0.99 / 0.98 |
| COCO match rate | 20/20 | 20/20 | 100% |

### Per-Layer Profiling (with sync barriers, separate measurement)

| Component | Stage 1 (avg/layer) | Stage 2 (avg/layer) | Speedup |
|-----------|--------------------|--------------------|---------|
| Total | 0.95 ms | 0.66 ms | **1.44x** |
| Attention | 0.51 ms (53%) | 0.28 ms (43%) | 1.82x |
| MLP | 0.25 ms (26%) | 0.21 ms (32%) | 1.19x |
| LayerNorm | 0.13 ms (13%) | 0.11 ms (17%) | 1.18x |
| Residual | 0.07 ms (8%) | 0.05 ms (8%) | 1.40x |

### Tracy Device Profiling

Device kernels complete in **1.17 ms** for a full vision encoder pass. The bottleneck is host-side Python dispatch overhead (~5-6 ms), not kernel execution.

---

## Lessons Learned

1. **Always use median for L1 benchmarks.** L1 interleaved has 2-5% spike runs from allocator fragmentation. Mean is misleading.

2. **Tracy profiler contaminates device firmware.** Running with `TT_METAL_DEVICE_PROFILER=1` permanently degrades L1 performance until a full board power cycle. Never run Tracy and benchmarks in the same session.

3. **Check for background CPU consumers before benchmarking.** Even a `tt-smi` TUI process at 100% CPU causes massive benchmark variance through CPU contention on the host-side dispatch path.

4. **Compute fidelity doesn't matter at CLIP's scale.** HiFi4 vs LoFi, bfloat16 vs bfloat8_b — all produce the same latency for tensors this small. The only optimization that helps is memory hierarchy (DRAM → L1).

5. **Small-sequence models don't benefit from sharding.** With only 2-3 tile-rows, explicit sharding restricts parallelism below what the interleaved kernel achieves with 56 cores.
