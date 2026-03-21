# Tracy Profiler Investigation

## Goal

Generate standard TT perf sheet and Tracy device-level profiling data for CLIP-ViT Stage 2.

## Environment

- **Hardware:** Wormhole B0 (N300), 8x7 = 56 cores
- **OS:** Linux 6.1.62-tenstorrent-gpu
- **Pre-installed ttnn:** `/opt/venv/` (Tracy stub — not profiler-enabled)
- **Source build:** `/root/tt-metal` (built with Tracy enabled, `./build_metal.sh --release`)

## Problem: Pre-installed ttnn has Tracy Disabled

The pip-installed ttnn was compiled without `ENABLE_TRACY=ON`. Evidence:
- `libtracy-4c2b3bfe.so.0.10.0` exports only 5 symbols (stub) vs 290 in a real build
- `TT_METAL_DEVICE_PROFILER=1` → `TT_FATAL: requires a Tracy-enabled build`
- `ttnn.profiler.get_all_programs_perf_data()` returns empty dict

## Solution: Build tt-metal from Source

```bash
git clone https://github.com/tenstorrent/tt-metal.git --recurse-submodules --depth 1 /root/tt-metal
cd /root/tt-metal && ./build_metal.sh --release  # Tracy enabled by default
```

Build produces:
- `build_Release/lib/libtracy.so` with 290 exported symbols (full Tracy)
- `build_Release/tools/profiler/bin/capture-release` and `csvexport-release`

## Tracy Device Profiling Results

Profiled `profile_tracy.py` — 3 warmup + 1 profiled run of vision + text encoders:

```bash
TT_METAL_HOME=/root/tt-metal python3 -m tracy -r -v -o ./results/profiler profile_tracy.py
```

**Captured:** 6,874,797 Tracy zones, 178 MB trace file, 1664 device ops.

### Device Kernel Time — Stage 2 Vision Encoder (Single Run)

| Device Op | Count | Avg (us) | Total (us) | % |
|-----------|-------|----------|------------|---|
| MatmulDeviceOperation | 62 | 10.4 | 643.3 | 55.2% |
| BinaryNgDeviceOperation | 121 | 2.8 | 334.0 | 28.7% |
| SoftmaxDeviceOperation | 13 | 10.1 | 131.4 | 11.3% |
| UnaryDeviceOperation | 12 | 4.8 | 57.0 | 4.9% |
| **Total kernel** | **208** | | **1,165.8** | |

### Key Finding: Dispatch Overhead Dominates

| Metric | Time |
|--------|------|
| Total device kernel time | 1.17 ms |
| Total op-to-op latency (dispatch) | 49.4 ms |
| End-to-end vision (host-side) | 6.6 ms (median) |

Device kernels complete in **1.17 ms** — the hardware is fast. The bottleneck is **host-side Python dispatch overhead** (op-to-op latency between kernel launches). This is why Stage 3 should focus on reducing op count (fused QKV, SDPA) rather than making individual kernels faster.

### All Device Ops Across Full Profiling Session (4 runs × vision + text)

| Device Op | Count | Avg Kernel (us) | Total Kernel (us) | % |
|-----------|-------|-----------------|-------------------|---|
| MatmulDeviceOperation | 449 | 15.9 | 7,160.4 | 42.0% |
| LayerNormDeviceOperation | 110 | 29.8 | 3,276.3 | 19.2% |
| BinaryNgDeviceOperation | 825 | 3.8 | 3,105.2 | 18.2% |
| NlpCreateHeadsDeviceOperation | 53 | 24.2 | 1,282.0 | 7.5% |
| SoftmaxDeviceOperation | 88 | 14.3 | 1,259.1 | 7.4% |
| UnaryDeviceOperation | 87 | 6.3 | 544.3 | 3.2% |
| NLPConcatHeadsDeviceOperation | 52 | 8.0 | 416.4 | 2.4% |
| **Total** | **1664** | | **17,043.8** | |

## Host-Side Per-Layer Profiling

Complementary to Tracy device profiling — measured with `time.perf_counter()` + `ttnn.synchronize_device()` barriers.

**Stage 2 Vision Encoder — Per-Layer Breakdown (avg of 12 layers):**

| Op | Time (ms) | % | Category |
|----|-----------|---|----------|
| layer_norm1 | 0.07 | 8.1% | LayerNorm |
| qkv_linear | 0.10 | 12.7% | Attention |
| attn_scores | 0.06 | 7.6% | Attention |
| softmax | 0.05 | 5.9% | Attention |
| attn_context | 0.04 | 5.0% | Attention |
| concat_heads | 0.03 | 3.9% | Attention |
| out_proj | 0.06 | 7.2% | Attention |
| residual1_add | 0.04 | 5.4% | Residual |
| layer_norm2 | 0.07 | 8.2% | LayerNorm |
| fc1_linear | 0.07 | 8.5% | MLP |
| quick_gelu | 0.07 | 8.6% | MLP |
| fc2_linear | 0.10 | 13.0% | MLP |
| residual2_add | 0.04 | 5.0% | Residual |
| **Total** | **0.66** | **100%** | |

## Raw Profiling Artifacts

Generated in `results/profiler/.logs/`:
- `tracy_profile_log_host.tracy` — 178 MB binary trace (viewable in Tracy GUI)
- `cpp_device_perf_report.csv` — 1664 rows of per-op device kernel timing
- `profile_log_device.csv` — 142 MB raw device-side profiling log
- `tracy_ops_times.csv` — 6.8M host-side zone timing entries
- `tracy_ops_data.csv` — 6016 op metadata entries

## Conclusions

1. **Device compute is not the bottleneck** — kernels take 1.17 ms total for a vision encoder pass
2. **Host dispatch overhead dominates** — Python op-by-op execution adds ~5-6 ms per vision pass
3. **Matmul is the biggest kernel** — 55% of device kernel time (62 calls, avg 10.4 us each)
4. **Stage 3 priority:** Reduce op count via fused QKV, SDPA, and trace-based execution to minimize dispatch overhead
