"""Benchmark block-sharded matmul vs interleaved baseline.

Tests fc1 matmul with various BLOCK_SHARDED grid configurations to find
whether block sharding can beat L1-interleaved at these tensor sizes.

Vision: [1,64,768] × [768,3072] → 2 tile-rows × 24 tile-cols
Text:   [1,96,512] × [512,2048] → 3 tile-rows × 16 tile-cols

Methodology: 100 runs between sync boundaries (amortized timing).
5 warmup runs per config to compile + cache kernels.
"""
import sys, time, torch
import ttnn

sys.path.insert(0, ".")

dispatch_config = ttnn.DispatchCoreConfig(type=ttnn.DispatchCoreType.WORKER)
device = ttnn.open_device(device_id=0, dispatch_core_config=dispatch_config)
device.enable_program_cache()

N = 100  # timed runs
WARMUP = 5  # warmup runs per config

lofi = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.LoFi)


def make_block_shard_config(shape, grid_rows, grid_cols):
    """Create a BLOCK_SHARDED memory config.

    shape: (batch*seq_tiles, hidden_tiles) in tiles
    grid_rows: number of core rows (must divide seq_tiles)
    grid_cols: number of core cols (must divide hidden_tiles)
    """
    seq_tiles, hidden_tiles = shape
    if seq_tiles % grid_rows != 0:
        return None  # not divisible
    if hidden_tiles % grid_cols != 0:
        return None  # not divisible

    shard_h = (seq_tiles // grid_rows) * 32  # in elements
    shard_w = (hidden_tiles // grid_cols) * 32  # in elements

    core_range = ttnn.CoreRange(
        ttnn.CoreCoord(0, 0),
        ttnn.CoreCoord(grid_cols - 1, grid_rows - 1),  # (x, y) = (col, row)
    )
    shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({core_range}),
        (shard_h, shard_w),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.BufferType.L1,
        shard_spec,
    )


def benchmark_config(label, x_tensor, w_tensor, mem_config_out, device):
    """Benchmark a single matmul configuration. Returns avg ms."""
    # Warmup (compile + cache)
    for _ in range(WARMUP):
        out = ttnn.linear(x_tensor, w_tensor, compute_kernel_config=lofi,
                          memory_config=mem_config_out)
        ttnn.deallocate(out)

    # Timed runs
    ttnn.synchronize_device(device)
    t0 = time.perf_counter()
    for _ in range(N):
        out = ttnn.linear(x_tensor, w_tensor, compute_kernel_config=lofi,
                          memory_config=mem_config_out)
        ttnn.deallocate(out)
    ttnn.synchronize_device(device)
    avg_ms = (time.perf_counter() - t0) / N * 1000
    return avg_ms


def run_suite(name, seq_len, hidden_size, intermediate_size, grid_configs):
    """Run all block shard configs for a given encoder type."""
    seq_tiles = seq_len // 32
    hidden_tiles = hidden_size // 32
    inter_tiles = intermediate_size // 32

    print(f"\n{'='*70}")
    print(f"  {name}: [{1},{seq_len},{hidden_size}] × [{hidden_size},{intermediate_size}]")
    print(f"  Tiles: {seq_tiles} × {hidden_tiles} (input)  →  {seq_tiles} × {inter_tiles} (output)")
    print(f"  Runs: {N} timed, {WARMUP} warmup per config")
    print(f"{'='*70}")

    # Create tensors
    x_torch = torch.randn(1, seq_len, hidden_size)
    w_torch = torch.randn(hidden_size, intermediate_size)

    x_l1 = ttnn.from_torch(x_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                            device=device, memory_config=ttnn.L1_MEMORY_CONFIG)
    w_dram = ttnn.from_torch(w_torch, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT,
                             device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    # --- Baseline: L1 interleaved ---
    baseline_ms = benchmark_config("interleaved", x_l1, w_dram, ttnn.L1_MEMORY_CONFIG, device)
    print(f"\n  {'Config':<30} {'Cores':>6} {'Shard':>12} {'Time (ms)':>10} {'vs Base':>10}")
    print(f"  {'-'*30} {'-'*6} {'-'*12} {'-'*10} {'-'*10}")
    print(f"  {'L1 Interleaved (baseline)':<30} {'all':>6} {'N/A':>12} {baseline_ms:>10.4f} {'1.00x':>10}")

    results = [("L1 Interleaved", "all", "N/A", baseline_ms, 1.0)]

    # --- Block-sharded configs ---
    for grid_r, grid_c in grid_configs:
        label = f"Block {grid_r}×{grid_c}"
        cores = grid_r * grid_c

        # Output shard config: output has shape [1, seq_len, intermediate_size]
        # Output tiles: seq_tiles × inter_tiles
        # Grid rows must divide seq_tiles, grid cols must divide inter_tiles
        if inter_tiles % grid_c != 0:
            print(f"  {label:<30} {cores:>6} {'SKIP':>12} {'--':>10} {'inter_tiles not divisible':>10}")
            results.append((label, cores, "SKIP", None, None))
            continue

        try:
            out_mem = make_block_shard_config((seq_tiles, inter_tiles), grid_r, grid_c)
            if out_mem is None:
                print(f"  {label:<30} {cores:>6} {'SKIP':>12} {'--':>10} {'output not divisible':>10}")
                results.append((label, cores, "SKIP", None, None))
                continue
            out_shard_h = (seq_tiles // grid_r) * 32
            out_shard_w = (inter_tiles // grid_c) * 32
            shard_str = f"{out_shard_h}×{out_shard_w}"

            avg_ms = benchmark_config(label, x_l1, w_dram, out_mem, device)
            ratio = baseline_ms / avg_ms if avg_ms > 0 else 0
            status = "FASTER" if avg_ms < baseline_ms else "SLOWER"
            print(f"  {label:<30} {cores:>6} {shard_str:>12} {avg_ms:>10.4f} {ratio:>9.2f}x {status}")
            results.append((label, cores, shard_str, avg_ms, ratio))

        except Exception as e:
            err_msg = str(e)[:60]
            print(f"  {label:<30} {cores:>6} {'ERROR':>12} {err_msg}")
            results.append((label, cores, "ERROR", None, None))

    # Also test: block-sharded INPUT → interleaved output
    # (shard the activation, but keep output interleaved)
    print(f"\n  --- Block-sharded input → Interleaved output ---")
    for grid_r, grid_c in grid_configs:
        label = f"BlkIn {grid_r}×{grid_c}→Intlvd"
        cores = grid_r * grid_c

        if hidden_tiles % grid_c != 0:
            print(f"  {label:<30} {cores:>6} {'SKIP':>12} {'--':>10} {'hidden_tiles not divisible':>10}")
            continue

        try:
            in_mem = make_block_shard_config((seq_tiles, hidden_tiles), grid_r, grid_c)
            if in_mem is None:
                print(f"  {label:<30} {cores:>6} {'SKIP':>12} {'--':>10} {'input not divisible':>10}")
                continue
            in_shard_h = (seq_tiles // grid_r) * 32
            in_shard_w = (hidden_tiles // grid_c) * 32
            shard_str = f"{in_shard_h}×{in_shard_w}"

            # Reshard input to block-sharded
            x_sharded = ttnn.to_memory_config(x_l1, in_mem)

            avg_ms = benchmark_config(label, x_sharded, w_dram, ttnn.L1_MEMORY_CONFIG, device)
            ratio = baseline_ms / avg_ms if avg_ms > 0 else 0
            status = "FASTER" if avg_ms < baseline_ms else "SLOWER"
            print(f"  {label:<30} {cores:>6} {shard_str:>12} {avg_ms:>10.4f} {ratio:>9.2f}x {status}")
            results.append((label, cores, shard_str, avg_ms, ratio))

            ttnn.deallocate(x_sharded)

        except Exception as e:
            err_msg = str(e)[:60]
            print(f"  {label:<30} {cores:>6} {'ERROR':>12} {err_msg}")
            results.append((label, cores, "ERROR", None, None))

    # Cleanup
    ttnn.deallocate(x_l1)
    ttnn.deallocate(w_dram)

    return results


# ============================================================
# Vision: [1,64,768] × [768,3072]
# seq_tiles=2, hidden_tiles=24, inter_tiles=96
# ============================================================
# Device compute grid is 8x7 (8 cols, 7 rows). Grid must fit within this.
# Valid: grid_cols <= 8, grid_rows <= 7
vision_grids = [
    (2, 2),   # 4 cores
    (2, 3),   # 6 cores
    (2, 4),   # 8 cores
    (2, 6),   # 12 cores
    (2, 8),   # 16 cores — max cols
    (1, 4),   # 4 cores — single row, more cols
    (1, 8),   # 8 cores — single row, max cols
]

# ============================================================
# Text: [1,96,512] × [512,2048]
# seq_tiles=3, hidden_tiles=16, inter_tiles=64
# ============================================================
text_grids = [
    (3, 2),   # 6 cores
    (3, 4),   # 12 cores
    (3, 8),   # 24 cores — max cols
    (1, 4),   # 4 cores — single row
    (1, 8),   # 8 cores — single row
]

print("=" * 70)
print("  BLOCK SHARD BENCHMARK — fc1 matmul")
print(f"  {N} timed runs per config, {WARMUP} warmup runs")
print("=" * 70)

vision_results = run_suite("Vision fc1", 64, 768, 3072, vision_grids)
text_results = run_suite("Text fc1", 96, 512, 2048, text_grids)

# ============================================================
# Final summary
# ============================================================
print(f"\n{'='*70}")
print("  SUMMARY")
print(f"{'='*70}")
print("\n  Vision — best configs:")
valid = [(r[0], r[3], r[4]) for r in vision_results if r[3] is not None]
for name, ms, ratio in sorted(valid, key=lambda x: x[1]):
    marker = " <<<" if ratio > 1.0 else ""
    print(f"    {name:<35} {ms:.4f} ms  ({ratio:.2f}x vs baseline){marker}")

print("\n  Text — best configs:")
valid = [(r[0], r[3], r[4]) for r in text_results if r[3] is not None]
for name, ms, ratio in sorted(valid, key=lambda x: x[1]):
    marker = " <<<" if ratio > 1.0 else ""
    print(f"    {name:<35} {ms:.4f} ms  ({ratio:.2f}x vs baseline){marker}")

ttnn.close_device(device)
print("\nDone.")
