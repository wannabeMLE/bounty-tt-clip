"""Diagnose: is the sharding regression from reshards or from slower matmul?

Tests:
1. Time the to_memory_config (interleaved → sharded, sharded → interleaved)
2. Time sharded matmul vs interleaved matmul
3. Test internal transition: interleaved input → sharded output (no explicit reshard)
"""
import sys, time, torch
import ttnn

sys.path.insert(0, ".")
from clip_vit_ttnn.tt.weight_loader import CLIPTTNNConfig

dispatch_config = ttnn.DispatchCoreConfig(type=ttnn.DispatchCoreType.WORKER)
device = ttnn.open_device(device_id=0, dispatch_core_config=dispatch_config)
device.enable_program_cache()

N = 100

config = CLIPTTNNConfig(stage=2)
shard_768 = config.get_vision_linear_shard_config(768)
shard_3072 = config.get_vision_linear_shard_config(3072)
lofi = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.LoFi)

# Create tensors
x_torch = torch.randn(1, 64, 768)
w_torch = torch.randn(768, 3072)

x_interleaved = ttnn.from_torch(x_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                 device=device, memory_config=ttnn.L1_MEMORY_CONFIG)
w = ttnn.from_torch(w_torch, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT,
                     device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

# ---- Test 1: Reshard cost (interleaved → sharded) ----
# Warmup
for _ in range(5):
    x_sharded = ttnn.to_memory_config(x_interleaved, shard_768)
    x_back = ttnn.to_memory_config(x_sharded, ttnn.L1_MEMORY_CONFIG)
    ttnn.deallocate(x_sharded)
    ttnn.deallocate(x_back)

# Measure interleaved → sharded
ttnn.synchronize_device(device)
t0 = time.perf_counter()
for _ in range(N):
    x_sharded = ttnn.to_memory_config(x_interleaved, shard_768)
    ttnn.deallocate(x_sharded)
ttnn.synchronize_device(device)
i2s_ms = (time.perf_counter() - t0) / N * 1000

# Measure sharded → interleaved
x_sharded = ttnn.to_memory_config(x_interleaved, shard_768)
ttnn.synchronize_device(device)
t0 = time.perf_counter()
for _ in range(N):
    x_back = ttnn.to_memory_config(x_sharded, ttnn.L1_MEMORY_CONFIG)
    ttnn.deallocate(x_back)
ttnn.synchronize_device(device)
s2i_ms = (time.perf_counter() - t0) / N * 1000
ttnn.deallocate(x_sharded)

print(f"=== Reshard costs (vision [1,64,768], {N} runs) ===")
print(f"  Interleaved → Sharded:  {i2s_ms:.3f} ms")
print(f"  Sharded → Interleaved:  {s2i_ms:.3f} ms")
print(f"  Per-layer overhead (3x): {(i2s_ms + s2i_ms * 2):.3f} ms")
print(f"  12-layer overhead:       {(i2s_ms + s2i_ms * 2) * 12:.3f} ms")

# ---- Test 2: Interleaved matmul vs sharded matmul ----
# Interleaved matmul
for _ in range(5):
    out = ttnn.linear(x_interleaved, w, compute_kernel_config=lofi,
                      memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn.deallocate(out)

ttnn.synchronize_device(device)
t0 = time.perf_counter()
for _ in range(N):
    out = ttnn.linear(x_interleaved, w, compute_kernel_config=lofi,
                      memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn.deallocate(out)
ttnn.synchronize_device(device)
interleaved_ms = (time.perf_counter() - t0) / N * 1000

# Sharded input → sharded output matmul
x_sharded = ttnn.to_memory_config(x_interleaved, shard_768)
for _ in range(5):
    out = ttnn.linear(x_sharded, w, compute_kernel_config=lofi,
                      memory_config=shard_3072)
    ttnn.deallocate(out)

ttnn.synchronize_device(device)
t0 = time.perf_counter()
for _ in range(N):
    out = ttnn.linear(x_sharded, w, compute_kernel_config=lofi,
                      memory_config=shard_3072)
    ttnn.deallocate(out)
ttnn.synchronize_device(device)
sharded_ms = (time.perf_counter() - t0) / N * 1000
ttnn.deallocate(x_sharded)

print(f"\n=== fc1 matmul [1,64,768]x[768,3072] ({N} runs) ===")
print(f"  Interleaved in → Interleaved out:  {interleaved_ms:.3f} ms")
print(f"  Sharded in → Sharded out:          {sharded_ms:.3f} ms")
print(f"  Difference:                         {sharded_ms - interleaved_ms:+.3f} ms")

# ---- Test 3: Internal transition (interleaved in → sharded out) ----
for _ in range(5):
    out = ttnn.linear(x_interleaved, w, compute_kernel_config=lofi,
                      memory_config=shard_3072)
    ttnn.deallocate(out)

ttnn.synchronize_device(device)
t0 = time.perf_counter()
for _ in range(N):
    out = ttnn.linear(x_interleaved, w, compute_kernel_config=lofi,
                      memory_config=shard_3072)
    ttnn.deallocate(out)
ttnn.synchronize_device(device)
internal_ms = (time.perf_counter() - t0) / N * 1000

print(f"\n=== Internal transition ({N} runs) ===")
print(f"  Interleaved in → Sharded out:      {internal_ms:.3f} ms")
print(f"  vs explicit reshard + sharded mm:   {i2s_ms + sharded_ms:.3f} ms")
print(f"  Savings from internal transition:   {(i2s_ms + sharded_ms) - internal_ms:.3f} ms")

# ---- Summary ----
total_explicit = i2s_ms + sharded_ms + s2i_ms  # reshard in + matmul + reshard out
print(f"\n=== Summary per fc1 call ===")
print(f"  Baseline (interleaved):             {interleaved_ms:.3f} ms")
print(f"  Sharded (reshard+mm+unshard):        {total_explicit:.3f} ms")
print(f"  Internal (intlvd→sharded out only): {internal_ms:.3f} ms")

ttnn.close_device(device)
