"""Isolate DRAM vs L1, HiFi4 vs LoFi, bfloat16 vs bfloat8_b on a single op.

Sync outside the 100-run loop so sync overhead is amortized to ~0.
This gives real per-op timing at 100x better resolution than the profiler.
"""
import time
import torch
import ttnn

dispatch_config = ttnn.DispatchCoreConfig(type=ttnn.DispatchCoreType.WORKER)
device = ttnn.open_device(device_id=0, dispatch_core_config=dispatch_config)
device.enable_program_cache()

# Create input + weight tensors (fc1 shape: [1, 64, 768] x [768, 3072])
x_torch = torch.randn(1, 64, 768)
w_torch = torch.randn(768, 3072)

hifi4 = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4)
lofi = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.LoFi)

N = 100

# ---- Test 1: DRAM interleaved, HiFi4, bfloat16 ----
x_dram = ttnn.from_torch(x_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                          device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
w_dram = ttnn.from_torch(w_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                          device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

for _ in range(5):
    out = ttnn.linear(x_dram, w_dram, compute_kernel_config=hifi4,
                      memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(out)

ttnn.synchronize_device(device)
t0 = time.perf_counter()
for _ in range(N):
    out = ttnn.linear(x_dram, w_dram, compute_kernel_config=hifi4,
                      memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(out)
ttnn.synchronize_device(device)
dram_hifi4_ms = (time.perf_counter() - t0) / N * 1000

ttnn.deallocate(x_dram)
ttnn.deallocate(w_dram)

# ---- Test 2: L1 interleaved, HiFi4, bfloat16 (isolate memory only) ----
x_l1_16 = ttnn.from_torch(x_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=ttnn.L1_MEMORY_CONFIG)
w_l1_16 = ttnn.from_torch(w_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=ttnn.L1_MEMORY_CONFIG)

for _ in range(5):
    out = ttnn.linear(x_l1_16, w_l1_16, compute_kernel_config=hifi4,
                      memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn.deallocate(out)

ttnn.synchronize_device(device)
t0 = time.perf_counter()
for _ in range(N):
    out = ttnn.linear(x_l1_16, w_l1_16, compute_kernel_config=hifi4,
                      memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn.deallocate(out)
ttnn.synchronize_device(device)
l1_hifi4_ms = (time.perf_counter() - t0) / N * 1000

ttnn.deallocate(x_l1_16)
ttnn.deallocate(w_l1_16)

# ---- Test 3: L1 interleaved, LoFi, bfloat8_b (all Stage 2 optimizations) ----
x_l1 = ttnn.from_torch(x_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                        device=device, memory_config=ttnn.L1_MEMORY_CONFIG)
w_l1 = ttnn.from_torch(w_torch, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT,
                        device=device, memory_config=ttnn.L1_MEMORY_CONFIG)

for _ in range(5):
    out = ttnn.linear(x_l1, w_l1, compute_kernel_config=lofi,
                      memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn.deallocate(out)

ttnn.synchronize_device(device)
t0 = time.perf_counter()
for _ in range(N):
    out = ttnn.linear(x_l1, w_l1, compute_kernel_config=lofi,
                      memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn.deallocate(out)
ttnn.synchronize_device(device)
l1_lofi_ms = (time.perf_counter() - t0) / N * 1000

ttnn.deallocate(x_l1)
ttnn.deallocate(w_l1)

# ---- Test 4: DRAM interleaved, LoFi, bfloat8_b (isolate dtype/fidelity only) ----
x_dram2 = ttnn.from_torch(x_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
w_dram2 = ttnn.from_torch(w_torch, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

for _ in range(5):
    out = ttnn.linear(x_dram2, w_dram2, compute_kernel_config=lofi,
                      memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(out)

ttnn.synchronize_device(device)
t0 = time.perf_counter()
for _ in range(N):
    out = ttnn.linear(x_dram2, w_dram2, compute_kernel_config=lofi,
                      memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(out)
ttnn.synchronize_device(device)
dram_lofi_ms = (time.perf_counter() - t0) / N * 1000

ttnn.deallocate(x_dram2)
ttnn.deallocate(w_dram2)

print(f"\nfc1 linear [1,64,768] x [768,3072] — {N} runs each:")
print(f"  DRAM + HiFi4 + bf16:  {dram_hifi4_ms:.3f} ms")
print(f"  DRAM + LoFi  + bf8:   {dram_lofi_ms:.3f} ms  (dtype/fidelity only)")
print(f"  L1   + HiFi4 + bf16:  {l1_hifi4_ms:.3f} ms  (memory only)")
print(f"  L1   + LoFi  + bf8:   {l1_lofi_ms:.3f} ms  (all optimizations)")
print()
print(f"  Effect of DRAM→L1 (holding HiFi4+bf16):  {dram_hifi4_ms/l1_hifi4_ms:.2f}x")
print(f"  Effect of DRAM→L1 (holding LoFi+bf8):    {dram_lofi_ms/l1_lofi_ms:.2f}x")
print(f"  Effect of HiFi4→LoFi+bf8 (holding DRAM): {dram_hifi4_ms/dram_lofi_ms:.2f}x")
print(f"  Effect of HiFi4→LoFi+bf8 (holding L1):   {l1_hifi4_ms/l1_lofi_ms:.2f}x")
print(f"  Total Stage1→Stage2:                      {dram_hifi4_ms/l1_lofi_ms:.2f}x")

ttnn.close_device(device)
