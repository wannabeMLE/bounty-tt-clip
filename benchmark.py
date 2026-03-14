#!/usr/bin/env python3
"""
benchmark.py -- Benchmark CLIP-ViT TTNN: timing + PCC for a given stage.

Tier 1: End-to-end latency, PCC, speedup vs CPU, speedup vs Stage 1
Tier 2: Compile time vs cached inference time

Usage:
    python benchmark.py --stage 1
    python benchmark.py --stage 2
    python benchmark.py --stage 1 --output results/stage1_benchmark.md
"""

import argparse
import datetime
import json
import os
import sys
import time

import torch
from PIL import Image
from urllib.request import urlopen
from transformers import CLIPModel, CLIPProcessor

import ttnn

from clip_vit_ttnn.tt.weight_loader import CLIPTTNNConfig, load_all_weights
from clip_vit_ttnn.tt.clip_model import (
    run_vision_encoder,
    run_text_encoder,
    compute_similarity,
)

MODEL_NAME = "openai/clip-vit-base-patch32"
IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"
TEXTS = ["a photo of a cat", "a photo of a dog", "a photo of a car"]

NUM_WARMUP = 2
NUM_RUNS = 10


def compute_pcc(a, b):
    a = a.detach().float().flatten()
    b = b.detach().float().flatten()
    if a.numel() == 0:
        return float("nan")
    min_len = min(len(a), len(b))
    return torch.corrcoef(torch.stack([a[:min_len], b[:min_len]]))[0, 1].item()


def _to_tensor(output):
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "pooler_output"):
        return output.pooler_output
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    return output


def load_stage1_results():
    """Load Stage 1 benchmark results for cross-stage comparison."""
    path = "results/stage1_benchmark.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def run_benchmark(stage, output_path=None):
    print(f"{'='*65}")
    print(f"  CLIP-ViT TTNN Benchmark — Stage {stage}")
    print(f"  {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*65}")

    # --- Load model ---
    print("\nLoading HuggingFace model...")
    hf_model = CLIPModel.from_pretrained(MODEL_NAME, attn_implementation="eager")
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    hf_model.eval()

    # --- Prepare inputs ---
    print("Downloading test image...")
    image = Image.open(urlopen(IMAGE_URL)).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt")["pixel_values"]
    text_inputs = processor(text=TEXTS, return_tensors="pt", padding=True, truncation=True)

    # --- PyTorch CPU baseline ---
    print("Running PyTorch CPU baseline...")
    for _ in range(NUM_WARMUP):
        with torch.no_grad():
            _to_tensor(hf_model.get_image_features(pixel_values=pixel_values))

    pt_vision_times = []
    for _ in range(NUM_RUNS):
        t0 = time.perf_counter()
        with torch.no_grad():
            pt_vision_embed = _to_tensor(hf_model.get_image_features(pixel_values=pixel_values))
        pt_vision_times.append(time.perf_counter() - t0)

    for _ in range(NUM_WARMUP):
        with torch.no_grad():
            _to_tensor(hf_model.get_text_features(**text_inputs))

    pt_text_times = []
    for _ in range(NUM_RUNS):
        t0 = time.perf_counter()
        with torch.no_grad():
            pt_text_embed = _to_tensor(hf_model.get_text_features(**text_inputs))
        pt_text_times.append(time.perf_counter() - t0)

    with torch.no_grad():
        ref_outputs = hf_model(**processor(text=TEXTS, images=image, return_tensors="pt", padding=True, truncation=True))
        ref_logits = ref_outputs.logits_per_image
        ref_probs = ref_logits.softmax(dim=-1)

    # --- Open TT device ---
    print("Opening Tenstorrent device...")
    dispatch_config = ttnn.DispatchCoreConfig(type=ttnn.DispatchCoreType.WORKER)
    device = ttnn.open_device(device_id=0, dispatch_core_config=dispatch_config)

    try:
        config = CLIPTTNNConfig(stage=stage)
        params = load_all_weights(hf_model, device, config)
        params["logit_scale"] = hf_model.logit_scale.data.clone()

        # =====================================================================
        # Tier 2: Compile time — first run includes kernel compilation
        # =====================================================================
        print(f"\n[Tier 2] Measuring compile time (first run)...")
        t0 = time.perf_counter()
        tt_v, _ = run_vision_encoder(pixel_values, params["vision"], config, device)
        ttnn.deallocate(tt_v)
        vision_compile_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        tt_t, _ = run_text_encoder(
            text_inputs["input_ids"][0:1], text_inputs["attention_mask"][0:1],
            params["text"], config, device,
        )
        ttnn.deallocate(tt_t)
        text_compile_ms = (time.perf_counter() - t0) * 1000

        print(f"  Vision compile: {vision_compile_ms:.1f} ms")
        print(f"  Text compile:   {text_compile_ms:.1f} ms")

        # =====================================================================
        # Tier 2: Cached time — second run uses compiled kernels
        # =====================================================================
        t0 = time.perf_counter()
        tt_v, _ = run_vision_encoder(pixel_values, params["vision"], config, device)
        ttnn.deallocate(tt_v)
        vision_cached_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        tt_t, _ = run_text_encoder(
            text_inputs["input_ids"][0:1], text_inputs["attention_mask"][0:1],
            params["text"], config, device,
        )
        ttnn.deallocate(tt_t)
        text_cached_ms = (time.perf_counter() - t0) * 1000

        print(f"  Vision cached:  {vision_cached_ms:.1f} ms (compile overhead: {vision_compile_ms - vision_cached_ms:.1f} ms)")
        print(f"  Text cached:    {text_cached_ms:.1f} ms (compile overhead: {text_compile_ms - text_cached_ms:.1f} ms)")

        # =====================================================================
        # Tier 1: Steady-state latency — avg/min over NUM_RUNS
        # =====================================================================
        print(f"\n[Tier 1] Benchmarking vision encoder ({NUM_RUNS} runs)...")
        vision_times = []
        for _ in range(NUM_RUNS):
            t0 = time.perf_counter()
            tt_v, vision_embed = run_vision_encoder(pixel_values, params["vision"], config, device)
            ttnn.deallocate(tt_v)
            vision_times.append(time.perf_counter() - t0)

        print(f"[Tier 1] Benchmarking text encoder ({NUM_RUNS} runs)...")
        text_times = []
        for _ in range(NUM_RUNS):
            t0 = time.perf_counter()
            tt_t, text_embed_single = run_text_encoder(
                text_inputs["input_ids"][0:1], text_inputs["attention_mask"][0:1],
                params["text"], config, device,
            )
            ttnn.deallocate(tt_t)
            text_times.append(time.perf_counter() - t0)

        # Full pipeline (vision + 3 texts + similarity)
        print("[Tier 1] Full pipeline...")
        t_full_start = time.perf_counter()
        tt_v, vision_embed_final = run_vision_encoder(pixel_values, params["vision"], config, device)
        ttnn.deallocate(tt_v)

        text_embeds = []
        for t in range(text_inputs["input_ids"].shape[0]):
            tt_t, te = run_text_encoder(
                text_inputs["input_ids"][t:t+1], text_inputs["attention_mask"][t:t+1],
                params["text"], config, device,
            )
            ttnn.deallocate(tt_t)
            text_embeds.append(te)
        text_embed_all = torch.cat(text_embeds, dim=0)

        logits = compute_similarity(vision_embed_final, text_embed_all, params["logit_scale"])
        probs = logits.softmax(dim=-1)
        t_full = time.perf_counter() - t_full_start
        predicted_idx = probs.argmax(dim=-1).item()

        # PCC
        vision_pcc = compute_pcc(pt_vision_embed, vision_embed_final)
        text_pcc = compute_pcc(pt_text_embed, text_embed_all[0:1])
        logits_pcc = compute_pcc(ref_logits, logits)

    finally:
        ttnn.close_device(device)

    # =====================================================================
    # Compute all metrics
    # =====================================================================
    r = {
        "stage": stage,
        "timestamp": datetime.datetime.now().isoformat(),
        "hardware": "Tenstorrent Wormhole B0 (N300)",
        "model": MODEL_NAME,
        "num_runs": NUM_RUNS,
        # PyTorch CPU
        "pt_vision_avg_ms": sum(pt_vision_times) / len(pt_vision_times) * 1000,
        "pt_vision_min_ms": min(pt_vision_times) * 1000,
        "pt_text_avg_ms": sum(pt_text_times) / len(pt_text_times) * 1000,
        "pt_text_min_ms": min(pt_text_times) * 1000,
        # TTNN steady-state
        "tt_vision_avg_ms": sum(vision_times) / len(vision_times) * 1000,
        "tt_vision_min_ms": min(vision_times) * 1000,
        "tt_text_avg_ms": sum(text_times) / len(text_times) * 1000,
        "tt_text_min_ms": min(text_times) * 1000,
        "tt_full_pipeline_ms": t_full * 1000,
        # Compile vs cached
        "vision_compile_ms": vision_compile_ms,
        "vision_cached_ms": vision_cached_ms,
        "text_compile_ms": text_compile_ms,
        "text_cached_ms": text_cached_ms,
        # Throughput
        "vision_fps": 1000.0 / (sum(vision_times) / len(vision_times) * 1000),
        # PCC
        "vision_pcc": vision_pcc,
        "text_pcc": text_pcc,
        "logits_pcc": logits_pcc,
        # Prediction
        "predicted_text": TEXTS[predicted_idx],
        "prediction_correct": predicted_idx == ref_probs.argmax(dim=-1).item(),
        "tt_logits": logits.tolist(),
        "ref_logits": ref_logits.tolist(),
        "tt_probs": probs.tolist(),
        "ref_probs": ref_probs.tolist(),
    }

    # Speedup vs CPU
    r["speedup_vision_vs_cpu"] = r["pt_vision_avg_ms"] / r["tt_vision_avg_ms"]
    r["speedup_text_vs_cpu"] = r["pt_text_avg_ms"] / r["tt_text_avg_ms"]

    # Speedup vs Stage 1
    stage1 = load_stage1_results()
    if stage1 and stage > 1:
        r["speedup_vision_vs_stage1"] = stage1["tt_vision_avg_ms"] / r["tt_vision_avg_ms"]
        r["speedup_text_vs_stage1"] = stage1["tt_text_avg_ms"] / r["tt_text_avg_ms"]
        r["stage1_vision_avg_ms"] = stage1["tt_vision_avg_ms"]
        r["stage1_text_avg_ms"] = stage1["tt_text_avg_ms"]

    # =====================================================================
    # Print results
    # =====================================================================
    print(f"\n{'='*65}")
    print(f"  RESULTS — Stage {stage}")
    print(f"{'='*65}")

    print(f"\n  --- Latency (Tier 1) ---")
    print(f"  {'':30s} {'avg':>8s} {'min':>8s}")
    print(f"  {'PyTorch CPU vision':30s} {r['pt_vision_avg_ms']:7.1f}ms {r['pt_vision_min_ms']:7.1f}ms")
    print(f"  {'PyTorch CPU text':30s} {r['pt_text_avg_ms']:7.1f}ms {r['pt_text_min_ms']:7.1f}ms")
    print(f"  {'TTNN vision':30s} {r['tt_vision_avg_ms']:7.1f}ms {r['tt_vision_min_ms']:7.1f}ms")
    print(f"  {'TTNN text':30s} {r['tt_text_avg_ms']:7.1f}ms {r['tt_text_min_ms']:7.1f}ms")
    print(f"  {'TTNN full pipeline':30s} {r['tt_full_pipeline_ms']:7.1f}ms")
    print(f"  {'Throughput (vision)':30s} {r['vision_fps']:7.1f} img/s")

    print(f"\n  --- Speedup ---")
    print(f"  {'Vision vs PyTorch CPU':30s} {r['speedup_vision_vs_cpu']:.2f}x")
    print(f"  {'Text vs PyTorch CPU':30s} {r['speedup_text_vs_cpu']:.2f}x")
    if "speedup_vision_vs_stage1" in r:
        print(f"  {'Vision vs Stage 1':30s} {r['speedup_vision_vs_stage1']:.2f}x")
        print(f"  {'Text vs Stage 1':30s} {r['speedup_text_vs_stage1']:.2f}x")

    print(f"\n  --- Compile vs Cached (Tier 2) ---")
    print(f"  {'':20s} {'compile':>10s} {'cached':>10s} {'overhead':>10s}")
    print(f"  {'Vision':20s} {r['vision_compile_ms']:9.1f}ms {r['vision_cached_ms']:9.1f}ms {r['vision_compile_ms']-r['vision_cached_ms']:9.1f}ms")
    print(f"  {'Text':20s} {r['text_compile_ms']:9.1f}ms {r['text_cached_ms']:9.1f}ms {r['text_compile_ms']-r['text_cached_ms']:9.1f}ms")

    print(f"\n  --- Accuracy (PCC) ---")
    pcc_ok = lambda v: "PASS" if v >= 0.98 else "FAIL"
    print(f"  {'Vision embedding':20s} {r['vision_pcc']:.6f}  {pcc_ok(r['vision_pcc'])}")
    print(f"  {'Text embedding':20s} {r['text_pcc']:.6f}  {pcc_ok(r['text_pcc'])}")
    print(f"  {'Logits':20s} {r['logits_pcc']:.6f}  {pcc_ok(r['logits_pcc'])}")

    print(f"\n  --- Prediction ---")
    print(f"  Predicted: \"{r['predicted_text']}\" — {'CORRECT' if r['prediction_correct'] else 'WRONG'}")

    config_desc = {
        1: "DRAM interleaved, HiFi4, bfloat16 weights, separate GELU",
        2: "L1 interleaved, LoFi, bfloat8_b weights, fused GELU",
        3: "L1 interleaved, LoFi, bfloat8_b weights, fused GELU, SDPA",
    }
    print(f"\n  --- Config ---")
    print(f"  {config_desc.get(stage, 'unknown')}")
    print(f"{'='*65}")

    # Save JSON (for cross-stage comparison)
    json_path = output_path.replace(".md", ".json") if output_path else f"results/stage{stage}_benchmark.json"
    os.makedirs(os.path.dirname(json_path) if os.path.dirname(json_path) else ".", exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(r, f, indent=2)
    print(f"\n  JSON saved to {json_path}")

    # Save markdown
    if output_path:
        write_markdown(r, output_path)
        print(f"  Markdown saved to {output_path}")

    return r


def write_markdown(r, path):
    speedup_v = r["speedup_vision_vs_cpu"]
    speedup_t = r["speedup_text_vs_cpu"]

    stage_cmp = ""
    if "speedup_vision_vs_stage1" in r:
        stage_cmp = f"""
## Speedup vs Stage 1

| Component | Stage 1 (avg) | Stage {r['stage']} (avg) | Speedup |
|-----------|---------------|-------------|---------|
| Vision encoder | {r['stage1_vision_avg_ms']:.1f} ms | {r['tt_vision_avg_ms']:.1f} ms | {r['speedup_vision_vs_stage1']:.2f}x |
| Text encoder | {r['stage1_text_avg_ms']:.1f} ms | {r['tt_text_avg_ms']:.1f} ms | {r['speedup_text_vs_stage1']:.2f}x |
"""

    config_desc = {
        1: ("DRAM interleaved", "HiFi4", "bfloat16", "separate"),
        2: ("L1 interleaved", "LoFi", "bfloat8_b", "fused in fc1"),
        3: ("L1 interleaved", "LoFi", "bfloat8_b", "fused in fc1 + SDPA"),
    }
    mem, math, wdtype, gelu = config_desc.get(r["stage"], ("?", "?", "?", "?"))

    md = f"""# CLIP-ViT TTNN Benchmark — Stage {r['stage']}

**Date:** {r['timestamp']}
**Hardware:** {r['hardware']}
**Model:** {r['model']}
**Runs:** {r['num_runs']} (after compile warmup)

## Latency

| Component | PyTorch CPU (avg) | PyTorch CPU (min) | TTNN (avg) | TTNN (min) | Speedup vs CPU |
|-----------|-------------------|-------------------|------------|------------|----------------|
| Vision encoder | {r['pt_vision_avg_ms']:.1f} ms | {r['pt_vision_min_ms']:.1f} ms | {r['tt_vision_avg_ms']:.1f} ms | {r['tt_vision_min_ms']:.1f} ms | {speedup_v:.2f}x |
| Text encoder | {r['pt_text_avg_ms']:.1f} ms | {r['pt_text_min_ms']:.1f} ms | {r['tt_text_avg_ms']:.1f} ms | {r['tt_text_min_ms']:.1f} ms | {speedup_t:.2f}x |
| Full pipeline | — | — | {r['tt_full_pipeline_ms']:.1f} ms | — | — |

**Throughput:** {r['vision_fps']:.1f} images/sec (vision encoder)
{stage_cmp}
## Compile vs Cached (Program Cache)

| Component | First run (compile) | Cached run | Compile overhead |
|-----------|--------------------:|----------:|-----------------:|
| Vision encoder | {r['vision_compile_ms']:.1f} ms | {r['vision_cached_ms']:.1f} ms | {r['vision_compile_ms'] - r['vision_cached_ms']:.1f} ms |
| Text encoder | {r['text_compile_ms']:.1f} ms | {r['text_cached_ms']:.1f} ms | {r['text_compile_ms'] - r['text_cached_ms']:.1f} ms |

## Accuracy (PCC)

| Metric | PCC | Status |
|--------|-----|--------|
| Vision embedding | {r['vision_pcc']:.6f} | {'PASS' if r['vision_pcc'] >= 0.98 else 'FAIL'} (>= 0.98) |
| Text embedding | {r['text_pcc']:.6f} | {'PASS' if r['text_pcc'] >= 0.98 else 'FAIL'} (>= 0.98) |
| Logits | {r['logits_pcc']:.6f} | {'PASS' if r['logits_pcc'] >= 0.98 else 'FAIL'} (>= 0.98) |

## Prediction

| | Value |
|--|-------|
| Predicted | "{r['predicted_text']}" |
| Correct | {'Yes' if r['prediction_correct'] else 'No'} |
| TTNN logits | {r['tt_logits']} |
| PyTorch logits | {r['ref_logits']} |

## Configuration

| Setting | Value |
|---------|-------|
| Memory | {mem} |
| Math fidelity | {math} |
| Weight dtype | {wdtype} |
| GELU | {gelu} |
| Compute grid | 8x7 (56 cores) |
| Dispatch | WORKER |
"""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w") as f:
        f.write(md)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLIP-ViT TTNN Benchmark")
    parser.add_argument("--stage", type=int, default=1, help="Stage (1, 2, or 3)")
    parser.add_argument("--output", type=str, default=None, help="Output markdown path")
    args = parser.parse_args()

    if args.output is None:
        args.output = f"results/stage{args.stage}_benchmark.md"

    run_benchmark(args.stage, args.output)
