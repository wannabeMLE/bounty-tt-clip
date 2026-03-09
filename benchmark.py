#!/usr/bin/env python3
"""
benchmark.py -- Benchmark CLIP-ViT TTNN: timing + PCC for a given stage.

Usage:
    python benchmark.py --stage 1
    python benchmark.py --stage 2
    python benchmark.py --stage 1 --output results/stage1.md
"""

import argparse
import datetime
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

NUM_WARMUP = 3
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


def run_benchmark(stage, output_path=None):
    print(f"{'='*65}")
    print(f"  CLIP-ViT TTNN Benchmark — Stage {stage}")
    print(f"  {datetime.datetime.now().isoformat()}")
    print(f"{'='*65}")

    # --- Load model ---
    print("\nLoading HuggingFace model...")
    hf_model = CLIPModel.from_pretrained(MODEL_NAME, attn_implementation="eager")
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    hf_model.eval()

    # --- Prepare inputs ---
    print("Downloading test image...")
    image = Image.open(urlopen(IMAGE_URL)).convert("RGB")
    image_inputs = processor(images=image, return_tensors="pt")
    pixel_values = image_inputs["pixel_values"]

    text_inputs = processor(
        text=TEXTS, return_tensors="pt", padding=True, truncation=True
    )

    # --- PyTorch baseline ---
    print("Running PyTorch CPU baseline...")
    # Warmup
    for _ in range(NUM_WARMUP):
        with torch.no_grad():
            _to_tensor(hf_model.get_image_features(pixel_values=pixel_values))

    pt_vision_times = []
    for _ in range(NUM_RUNS):
        t0 = time.perf_counter()
        with torch.no_grad():
            pt_vision_embed = _to_tensor(hf_model.get_image_features(pixel_values=pixel_values))
        pt_vision_times.append(time.perf_counter() - t0)

    pt_text_times = []
    for _ in range(NUM_RUNS):
        t0 = time.perf_counter()
        with torch.no_grad():
            pt_text_embed = _to_tensor(hf_model.get_text_features(**text_inputs))
        pt_text_times.append(time.perf_counter() - t0)

    # Full pipeline reference
    with torch.no_grad():
        ref_outputs = hf_model(**processor(text=TEXTS, images=image, return_tensors="pt", padding=True, truncation=True))
        ref_logits = ref_outputs.logits_per_image
        ref_probs = ref_logits.softmax(dim=-1)

    # --- Open device ---
    print("Opening Tenstorrent device...")
    device = ttnn.open_device(device_id=0)

    try:
        config = CLIPTTNNConfig(stage=stage)
        params = load_all_weights(hf_model, device, config)
        params["logit_scale"] = hf_model.logit_scale.data.clone()

        # --- Vision encoder benchmark ---
        print(f"\nBenchmarking vision encoder (stage {stage})...")
        for _ in range(NUM_WARMUP):
            tt_v, _ = run_vision_encoder(pixel_values, params["vision"], config, device)
            ttnn.deallocate(tt_v)

        vision_times = []
        for _ in range(NUM_RUNS):
            t0 = time.perf_counter()
            tt_v, vision_embed = run_vision_encoder(pixel_values, params["vision"], config, device)
            ttnn.deallocate(tt_v)
            vision_times.append(time.perf_counter() - t0)

        # --- Text encoder benchmark (per-text, batch=1) ---
        print(f"Benchmarking text encoder (stage {stage})...")
        # Warmup
        for _ in range(NUM_WARMUP):
            tt_t, _ = run_text_encoder(
                text_inputs["input_ids"][0:1], text_inputs["attention_mask"][0:1],
                params["text"], config, device,
            )
            ttnn.deallocate(tt_t)

        text_times = []
        for _ in range(NUM_RUNS):
            t0 = time.perf_counter()
            tt_t, text_embed_single = run_text_encoder(
                text_inputs["input_ids"][0:1], text_inputs["attention_mask"][0:1],
                params["text"], config, device,
            )
            ttnn.deallocate(tt_t)
            text_times.append(time.perf_counter() - t0)

        # --- Full pipeline (all texts) ---
        print("Running full pipeline...")
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

        # --- PCC ---
        vision_pcc = compute_pcc(pt_vision_embed, vision_embed_final)
        text_pcc = compute_pcc(pt_text_embed, text_embed_all[0:1])
        logits_pcc = compute_pcc(ref_logits, logits)

    finally:
        ttnn.close_device(device)

    # --- Results ---
    results = {
        "stage": stage,
        "timestamp": datetime.datetime.now().isoformat(),
        "hardware": "Tenstorrent Wormhole B0 (N300)",
        "model": MODEL_NAME,
        "num_warmup": NUM_WARMUP,
        "num_runs": NUM_RUNS,
        # PyTorch baseline
        "pt_vision_avg_ms": sum(pt_vision_times) / len(pt_vision_times) * 1000,
        "pt_vision_min_ms": min(pt_vision_times) * 1000,
        "pt_text_avg_ms": sum(pt_text_times) / len(pt_text_times) * 1000,
        "pt_text_min_ms": min(pt_text_times) * 1000,
        # TTNN
        "tt_vision_avg_ms": sum(vision_times) / len(vision_times) * 1000,
        "tt_vision_min_ms": min(vision_times) * 1000,
        "tt_text_avg_ms": sum(text_times) / len(text_times) * 1000,
        "tt_text_min_ms": min(text_times) * 1000,
        "tt_full_pipeline_ms": t_full * 1000,
        # PCC
        "vision_pcc": vision_pcc,
        "text_pcc": text_pcc,
        "logits_pcc": logits_pcc,
        # Predictions
        "ref_logits": ref_logits.tolist(),
        "tt_logits": logits.tolist(),
        "ref_probs": ref_probs.tolist(),
        "tt_probs": probs.tolist(),
        "predicted_idx": predicted_idx,
        "predicted_text": TEXTS[predicted_idx],
        "prediction_correct": predicted_idx == ref_probs.argmax(dim=-1).item(),
    }

    # Print summary
    print(f"\n{'='*65}")
    print(f"  Results — Stage {stage}")
    print(f"{'='*65}")
    print(f"")
    print(f"  PyTorch CPU:")
    print(f"    Vision: avg={results['pt_vision_avg_ms']:.1f}ms  min={results['pt_vision_min_ms']:.1f}ms")
    print(f"    Text:   avg={results['pt_text_avg_ms']:.1f}ms  min={results['pt_text_min_ms']:.1f}ms")
    print(f"")
    print(f"  TTNN (Stage {stage}):")
    print(f"    Vision: avg={results['tt_vision_avg_ms']:.1f}ms  min={results['tt_vision_min_ms']:.1f}ms")
    print(f"    Text:   avg={results['tt_text_avg_ms']:.1f}ms  min={results['tt_text_min_ms']:.1f}ms")
    print(f"    Full pipeline: {results['tt_full_pipeline_ms']:.1f}ms")
    print(f"")
    print(f"  PCC:")
    print(f"    Vision embedding: {results['vision_pcc']:.6f}")
    print(f"    Text embedding:   {results['text_pcc']:.6f}")
    print(f"    Logits:           {results['logits_pcc']:.6f}")
    print(f"")
    print(f"  Prediction: '{results['predicted_text']}' — {'CORRECT' if results['prediction_correct'] else 'WRONG'}")
    print(f"")

    speedup_vision = results["pt_vision_avg_ms"] / results["tt_vision_avg_ms"] if results["tt_vision_avg_ms"] > 0 else 0
    print(f"  Vision speedup vs PyTorch CPU: {speedup_vision:.2f}x")

    # Write markdown
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        write_markdown(results, output_path)
        print(f"\n  Results written to {output_path}")

    return results


def write_markdown(r, path):
    speedup_v = r["pt_vision_avg_ms"] / r["tt_vision_avg_ms"] if r["tt_vision_avg_ms"] > 0 else 0
    speedup_t = r["pt_text_avg_ms"] / r["tt_text_avg_ms"] if r["tt_text_avg_ms"] > 0 else 0

    md = f"""# CLIP-ViT TTNN Benchmark — Stage {r['stage']}

**Date:** {r['timestamp']}
**Hardware:** {r['hardware']}
**Model:** {r['model']}
**Runs:** {r['num_runs']} (after {r['num_warmup']} warmup)

## Timing

| Component | PyTorch CPU (avg) | PyTorch CPU (min) | TTNN (avg) | TTNN (min) | Speedup |
|-----------|-------------------|-------------------|------------|------------|---------|
| Vision encoder | {r['pt_vision_avg_ms']:.1f} ms | {r['pt_vision_min_ms']:.1f} ms | {r['tt_vision_avg_ms']:.1f} ms | {r['tt_vision_min_ms']:.1f} ms | {speedup_v:.2f}x |
| Text encoder | {r['pt_text_avg_ms']:.1f} ms | {r['pt_text_min_ms']:.1f} ms | {r['tt_text_avg_ms']:.1f} ms | {r['tt_text_min_ms']:.1f} ms | {speedup_t:.2f}x |
| Full pipeline | — | — | {r['tt_full_pipeline_ms']:.1f} ms | — | — |

## Accuracy (PCC)

| Metric | PCC |
|--------|-----|
| Vision embedding | {r['vision_pcc']:.6f} |
| Text embedding | {r['text_pcc']:.6f} |
| Logits | {r['logits_pcc']:.6f} |

## Prediction

| | Value |
|--|-------|
| Predicted | "{r['predicted_text']}" |
| Correct | {'Yes' if r['prediction_correct'] else 'No'} |
| TTNN logits | {r['tt_logits']} |
| PyTorch logits | {r['ref_logits']} |
| TTNN probs | {r['tt_probs']} |
| PyTorch probs | {r['ref_probs']} |

## Configuration

- Memory: {'DRAM interleaved' if r['stage'] == 1 else 'L1 interleaved' if r['stage'] == 2 else 'L1 + sharded'}
- Math fidelity: {'HiFi4' if r['stage'] == 1 else 'LoFi'}
- Weight dtype: {'bfloat16' if r['stage'] == 1 else 'bfloat8_b'}
- GELU: {'separate' if r['stage'] == 1 else 'fused in fc1 linear'}
- Softmax: native ttnn.softmax
"""
    with open(path, "w") as f:
        f.write(md)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLIP-ViT TTNN Benchmark")
    parser.add_argument("--stage", type=int, default=1, help="Stage to benchmark (1, 2, or 3)")
    parser.add_argument("--output", type=str, default=None, help="Output markdown file path")
    args = parser.parse_args()

    if args.output is None:
        args.output = f"results/stage{args.stage}_benchmark.md"

    run_benchmark(args.stage, args.output)
