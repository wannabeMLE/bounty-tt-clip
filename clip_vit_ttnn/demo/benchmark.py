# SPDX-License-Identifier: Apache-2.0
#
# Performance Benchmark for CLIP-ViT TTNN
# Runs all 3 stages and produces a comparison report.
#
# Usage:
#   python demo/benchmark.py

import time
import sys
import os

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ttnn
from tt.weight_loader import CLIPTTNNConfig, load_all_weights
from tt.clip_model import run_vision_encoder, run_text_encoder, compute_similarity
from reference.torch_clip import compute_pcc, get_vision_encoder_intermediates, get_text_encoder_intermediates


MODEL_NAME = "openai/clip-vit-base-patch32"
NUM_WARMUP = 2
NUM_RUNS = 10


def benchmark_stage(hf_model, processor, device, stage, pixel_values, text_inputs):
    """Benchmark a single stage, return timing and PCC results."""
    config = CLIPTTNNConfig.from_huggingface(hf_model.config)
    config.stage = stage

    params = load_all_weights(hf_model, device, config)

    # --- Vision Encoder ---
    # Warmup
    for _ in range(NUM_WARMUP):
        run_vision_encoder(pixel_values, params["vision"], config, device)

    # Benchmark
    vision_times = []
    for _ in range(NUM_RUNS):
        t0 = time.perf_counter()
        _, vision_embed = run_vision_encoder(pixel_values, params["vision"], config, device)
        vision_times.append(time.perf_counter() - t0)

    # --- Text Encoder ---
    # Warmup
    for _ in range(NUM_WARMUP):
        run_text_encoder(
            text_inputs["input_ids"], text_inputs["attention_mask"],
            params["text"], config, device
        )

    # Benchmark
    text_times = []
    for _ in range(NUM_RUNS):
        t0 = time.perf_counter()
        _, text_embed = run_text_encoder(
            text_inputs["input_ids"], text_inputs["attention_mask"],
            params["text"], config, device
        )
        text_times.append(time.perf_counter() - t0)

    # --- PCC ---
    ref_vision = get_vision_encoder_intermediates(hf_model, pixel_values)
    ref_text = get_text_encoder_intermediates(
        hf_model, text_inputs["input_ids"], text_inputs["attention_mask"]
    )

    vision_pcc = compute_pcc(ref_vision["projected"], vision_embed)
    text_pcc = compute_pcc(ref_text["projected"], text_embed)

    return {
        "stage": stage,
        "vision_avg_ms": sum(vision_times) / len(vision_times) * 1000,
        "vision_min_ms": min(vision_times) * 1000,
        "text_avg_ms": sum(text_times) / len(text_times) * 1000,
        "text_min_ms": min(text_times) * 1000,
        "vision_pcc": vision_pcc,
        "text_pcc": text_pcc,
    }


def main():
    print("=" * 70)
    print("  CLIP-ViT TTNN Performance Benchmark")
    print("=" * 70)

    # Load model
    print("\nLoading HuggingFace model...")
    hf_model = CLIPModel.from_pretrained(MODEL_NAME)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    hf_model.eval()

    # Open device
    print("Opening Tenstorrent device...")
    device = ttnn.open_device(device_id=0)

    try:
        # Prepare inputs
        from urllib.request import urlopen
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(urlopen(url))

        image_inputs = processor(images=image, return_tensors="pt")
        pixel_values = image_inputs["pixel_values"]

        config = CLIPTTNNConfig.from_huggingface(hf_model.config)
        text_inputs = processor(
            text="a photo of a cat", return_tensors="pt",
            padding="max_length", max_length=config.text_max_position_embeddings,
            truncation=True,
        )

        # PyTorch baseline
        print("\nPyTorch CPU baseline...")
        pt_times = []
        for _ in range(NUM_RUNS):
            t0 = time.perf_counter()
            with torch.no_grad():
                hf_model.get_image_features(pixel_values)
            pt_times.append(time.perf_counter() - t0)
        pt_vision_ms = sum(pt_times) / len(pt_times) * 1000

        # Benchmark each stage
        results = []
        for stage in [1, 2, 3]:
            print(f"\nBenchmarking Stage {stage}...")
            try:
                result = benchmark_stage(
                    hf_model, processor, device, stage, pixel_values, text_inputs
                )
                results.append(result)
            except Exception as e:
                print(f"  Stage {stage} failed: {e}")
                results.append({"stage": stage, "error": str(e)})

        # --- Report ---
        print("\n")
        print("=" * 70)
        print("  PERFORMANCE REPORT")
        print("=" * 70)
        print(f"\nPyTorch CPU vision encoder: {pt_vision_ms:.1f} ms (avg over {NUM_RUNS} runs)")
        print(f"Hardware: Tenstorrent Wormhole B0")
        print(f"Model: CLIP ViT-B/32 (openai/clip-vit-base-patch32)")
        print(f"Runs: {NUM_RUNS} (after {NUM_WARMUP} warmup)")
        print()

        # Table header
        print(f"{'Stage':<8} {'Vision Avg':>12} {'Vision Min':>12} {'Text Avg':>10} "
              f"{'Text Min':>10} {'V-PCC':>8} {'T-PCC':>8} {'Speedup':>10}")
        print("-" * 80)

        for r in results:
            if "error" in r:
                print(f"  {r['stage']:<6} ERROR: {r['error']}")
                continue

            speedup = pt_vision_ms / r["vision_avg_ms"] if r["vision_avg_ms"] > 0 else 0
            print(
                f"  {r['stage']:<6} {r['vision_avg_ms']:>10.1f}ms {r['vision_min_ms']:>10.1f}ms "
                f"{r['text_avg_ms']:>8.1f}ms {r['text_min_ms']:>8.1f}ms "
                f"{r['vision_pcc']:>7.4f} {r['text_pcc']:>7.4f} "
                f"{speedup:>8.2f}x"
            )

        print()
        print("PCC Thresholds: Stage 1 > 0.99, Stage 2 > 0.98, Stage 3 > 0.97")

        # Check pass/fail
        thresholds = {1: 0.99, 2: 0.98, 3: 0.97}
        all_pass = True
        for r in results:
            if "error" in r:
                all_pass = False
                continue
            stage = r["stage"]
            if r["vision_pcc"] < thresholds[stage] or r["text_pcc"] < thresholds[stage]:
                all_pass = False
                print(f"  FAIL: Stage {stage} PCC below threshold")

        if all_pass:
            print("\n  ALL STAGES PASSED!")
        print()

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
