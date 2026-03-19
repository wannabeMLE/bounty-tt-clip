#!/usr/bin/env python3
"""
profile_layers.py -- Per-op profiling for CLIP-ViT TTNN encoder layers.

Development tool for sharding decisions. Shows latency and percentage breakdown
per operation within each encoder layer.

NOTE: This inserts ttnn.synchronize_device() between every op, which kills
pipelining. Numbers here are per-op isolation costs, NOT production throughput.
Use benchmark.py for submission-grade end-to-end numbers.

Usage:
    python profile_layers.py --stage 1
    python profile_layers.py --stage 2 --encoder both
"""

import argparse
import os
import sys
import torch
from PIL import Image
from urllib.request import urlopen
from transformers import CLIPModel, CLIPProcessor

import ttnn

from clip_vit_ttnn.tt.weight_loader import CLIPTTNNConfig, load_all_weights
from clip_vit_ttnn.tt.clip_model import run_vision_encoder, run_text_encoder

MODEL_NAME = "openai/clip-vit-base-patch32"
IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"

# Op categories for the summary table
OP_CATEGORY = {
    "layer_norm1": "LayerNorm",
    "qkv_linear": "Attention",
    "attn_scores": "Attention",
    "softmax": "Attention",
    "attn_context": "Attention",
    "concat_heads": "Attention",
    "out_proj": "Attention",
    "residual1_add": "Residual",
    "layer_norm2": "LayerNorm",
    "fc1_linear": "MLP",
    "quick_gelu": "MLP",
    "fc2_linear": "MLP",
    "residual2_add": "Residual",
}

# Canonical op order for display
OP_ORDER = list(OP_CATEGORY.keys())


def format_layer_table(layer_timings, encoder_name, stage, layer_idx):
    """Format a table for one layer's profiling results. Returns list of lines."""
    td = layer_timings[layer_idx]
    total = sum(td.values())
    if total == 0:
        return [f"  (no timing data for layer {layer_idx})"]

    lines = [f"\n{encoder_name} encoder layer {layer_idx} (stage {stage}):"]
    for op in OP_ORDER:
        if op in td:
            ms = td[op]
            pct = ms / total * 100
            cat = OP_CATEGORY.get(op, "?")
            lines.append(f"  {op:<20s} {ms:6.2f}ms {pct:5.1f}%  {cat}")
    lines.append(f"  {'─' * 40}")
    lines.append(f"  {'Total':<20s} {total:6.2f}ms")
    return lines


def print_layer_table(layer_timings, encoder_name, stage, layer_idx):
    for line in format_layer_table(layer_timings, encoder_name, stage, layer_idx):
        print(line)


def format_average_table(layer_timings, encoder_name, stage, num_layers):
    """Format averaged breakdown across all layers. Returns list of lines."""
    avg = {}
    for td in layer_timings:
        for op, ms in td.items():
            avg[op] = avg.get(op, 0) + ms
    for op in avg:
        avg[op] /= num_layers

    total = sum(avg.values())
    if total == 0:
        return []

    lines = [f"\n{encoder_name} encoder AVERAGE across {num_layers} layers (stage {stage}):"]
    for op in OP_ORDER:
        if op in avg:
            ms = avg[op]
            pct = ms / total * 100
            cat = OP_CATEGORY.get(op, "?")
            lines.append(f"  {op:<20s} {ms:6.2f}ms {pct:5.1f}%  {cat}")
    lines.append(f"  {'─' * 40}")
    lines.append(f"  {'Total':<20s} {total:6.2f}ms")

    cat_totals = {}
    for op in OP_ORDER:
        if op in avg:
            cat = OP_CATEGORY[op]
            cat_totals[cat] = cat_totals.get(cat, 0) + avg[op]

    lines.append(f"\n  Category breakdown:")
    for cat in ["Attention", "MLP", "LayerNorm", "Residual"]:
        if cat in cat_totals:
            ms = cat_totals[cat]
            pct = ms / total * 100
            lines.append(f"    {cat:<15s} {ms:6.2f}ms {pct:5.1f}%")
    return lines


def print_average_table(layer_timings, encoder_name, stage, num_layers):
    for line in format_average_table(layer_timings, encoder_name, stage, num_layers):
        print(line)


def main():
    parser = argparse.ArgumentParser(description="CLIP-ViT TTNN per-layer profiler")
    parser.add_argument("--stage", type=int, default=1, help="Stage (1 or 2)")
    parser.add_argument("--encoder", type=str, default="both",
                        choices=["vision", "text", "both"],
                        help="Which encoder to profile")
    parser.add_argument("--layer", type=int, default=None,
                        help="Profile specific layer index (default: all + average)")
    parser.add_argument("--output", type=str, default=None,
                        help="Save results to file (default: results/stage{N}_profile.txt)")
    args = parser.parse_args()

    if args.output is None:
        args.output = f"results/stage{args.stage}_profile.txt"

    print(f"{'=' * 55}")
    print(f"  CLIP-ViT TTNN Layer Profiler — Stage {args.stage}")
    print(f"{'=' * 55}")

    # Load model
    print("\nLoading model...")
    hf_model = CLIPModel.from_pretrained(MODEL_NAME, attn_implementation="eager")
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    hf_model.eval()

    # Prepare inputs
    image = Image.open(urlopen(IMAGE_URL)).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt")["pixel_values"]
    text_inputs = processor(
        text="a photo of a cat", return_tensors="pt", padding=True, truncation=True,
    )

    # Open device
    print("Opening device...")
    dispatch_config = ttnn.DispatchCoreConfig(type=ttnn.DispatchCoreType.WORKER)
    device = ttnn.open_device(device_id=0, dispatch_core_config=dispatch_config)

    try:
        config = CLIPTTNNConfig(stage=args.stage)
        params = load_all_weights(hf_model, device, config)

        # Enable program cache BEFORE any runs so ALL ops get cached
        device.enable_program_cache()

        # Warmup (compile kernels + populate program cache)
        print("Warmup run (compiling kernels)...")
        tt_v, _ = run_vision_encoder(pixel_values, params["vision"], config, device)
        ttnn.deallocate(tt_v)
        run_text_encoder(
            text_inputs["input_ids"], text_inputs["attention_mask"],
            params["text"], config, device,
        )

        # Collect all output lines for both console and file
        all_lines = [
            f"CLIP-ViT TTNN Layer Profile — Stage {args.stage}",
            f"{'=' * 55}",
        ]

        # Profile vision encoder
        if args.encoder in ("vision", "both"):
            num_layers = config.vision_num_layers
            vision_timings = [{} for _ in range(num_layers)]
            print(f"\nProfiling vision encoder ({num_layers} layers)...")
            run_vision_encoder(
                pixel_values, params["vision"], config, device,
                layer_timings=vision_timings,
            )

            if args.layer is not None:
                lines = format_layer_table(vision_timings, "Vision", args.stage, args.layer)
            else:
                lines = format_layer_table(vision_timings, "Vision", args.stage, 0)
                if num_layers > 1:
                    lines += format_layer_table(vision_timings, "Vision", args.stage, num_layers - 1)
                lines += format_average_table(vision_timings, "Vision", args.stage, num_layers)
            for line in lines:
                print(line)
            all_lines += lines

        # Profile text encoder
        if args.encoder in ("text", "both"):
            num_layers = config.text_num_layers
            text_timings = [{} for _ in range(num_layers)]
            print(f"\nProfiling text encoder ({num_layers} layers)...")
            run_text_encoder(
                text_inputs["input_ids"], text_inputs["attention_mask"],
                params["text"], config, device,
                layer_timings=text_timings,
            )

            if args.layer is not None:
                lines = format_layer_table(text_timings, "Text", args.stage, args.layer)
            else:
                lines = format_layer_table(text_timings, "Text", args.stage, 0)
                if num_layers > 1:
                    lines += format_layer_table(text_timings, "Text", args.stage, num_layers - 1)
                lines += format_average_table(text_timings, "Text", args.stage, num_layers)
            for line in lines:
                print(line)
            all_lines += lines

    finally:
        ttnn.close_device(device)

    # Save to file
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    with open(args.output, "w") as f:
        f.write("\n".join(all_lines) + "\n")
    print(f"\n  Saved to {args.output}")

    print(f"\n{'=' * 55}")
    print("  Done. Use benchmark.py for submission-grade numbers.")
    print(f"{'=' * 55}")


if __name__ == "__main__":
    main()
