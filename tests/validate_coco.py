#!/usr/bin/env python3
"""
validate_coco.py -- COCO zero-shot classification: PyTorch vs TTNN.

Streams COCO val2017 images, classifies each with 80 COCO classes,
compares TTNN predictions against PyTorch reference.

Usage:
    python tests/validate_coco.py --stage 1 --num_images 20
    python tests/validate_coco.py --stage 2 --num_images 20
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import ttnn
from transformers import CLIPModel, CLIPProcessor
from datasets import load_dataset

from clip_vit_ttnn.tt.weight_loader import CLIPTTNNConfig, load_all_weights
from clip_vit_ttnn.tt.clip_model import run_vision_encoder, run_text_encoder, compute_similarity

MODEL_NAME = "openai/clip-vit-base-patch32"

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]


def _to_tensor(output):
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "pooler_output"):
        return output.pooler_output
    return output.last_hidden_state


def compute_pcc(a, b):
    a = a.detach().float().flatten()
    b = b.detach().float().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--num_images", type=int, default=20)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.output is None:
        args.output = f"results/coco_stage{args.stage}.md"

    print("=" * 70)
    print(f"  COCO Zero-Shot Validation — Stage {args.stage}")
    print(f"  {args.num_images} images, {len(COCO_CLASSES)} classes")
    print("=" * 70)

    # Load model
    print("\nLoading model...")
    hf_model = CLIPModel.from_pretrained(MODEL_NAME, attn_implementation="eager")
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    hf_model.eval()

    # Pre-encode 80 COCO prompts with PyTorch
    print("Encoding 80 COCO class prompts (PyTorch)...")
    prompts = [f"a photo of a {c}" for c in COCO_CLASSES]
    text_inputs = processor(text=prompts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        pt_text_features = _to_tensor(hf_model.get_text_features(**text_inputs))
        pt_text_norm = pt_text_features / pt_text_features.norm(p=2, dim=-1, keepdim=True)

    # Open TT device
    print("Opening TT device...")
    dispatch_config = ttnn.DispatchCoreConfig(type=ttnn.DispatchCoreType.WORKER)
    device = ttnn.open_device(device_id=0, dispatch_core_config=dispatch_config)
    config = CLIPTTNNConfig(stage=args.stage)
    params = load_all_weights(hf_model, device, config)
    params["logit_scale"] = hf_model.logit_scale.data.clone()

    # Pre-encode 80 COCO prompts with TTNN
    print("Encoding 80 COCO class prompts (TTNN)...")
    ttnn_text_embeds = []
    for t in range(len(prompts)):
        tt_t, t_torch = run_text_encoder(
            text_inputs["input_ids"][t:t+1],
            text_inputs["attention_mask"][t:t+1],
            params["text"], config, device,
        )
        ttnn.deallocate(tt_t)
        ttnn_text_embeds.append(t_torch)
    ttnn_text_all = torch.cat(ttnn_text_embeds, dim=0)
    ttnn_text_norm = ttnn_text_all / ttnn_text_all.norm(p=2, dim=-1, keepdim=True)

    # Load COCO images
    print(f"\nStreaming {args.num_images} COCO val2017 images...")
    ds = load_dataset("detection-datasets/coco", split="val", streaming=True)

    logit_scale = hf_model.logit_scale.exp()

    # Results
    results = []
    pt_matches_ttnn = 0
    total = 0

    print(f"\n{'#':>3}  {'PyTorch Top-1':<22} {'TTNN Top-1':<22} {'Match':>5}  {'PCC':>7}")
    print("-" * 70)

    for i, sample in enumerate(ds):
        if i >= args.num_images:
            break

        img = sample["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")

        img_inputs = processor(images=img, return_tensors="pt")
        pixel_values = img_inputs["pixel_values"]

        # PyTorch
        with torch.no_grad():
            pt_img = _to_tensor(hf_model.get_image_features(pixel_values=pixel_values))
            pt_img_norm = pt_img / pt_img.norm(p=2, dim=-1, keepdim=True)
            pt_logits = (pt_img_norm @ pt_text_norm.T) * logit_scale
            pt_probs = pt_logits.softmax(dim=-1)[0]
            pt_top = pt_probs.argmax().item()

        # TTNN
        tt_v, vis_torch = run_vision_encoder(pixel_values, params["vision"], config, device)
        ttnn.deallocate(tt_v)
        ttnn_logits = compute_similarity(vis_torch, ttnn_text_all, params["logit_scale"])
        ttnn_probs = ttnn_logits.softmax(dim=-1)[0]
        ttnn_top = ttnn_probs.argmax().item()

        pcc = compute_pcc(pt_logits, ttnn_logits)
        match = pt_top == ttnn_top
        if match:
            pt_matches_ttnn += 1
        total += 1

        pt_label = f"{COCO_CLASSES[pt_top]} ({pt_probs[pt_top]:.0%})"
        tt_label = f"{COCO_CLASSES[ttnn_top]} ({ttnn_probs[ttnn_top]:.0%})"
        print(f"{i+1:>3}  {pt_label:<22} {tt_label:<22} {'YES':>5}  {pcc:.4f}" if match
              else f"{i+1:>3}  {pt_label:<22} {tt_label:<22} {'NO':>5}  {pcc:.4f}")

        results.append({
            "idx": i,
            "pt_top1": COCO_CLASSES[pt_top],
            "tt_top1": COCO_CLASSES[ttnn_top],
            "match": match,
            "pcc": pcc,
            "pt_conf": pt_probs[pt_top].item(),
            "tt_conf": ttnn_probs[ttnn_top].item(),
        })

    ttnn.close_device(device)

    # Summary
    avg_pcc = sum(r["pcc"] for r in results) / len(results)
    match_rate = pt_matches_ttnn / total * 100

    print(f"\n{'='*70}")
    print(f"  SUMMARY — Stage {args.stage}, {total} images, {len(COCO_CLASSES)} classes")
    print(f"{'='*70}")
    print(f"  TTNN matches PyTorch: {pt_matches_ttnn}/{total} ({match_rate:.0f}%)")
    print(f"  Avg logits PCC:       {avg_pcc:.6f}")
    mismatches = [r for r in results if not r["match"]]
    if mismatches:
        print(f"\n  Mismatches:")
        for r in mismatches:
            print(f"    Image {r['idx']+1}: PT={r['pt_top1']} ({r['pt_conf']:.0%}) vs TT={r['tt_top1']} ({r['tt_conf']:.0%})")
    print(f"{'='*70}")

    # Save markdown
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    with open(args.output, "w") as f:
        f.write(f"# COCO Zero-Shot Validation — Stage {args.stage}\n\n")
        f.write(f"**Images:** {total} | **Classes:** {len(COCO_CLASSES)} | ")
        f.write(f"**Match rate:** {pt_matches_ttnn}/{total} ({match_rate:.0f}%) | ")
        f.write(f"**Avg PCC:** {avg_pcc:.6f}\n\n")
        f.write(f"| # | PyTorch Top-1 | TTNN Top-1 | Match | PCC |\n")
        f.write(f"|---|---------------|------------|-------|-----|\n")
        for r in results:
            m = "YES" if r["match"] else "NO"
            f.write(f"| {r['idx']+1} | {r['pt_top1']} ({r['pt_conf']:.0%}) | {r['tt_top1']} ({r['tt_conf']:.0%}) | {m} | {r['pcc']:.4f} |\n")
        f.write(f"\n")
        if mismatches:
            f.write(f"**Mismatches:** {len(mismatches)}\n")
            for r in mismatches:
                f.write(f"- Image {r['idx']+1}: PT={r['pt_top1']} vs TT={r['tt_top1']}\n")

    print(f"  Saved to {args.output}")


if __name__ == "__main__":
    main()
