# SPDX-License-Identifier: Apache-2.0
#
# CLIP-ViT TTNN Demo
# End-to-end demo: load image + text, encode on TT hardware, compute similarity.
#
# Usage:
#   python demo/demo_clip.py
#   python demo/demo_clip.py --stage 2
#   python demo/demo_clip.py --stage 3 --image path/to/image.jpg

import argparse
import time
import sys
import os

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ttnn
from tt.weight_loader import CLIPTTNNConfig, load_all_weights
from tt.clip_model import CLIPModelTTNN, run_vision_encoder, run_text_encoder, compute_similarity
from reference.torch_clip import compute_pcc


MODEL_NAME = "openai/clip-vit-base-patch32"


def load_sample_image(image_path=None):
    """Load image from path or download COCO sample."""
    if image_path and os.path.exists(image_path):
        return Image.open(image_path).convert("RGB")

    print("Downloading sample COCO image...")
    from urllib.request import urlopen
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    return Image.open(urlopen(url))


def run_demo(stage: int = 1, image_path: str = None):
    """Run full CLIP-ViT demo on Tenstorrent hardware."""

    print("=" * 60)
    print(f"  CLIP-ViT TTNN Demo — Stage {stage}")
    print("=" * 60)

    # --- Load HuggingFace model ---
    print("\n[1/5] Loading HuggingFace CLIP model...")
    hf_model = CLIPModel.from_pretrained(MODEL_NAME)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    hf_model.eval()

    # --- Open device ---
    print("[2/5] Opening Tenstorrent device...")
    dispatch_config = ttnn.DispatchCoreConfig(type=ttnn.DispatchCoreType.WORKER)
    device = ttnn.open_device(device_id=0, dispatch_core_config=dispatch_config)

    try:
        # --- Create config and load weights ---
        config = CLIPTTNNConfig.from_huggingface(hf_model.config)
        config.stage = stage

        print(f"[3/5] Loading weights (stage {stage})...")
        clip_ttnn = CLIPModelTTNN(hf_model, device, config)

        # --- Prepare inputs ---
        print("[4/5] Preparing inputs...")
        image = load_sample_image(image_path)
        texts = [
            "a photo of a cat",
            "a photo of a dog",
            "two cats sitting on a couch",
            "a sunny beach with palm trees",
        ]

        image_inputs = processor(images=image, return_tensors="pt")
        pixel_values = image_inputs["pixel_values"]

        # --- Run inference ---
        print("[5/5] Running inference on TT hardware...")
        print()

        # Vision encoder
        t0 = time.perf_counter()
        vision_embed = clip_ttnn.encode_image(pixel_values)
        t_vision = time.perf_counter() - t0

        # Text encoder (process each text separately)
        text_embeds = []
        t_text_total = 0
        for text in texts:
            text_inputs = processor(
                text=text, return_tensors="pt", padding="max_length",
                max_length=config.text_max_position_embeddings, truncation=True,
            )
            t1 = time.perf_counter()
            text_embed = clip_ttnn.encode_text(
                text_inputs["input_ids"], text_inputs["attention_mask"]
            )
            t_text_total += time.perf_counter() - t1
            text_embeds.append(text_embed)

        text_embeds_cat = torch.cat(text_embeds, dim=0)

        # Similarity
        logits = compute_similarity(vision_embed, text_embeds_cat, clip_ttnn.params["logit_scale"])
        probs = logits.softmax(dim=-1)

        # --- PyTorch reference for comparison ---
        with torch.no_grad():
            ref_inputs = processor(
                text=texts, images=image, return_tensors="pt",
                padding=True, truncation=True,
            )
            ref_outputs = hf_model(**ref_inputs)
            ref_logits = ref_outputs.logits_per_image
            ref_probs = ref_logits.softmax(dim=-1)

        # --- Results ---
        print("=" * 60)
        print("  Results")
        print("=" * 60)
        print()
        print("Image-Text Similarity Scores:")
        print("-" * 50)
        for i, text in enumerate(texts):
            ttnn_p = probs[0, i].item()
            ref_p = ref_probs[0, i].item()
            match = "OK" if abs(ttnn_p - ref_p) < 0.05 else "DIFF"
            print(f"  [{match}] '{text}'")
            print(f"        TTNN: {ttnn_p:.4f}  |  PyTorch: {ref_p:.4f}")
        print()

        # PCC
        pcc = compute_pcc(ref_logits, logits)
        thresholds = {1: 0.99, 2: 0.98, 3: 0.97}
        threshold = thresholds[stage]
        status = "PASS" if pcc > threshold else "FAIL"
        print(f"PCC vs PyTorch: {pcc:.6f} (threshold: {threshold}) [{status}]")
        print()

        # Ranking comparison
        ref_rank = ref_probs.argsort(dim=-1, descending=True)[0]
        ttnn_rank = probs.argsort(dim=-1, descending=True)[0]
        rank_match = torch.equal(ref_rank, ttnn_rank)
        print(f"Ranking match: {'YES' if rank_match else 'NO'}")
        print(f"  PyTorch ranking: {[texts[i] for i in ref_rank.tolist()]}")
        print(f"  TTNN ranking:    {[texts[i] for i in ttnn_rank.tolist()]}")
        print()

        # Timing
        print("Performance:")
        print(f"  Vision encoder: {t_vision*1000:.1f} ms")
        print(f"  Text encoder:   {t_text_total*1000:.1f} ms ({len(texts)} texts)")
        print(f"  Total:          {(t_vision + t_text_total)*1000:.1f} ms")
        print()

        # Warmup + benchmark
        print("Benchmarking (5 runs after warmup)...")
        # Warmup
        clip_ttnn.encode_image(pixel_values)

        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            clip_ttnn.encode_image(pixel_values)
            times.append(time.perf_counter() - t0)

        avg_ms = sum(times) / len(times) * 1000
        min_ms = min(times) * 1000
        print(f"  Vision encoder: avg={avg_ms:.1f}ms, best={min_ms:.1f}ms")

    finally:
        ttnn.close_device(device)

    print("\nDone!")


def main():
    parser = argparse.ArgumentParser(description="CLIP-ViT TTNN Demo")
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2, 3],
                       help="Optimization stage (1=basic, 2=optimized, 3=deep)")
    parser.add_argument("--image", type=str, default=None,
                       help="Path to input image (default: COCO sample)")
    args = parser.parse_args()

    run_demo(stage=args.stage, image_path=args.image)


if __name__ == "__main__":
    main()
