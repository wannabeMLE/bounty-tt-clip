#!/usr/bin/env python3
"""
validate_multi_image.py -- Test TTNN CLIP on multiple images vs PyTorch reference.

Downloads 5 COCO val2017 images, runs both PyTorch and TTNN, compares predictions.
"""

import os
import sys

# Ensure project root is on path when running from tests/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F
import ttnn
from PIL import Image
from urllib.request import urlopen
from transformers import CLIPModel, CLIPProcessor

from clip_vit_ttnn.tt.weight_loader import CLIPTTNNConfig, load_all_weights
from clip_vit_ttnn.tt.clip_model import run_vision_encoder, run_text_encoder, compute_similarity

MODEL_NAME = "openai/clip-vit-base-patch32"

# 5 diverse COCO val2017 images (verified accessible)
TEST_IMAGES = [
    ("cats",     "http://images.cocodataset.org/val2017/000000039769.jpg"),
    ("people",   "http://images.cocodataset.org/val2017/000000350002.jpg"),
    ("food",     "http://images.cocodataset.org/val2017/000000181796.jpg"),
    ("person2",  "http://images.cocodataset.org/val2017/000000186980.jpg"),
    ("umbrella", "http://images.cocodataset.org/val2017/000000087038.jpg"),
]

CANDIDATE_TEXTS = [
    "a photo of a cat",
    "a photo of a dog",
    "a photo of a giraffe",
    "a photo of a horse",
    "a photo of a bus",
    "a photo of a pizza",
    "a photo of a car",
    "a photo of a person",
]


def compute_pcc(a, b):
    a = a.detach().float().flatten()
    b = b.detach().float().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def main():
    print("=" * 65)
    print("  CLIP-ViT Multi-Image Validation: PyTorch vs TTNN")
    print("=" * 65)

    # Load HF model
    print("\nLoading HuggingFace model...")
    hf_model = CLIPModel.from_pretrained(MODEL_NAME, attn_implementation="eager")
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    hf_model.eval()

    # Init TTNN
    print("Initializing TTNN device...")
    device = ttnn.open_device(device_id=0)
    config = CLIPTTNNConfig(stage=1)
    params = load_all_weights(hf_model, device, config)
    params["logit_scale"] = hf_model.logit_scale.data.clone()

    # Pre-encode texts with TTNN (reuse across images)
    print(f"\nEncoding {len(CANDIDATE_TEXTS)} candidate texts...")
    text_inputs = processor(text=CANDIDATE_TEXTS, return_tensors="pt", padding=True, truncation=True)

    # TTNN: encode each text separately (batch=1)
    ttnn_text_embeds = []
    for t in range(len(CANDIDATE_TEXTS)):
        tt_embed, text_torch = run_text_encoder(
            text_inputs["input_ids"][t:t+1],
            text_inputs["attention_mask"][t:t+1],
            params["text"], config, device,
        )
        ttnn.deallocate(tt_embed)
        ttnn_text_embeds.append(text_torch)
    ttnn_text_all = torch.cat(ttnn_text_embeds, dim=0)

    # PyTorch: encode all texts at once
    with torch.no_grad():
        pt_text_out = hf_model.get_text_features(**text_inputs)
    pt_text_all = pt_text_out

    print(f"  TTNN text embeds: {ttnn_text_all.shape}")
    print(f"  PT text embeds:   {pt_text_all.shape}")

    # Test each image
    print(f"\n{'='*65}")
    print(f"  {'Image':<15} {'PyTorch Top-1':<25} {'TTNN Top-1':<25} {'Match'}")
    print(f"{'='*65}")

    matches = 0
    all_pcc = []

    for img_id, img_url in TEST_IMAGES:
        try:
            image = Image.open(urlopen(img_url)).convert("RGB")
        except Exception as e:
            print(f"  {img_id:<15} DOWNLOAD FAILED: {e}")
            continue

        img_inputs = processor(images=image, return_tensors="pt")
        pixel_values = img_inputs["pixel_values"]

        # PyTorch prediction
        with torch.no_grad():
            pt_img_out = hf_model.get_image_features(pixel_values=pixel_values)
            pt_img_norm = pt_img_out / pt_img_out.norm(p=2, dim=-1, keepdim=True)
            pt_text_norm = pt_text_all / pt_text_all.norm(p=2, dim=-1, keepdim=True)
            pt_logits = (pt_img_norm @ pt_text_norm.T) * hf_model.logit_scale.exp()
            pt_probs = pt_logits.softmax(dim=-1)[0]
            pt_top = pt_probs.argmax().item()

        # TTNN prediction
        tt_vis_embed, vis_torch = run_vision_encoder(
            pixel_values, params["vision"], config, device
        )
        ttnn.deallocate(tt_vis_embed)

        ttnn_logits = compute_similarity(vis_torch, ttnn_text_all, params["logit_scale"])
        ttnn_probs = ttnn_logits.softmax(dim=-1)[0]
        ttnn_top = ttnn_probs.argmax().item()

        pcc = compute_pcc(pt_logits, ttnn_logits)
        all_pcc.append(pcc)

        match = "YES" if pt_top == ttnn_top else "NO"
        if pt_top == ttnn_top:
            matches += 1

        pt_label = f"{CANDIDATE_TEXTS[pt_top]} ({pt_probs[pt_top]:.1%})"
        tt_label = f"{CANDIDATE_TEXTS[ttnn_top]} ({ttnn_probs[ttnn_top]:.1%})"

        print(f"  {img_id:<15} {pt_label:<25} {tt_label:<25} {match}")

    # Summary
    print(f"\n{'='*65}")
    print(f"  Results: {matches}/{len(TEST_IMAGES)} predictions match")
    print(f"  Avg logits PCC: {sum(all_pcc)/len(all_pcc):.6f}")
    print(f"{'='*65}")

    ttnn.close_device(device)
    print("Done.")


if __name__ == "__main__":
    main()
