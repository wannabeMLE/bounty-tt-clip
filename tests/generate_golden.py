#!/usr/bin/env python3
"""
generate_golden.py -- Generate golden reference tensors for CLIP-ViT TTNN validation.

Runs the full PyTorch CLIP model (openai/clip-vit-base-patch32) on CPU and saves
ALL intermediate tensors needed to validate the TTNN implementation phase by phase.

Usage:
    python generate_golden.py                        # default: 10 COCO images
    python generate_golden.py --num_coco_images 50   # more COCO images
    python generate_golden.py --output golden.pt     # custom output path
    python generate_golden.py --skip_coco            # skip Phase 6 (faster)
"""

import argparse
import datetime
import os
import time

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from urllib.request import urlopen

MODEL_NAME = "openai/clip-vit-base-patch32"
TEST_IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"
TEST_TEXTS = ["a photo of a cat", "a photo of a dog", "a photo of a car"]

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


def compute_pcc(a, b):
    """Pearson Correlation Coefficient between two tensors."""
    a = a.detach().float().flatten()
    b = b.detach().float().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def load_model_and_inputs():
    """Load CLIP model, processor, test image, and prepare inputs."""
    print("Loading model...")
    model = CLIPModel.from_pretrained(MODEL_NAME, attn_implementation="eager")
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model.eval()

    print(f"Downloading test image from {TEST_IMAGE_URL}...")
    image = Image.open(urlopen(TEST_IMAGE_URL)).convert("RGB")

    inputs = processor(
        text=TEST_TEXTS, images=image, return_tensors="pt", padding=True, truncation=True
    )
    print(f"  pixel_values: {inputs['pixel_values'].shape}")
    print(f"  input_ids:    {inputs['input_ids'].shape}")
    print(f"  attention_mask: {inputs['attention_mask'].shape}")

    return model, processor, image, inputs


def capture_phase1(model, pixel_values):
    """Phase 1: Vision patch embedding sub-steps."""
    print("\n--- Phase 1: Vision Patch Embedding ---")
    vision = model.vision_model
    result = {}

    with torch.no_grad():
        # Conv2d patch embedding
        patch_conv_out = vision.embeddings.patch_embedding(pixel_values)
        result["patch_conv_output"] = patch_conv_out.clone()
        print(f"  patch_conv_output: {patch_conv_out.shape}")

        # Flatten and transpose: [B, 768, 7, 7] -> [B, 49, 768]
        patches = patch_conv_out.flatten(2).transpose(1, 2)
        result["patches_flat"] = patches.clone()
        print(f"  patches_flat: {patches.shape}")

        # Prepend CLS token
        batch_size = pixel_values.shape[0]
        cls_token = vision.embeddings.class_embedding.unsqueeze(0).unsqueeze(0)
        cls_token = cls_token.expand(batch_size, -1, -1)
        cls_patches = torch.cat([cls_token, patches], dim=1)
        result["cls_plus_patches"] = cls_patches.clone()
        print(f"  cls_plus_patches: {cls_patches.shape}")

        # Add position embeddings
        pos_embed = vision.embeddings.position_embedding.weight.unsqueeze(0)
        with_pos = cls_patches + pos_embed
        result["with_position_embed"] = with_pos.clone()
        print(f"  with_position_embed: {with_pos.shape}")

        # Pre-LayerNorm (note: HF has typo "pre_layrnorm")
        ln_pre = vision.pre_layrnorm(with_pos)
        result["ln_pre_output"] = ln_pre.clone()
        print(f"  ln_pre_output: {ln_pre.shape}")

        # Cross-check against HF's own embeddings call
        hf_embed = vision.embeddings(pixel_values)
        hf_embed_pcc = compute_pcc(with_pos, hf_embed)
        print(f"  Self-check: manual embed vs HF embed PCC = {hf_embed_pcc:.6f}")

    return result


def capture_block_internals(layer, hidden_states, num_heads, head_dim, causal_mask=None):
    """Decompose a single encoder block into every sub-operation.

    Works for both vision and text encoder blocks.
    """
    result = {}
    result["input"] = hidden_states.clone()

    with torch.no_grad():
        # LayerNorm 1
        ln1 = layer.layer_norm1(hidden_states)
        result["layer_norm1"] = ln1.clone()

        # Q, K, V projections
        Q = layer.self_attn.q_proj(ln1)
        K = layer.self_attn.k_proj(ln1)
        V = layer.self_attn.v_proj(ln1)
        result["Q_proj"] = Q.clone()
        result["K_proj"] = K.clone()
        result["V_proj"] = V.clone()

        # Fused QKV (as TTNN uses it)
        QKV_fused = torch.cat([Q, K, V], dim=-1)
        result["QKV_fused"] = QKV_fused.clone()

        # Reshape to multi-head
        B, S, _ = Q.shape
        Q_heads = Q.view(B, S, num_heads, head_dim).transpose(1, 2)
        K_heads = K.view(B, S, num_heads, head_dim).transpose(1, 2)
        V_heads = V.view(B, S, num_heads, head_dim).transpose(1, 2)
        result["Q_heads"] = Q_heads.clone()
        result["K_heads"] = K_heads.clone()
        result["V_heads"] = V_heads.clone()

        # Attention scores: Q @ K^T * scale
        scale = head_dim ** -0.5
        attn_scores = torch.matmul(Q_heads, K_heads.transpose(-1, -2)) * scale
        result["attn_scores_raw"] = attn_scores.clone()

        # Apply causal mask if provided (text encoder)
        if causal_mask is not None:
            attn_scores = attn_scores + causal_mask
        result["attn_scores_masked"] = attn_scores.clone()

        # Softmax
        attn_probs = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(Q.dtype)
        result["attn_probs"] = attn_probs.clone()

        # Context: attn_probs @ V
        context = torch.matmul(attn_probs, V_heads)
        result["context_heads"] = context.clone()

        # Concatenate heads
        context_concat = context.transpose(1, 2).contiguous().reshape(B, S, -1)
        result["context_concat"] = context_concat.clone()

        # Output projection
        attn_output = layer.self_attn.out_proj(context_concat)
        result["out_proj"] = attn_output.clone()

        # Residual 1
        residual1 = hidden_states + attn_output
        result["residual1"] = residual1.clone()

        # LayerNorm 2
        ln2 = layer.layer_norm2(residual1)
        result["layer_norm2"] = ln2.clone()

        # MLP: fc1
        fc1_out = layer.mlp.fc1(ln2)
        result["mlp_fc1"] = fc1_out.clone()

        # QuickGELU activation
        gelu_out = layer.mlp.activation_fn(fc1_out)
        result["mlp_quickgelu"] = gelu_out.clone()

        # MLP: fc2
        fc2_out = layer.mlp.fc2(gelu_out)
        result["mlp_fc2"] = fc2_out.clone()

        # Residual 2
        residual2 = residual1 + fc2_out
        result["residual2"] = residual2.clone()
        result["block_output"] = residual2.clone()

        # Cross-check: run the block normally and compare
        hf_output = layer(hidden_states, attention_mask=causal_mask)
        pcc = compute_pcc(residual2, hf_output)
        result["_self_check_pcc"] = pcc

    return result


def capture_phase2(model, ln_pre_output):
    """Phase 2: Single vision transformer block (block 0) internals."""
    print("\n--- Phase 2: Vision Block 0 Internals ---")
    layer = model.vision_model.encoder.layers[0]
    num_heads = model.config.vision_config.num_attention_heads
    head_dim = model.config.vision_config.hidden_size // num_heads

    result = capture_block_internals(layer, ln_pre_output, num_heads, head_dim)

    print(f"  Keys captured: {len([k for k in result if not k.startswith('_')])}")
    print(f"  Self-check PCC: {result['_self_check_pcc']:.6f}")
    for key in ["layer_norm1", "Q_proj", "attn_probs", "out_proj", "mlp_quickgelu", "block_output"]:
        print(f"  {key}: {result[key].shape}")

    return result


def capture_phase3(model, pixel_values):
    """Phase 3: Full vision encoder (all 12 layers + post-processing)."""
    print("\n--- Phase 3: Full Vision Encoder ---")
    vision = model.vision_model
    result = {}

    with torch.no_grad():
        # Run embeddings + pre-layernorm
        embeddings = vision.embeddings(pixel_values)
        hidden = vision.pre_layrnorm(embeddings)

        # All 12 layers (vision: no attention mask)
        for i, layer in enumerate(vision.encoder.layers):
            hidden = layer(hidden, attention_mask=None)
            result[f"layer_{i}"] = hidden.clone()

        # Post-layernorm on CLS token (position 0)
        cls_token = hidden[:, 0, :]
        post_ln = vision.post_layernorm(cls_token)
        result["post_layernorm_cls"] = post_ln.clone()
        print(f"  post_layernorm_cls: {post_ln.shape}")

        # Visual projection
        projected = model.visual_projection(post_ln)
        result["visual_projection"] = projected.clone()
        print(f"  visual_projection: {projected.shape}")

    print(f"  Captured {len(result)} tensors (12 layers + post_ln + projection)")
    return result


def capture_phase4(model, input_ids, attention_mask):
    """Phase 4: Full text encoder with embeddings, layers, and block 0 internals."""
    print("\n--- Phase 4: Text Encoder ---")
    text = model.text_model
    result = {}

    with torch.no_grad():
        # Token embeddings
        token_embed = text.embeddings.token_embedding(input_ids)
        result["token_embeddings"] = token_embed.clone()
        print(f"  token_embeddings: {token_embed.shape}")

        # Position embeddings
        seq_len = input_ids.shape[1]
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(input_ids.shape[0], -1)
        pos_embed = text.embeddings.position_embedding(position_ids)
        result["position_embeddings"] = pos_embed.clone()
        print(f"  position_embeddings: {pos_embed.shape}")

        # Combined embeddings
        combined = token_embed + pos_embed
        result["combined_embeddings"] = combined.clone()
        print(f"  combined_embeddings: {combined.shape}")

        # Build causal mask
        causal_mask = torch.full((seq_len, seq_len), float("-inf"))
        causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, S, S]
        result["causal_mask"] = causal_mask.clone()
        print(f"  causal_mask: {causal_mask.shape}")

        # Cross-check embeddings against HF
        hf_embed = text.embeddings(input_ids=input_ids, position_ids=None)
        embed_pcc = compute_pcc(combined, hf_embed)
        print(f"  Self-check: manual embed vs HF embed PCC = {embed_pcc:.6f}")

        # Block 0 internals
        num_heads = model.config.text_config.num_attention_heads
        head_dim = model.config.text_config.hidden_size // num_heads
        block0_internals = capture_block_internals(
            text.encoder.layers[0], combined, num_heads, head_dim, causal_mask=causal_mask
        )
        result["block0_internals"] = block0_internals
        print(f"  block0 self-check PCC: {block0_internals['_self_check_pcc']:.6f}")

        # Run all 12 layers (text: with causal mask)
        hidden = combined
        for i, layer in enumerate(text.encoder.layers):
            hidden = layer(hidden, attention_mask=causal_mask)
            result[f"layer_{i}"] = hidden.clone()

        # Final layer norm
        hidden_ln = text.final_layer_norm(hidden)
        result["final_layer_norm"] = hidden_ln.clone()
        print(f"  final_layer_norm: {hidden_ln.shape}")

        # EOT token extraction (argmax of input_ids = EOS position)
        eos_indices = input_ids.argmax(dim=-1)
        result["eos_indices"] = eos_indices.clone()
        print(f"  eos_indices: {eos_indices}")

        pooled = hidden_ln[torch.arange(hidden_ln.shape[0]), eos_indices]
        result["pooled"] = pooled.clone()
        print(f"  pooled: {pooled.shape}")

        # Text projection
        projected = model.text_projection(pooled)
        result["text_projection"] = projected.clone()
        print(f"  text_projection: {projected.shape}")

    print(f"  Captured {len(result)} top-level keys")
    return result


def capture_phase5(model, processor, image, texts):
    """Phase 5: Full CLIP pipeline (similarity computation)."""
    print("\n--- Phase 5: Full CLIP Pipeline ---")
    result = {}

    inputs = processor(text=texts, images=image, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        outputs = model(**inputs)

        # Raw embeddings (before L2 norm)
        result["image_embeds_raw"] = outputs.image_embeds.clone()
        result["text_embeds_raw"] = outputs.text_embeds.clone()
        print(f"  image_embeds_raw: {outputs.image_embeds.shape}")
        print(f"  text_embeds_raw: {outputs.text_embeds.shape}")

        # L2 normalized
        img_norm = outputs.image_embeds / outputs.image_embeds.norm(p=2, dim=-1, keepdim=True)
        txt_norm = outputs.text_embeds / outputs.text_embeds.norm(p=2, dim=-1, keepdim=True)
        result["image_embeds_norm"] = img_norm.clone()
        result["text_embeds_norm"] = txt_norm.clone()

        # Logit scale
        result["logit_scale"] = model.logit_scale.data.clone()
        result["logit_scale_exp"] = model.logit_scale.exp().clone()
        print(f"  logit_scale: {model.logit_scale.item():.4f}")
        print(f"  logit_scale_exp: {model.logit_scale.exp().item():.4f}")

        # Similarity
        result["logits_per_image"] = outputs.logits_per_image.clone()
        result["logits_per_text"] = outputs.logits_per_text.clone()
        print(f"  logits_per_image: {outputs.logits_per_image}")

        # Softmax probabilities
        probs = outputs.logits_per_image.softmax(dim=-1)
        result["probs"] = probs.clone()
        print(f"  probs: {probs}")

        # Prediction
        predicted_idx = probs.argmax(dim=-1).item()
        result["predicted_idx"] = predicted_idx
        result["predicted_text"] = texts[predicted_idx]
        print(f"  Predicted: '{texts[predicted_idx]}' (idx={predicted_idx})")

    return result


def capture_phase6(model, processor, num_images):
    """Phase 6: COCO zero-shot classification."""
    print(f"\n--- Phase 6: COCO Zero-Shot ({num_images} images) ---")
    from datasets import load_dataset

    result = {
        "coco_classes": COCO_CLASSES,
        "num_images": num_images,
        "per_image": [],
    }

    # Pre-encode all 80 COCO class prompts
    print("  Encoding 80 COCO class prompts...")
    prompts = [f"a photo of a {c}" for c in COCO_CLASSES]
    text_inputs = processor(text=prompts, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        text_outputs = model.get_text_features(**text_inputs)
        if not isinstance(text_outputs, torch.Tensor):
            text_outputs = text_outputs.pooler_output if hasattr(text_outputs, "pooler_output") else text_outputs.last_hidden_state
        text_features = text_outputs / text_outputs.norm(p=2, dim=-1, keepdim=True)

    result["text_features_norm"] = text_features.clone()
    print(f"  text_features_norm: {text_features.shape}")

    # Load COCO val2017
    print(f"  Loading {num_images} COCO val2017 images...")
    ds = load_dataset("detection-datasets/coco", split="val", streaming=True)

    logit_scale = model.logit_scale.exp()
    images_processed = 0

    for i, sample in enumerate(ds):
        if i >= num_images:
            break

        img = sample["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")

        img_inputs = processor(images=img, return_tensors="pt")

        with torch.no_grad():
            img_outputs = model.get_image_features(**img_inputs)
            if not isinstance(img_outputs, torch.Tensor):
                img_outputs = img_outputs.pooler_output if hasattr(img_outputs, "pooler_output") else img_outputs.last_hidden_state
            img_features = img_outputs / img_outputs.norm(p=2, dim=-1, keepdim=True)

        logits = (img_features @ text_features.T) * logit_scale
        probs = logits.softmax(dim=-1)[0]
        top5 = probs.argsort(descending=True)[:5]

        entry = {
            "image_features_norm": img_features.clone(),
            "logits": logits.clone(),
            "probs": probs.clone(),
            "top5_indices": top5.clone(),
            "top5_labels": [COCO_CLASSES[j] for j in top5],
            "top5_probs": [probs[j].item() for j in top5],
            "top1_label": COCO_CLASSES[top5[0]],
            "top1_prob": probs[top5[0]].item(),
        }

        # Include ground truth if available
        if "objects" in sample and "category" in sample["objects"]:
            entry["ground_truth_ids"] = sample["objects"]["category"]

        result["per_image"].append(entry)
        images_processed += 1

        if (i + 1) % 5 == 0:
            print(f"    [{i+1}/{num_images}] top-1: {entry['top1_label']} ({entry['top1_prob']:.3f})")

    print(f"  Processed {images_processed} images")
    return result


def main():
    parser = argparse.ArgumentParser(description="Generate golden reference tensors for CLIP-ViT TTNN validation")
    parser.add_argument("--output", type=str, default="golden_reference.pt", help="Output file path")
    parser.add_argument("--num_coco_images", type=int, default=10, help="Number of COCO images for Phase 6")
    parser.add_argument("--skip_coco", action="store_true", help="Skip Phase 6 (COCO zero-shot)")
    args = parser.parse_args()

    print("=" * 60)
    print("  CLIP-ViT Golden Reference Generator")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Output: {args.output}")
    print("=" * 60)

    t_start = time.perf_counter()

    # Load model and inputs
    model, processor, image, inputs = load_model_and_inputs()

    golden = {
        "metadata": {
            "model_name": MODEL_NAME,
            "timestamp": datetime.datetime.now().isoformat(),
            "torch_version": torch.__version__,
            "transformers_version": __import__("transformers").__version__,
            "image_url": TEST_IMAGE_URL,
            "texts": TEST_TEXTS,
        },
        "inputs": {
            "pixel_values": inputs["pixel_values"].clone(),
            "input_ids": inputs["input_ids"].clone(),
            "attention_mask": inputs["attention_mask"].clone(),
        },
    }

    # Phase 1
    golden["phase1_vision_embed"] = capture_phase1(model, inputs["pixel_values"])

    # Phase 2
    golden["phase2_block0_vision"] = capture_phase2(
        model, golden["phase1_vision_embed"]["ln_pre_output"]
    )

    # Phase 3
    golden["phase3_vision_encoder"] = capture_phase3(model, inputs["pixel_values"])

    # Phase 4
    golden["phase4_text_encoder"] = capture_phase4(
        model, inputs["input_ids"], inputs["attention_mask"]
    )

    # Phase 5
    golden["phase5_full_pipeline"] = capture_phase5(model, processor, image, TEST_TEXTS)

    # Phase 6
    if not args.skip_coco:
        golden["phase6_coco"] = capture_phase6(model, processor, args.num_coco_images)
    else:
        print("\n--- Phase 6: SKIPPED (--skip_coco) ---")

    # Save
    print(f"\nSaving to {args.output}...")
    torch.save(golden, args.output)
    file_size = os.path.getsize(args.output)
    elapsed = time.perf_counter() - t_start

    print(f"\n{'=' * 60}")
    print(f"  Done in {elapsed:.1f}s")
    print(f"  File: {args.output} ({file_size / 1e6:.1f} MB)")
    print(f"  Phases: 1-5{' + 6 (COCO)' if not args.skip_coco else ''}")
    print(f"{'=' * 60}")

    # Quick summary of what's inside
    print("\nSaved tensor summary:")
    def count_tensors(d, prefix=""):
        count = 0
        for k, v in d.items():
            if isinstance(v, torch.Tensor):
                count += 1
            elif isinstance(v, dict):
                count += count_tensors(v, f"{prefix}{k}.")
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, dict):
                        count += count_tensors(item)
        return count

    total = count_tensors(golden)
    print(f"  Total tensors saved: {total}")


if __name__ == "__main__":
    main()
