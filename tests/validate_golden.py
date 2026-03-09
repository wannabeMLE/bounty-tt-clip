#!/usr/bin/env python3
"""
validate_golden.py -- Validate TTNN CLIP implementation against golden_reference.pt

Runs the TTNN implementation on the same inputs saved by generate_golden.py and
compares outputs phase by phase, layer by layer, to pinpoint where errors occur.

Usage:
    python validate_golden.py
    python validate_golden.py --golden path/to/golden_reference.pt
    python validate_golden.py --stage 1
"""

import argparse
import os
import sys
import traceback

# Ensure project root is on path when running from tests/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import ttnn

from clip_vit_ttnn.tt.weight_loader import CLIPTTNNConfig, load_all_weights
from clip_vit_ttnn.tt.clip_model import (
    vision_patch_embeddings,
    run_vision_encoder,
    run_text_encoder,
    compute_similarity,
)


def compute_pcc(a, b):
    """Pearson Correlation Coefficient between two tensors."""
    a = a.detach().float().flatten()
    b = b.detach().float().flatten()
    if a.numel() == 0 or b.numel() == 0:
        return float("nan")
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def pcc_status(pcc, threshold=0.99):
    """Return a PASS/FAIL string based on PCC threshold."""
    if pcc != pcc:  # NaN
        return "NaN"
    return "PASS" if pcc >= threshold else "FAIL"


def print_header(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def stage_threshold(config, default=0.99):
    """Return PCC threshold for the given stage (LoFi math is less precise)."""
    return 0.98 if config.stage >= 2 else default


def print_result(name, pcc, threshold=0.99):
    status = pcc_status(pcc, threshold)
    marker = "OK" if status == "PASS" else "**"
    print(f"  [{marker}] {name}: PCC = {pcc:.6f}  ({status} @ {threshold})")


def validate_phase1(golden, params, config, device):
    """Phase 1: Vision patch embeddings."""
    print_header("Phase 1: Vision Patch Embeddings")

    pixel_values = golden["inputs"]["pixel_values"]
    golden_embed = golden["phase1_vision_embed"]["with_position_embed"]

    try:
        tt_embed = vision_patch_embeddings(pixel_values, params["vision"], config, device)
        tt_embed_torch = ttnn.to_torch(tt_embed)
        ttnn.deallocate(tt_embed)

        # Golden is [1, 50, 768], TTNN may be padded to [1, 64, 768]
        seq_len = golden_embed.shape[1]
        tt_trimmed = tt_embed_torch[:, :seq_len, :]

        pcc = compute_pcc(golden_embed, tt_trimmed)
        print_result("vision_patch_embeddings", pcc, stage_threshold(config))
        return pcc
    except Exception as e:
        print(f"  [!!] ERROR: {e}")
        traceback.print_exc()
        return 0.0


def validate_phase2(golden, params, config, device):
    """Phase 2: Vision block 0 internals — sub-operation accuracy."""
    print_header("Phase 2: Vision Block 0 Internals")

    golden_phase2 = golden["phase2_block0_vision"]
    golden_phase1 = golden["phase1_vision_embed"]

    try:
        from clip_vit_ttnn.tt.clip_model import _softmax

        threshold = stage_threshold(config)
        memory_config = config.get_memory_config()
        compute_config = config.get_compute_kernel_config()
        layer_params = params["vision"]["layers"][0]
        num_heads = config.vision_num_heads
        seq_len = golden_phase2["input"].shape[1]  # 50

        # Start from golden ln_pre output (the input to block 0)
        golden_input = golden_phase1["ln_pre_output"]  # [1, 50, 768]

        tt_hidden = ttnn.from_torch(
            golden_input,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=memory_config,
        )

        # --- Layer Norm 1 ---
        residual = tt_hidden
        tt_hidden = ttnn.layer_norm(
            tt_hidden,
            weight=layer_params["layer_norm1"]["weight"],
            bias=layer_params["layer_norm1"]["bias"],
            epsilon=config.layer_norm_eps,
            memory_config=memory_config,
        )
        tt_ln1 = ttnn.to_torch(tt_hidden)
        pcc_ln1 = compute_pcc(golden_phase2["layer_norm1"], tt_ln1[:, :seq_len, :])
        print_result("layer_norm1", pcc_ln1, threshold)

        # --- Fused QKV ---
        qkv = ttnn.linear(
            tt_hidden,
            layer_params["self_attn"]["qkv_weight"],
            bias=layer_params["self_attn"]["qkv_bias"],
            memory_config=memory_config,
            compute_kernel_config=compute_config,
        )
        tt_qkv = ttnn.to_torch(qkv)
        pcc_qkv = compute_pcc(golden_phase2["QKV_fused"], tt_qkv[:, :seq_len, :])
        print_result("QKV_fused", pcc_qkv, threshold)

        ttnn.deallocate(tt_hidden)

        # --- Split heads ---
        q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(
            qkv, num_heads=num_heads
        )
        ttnn.deallocate(qkv)

        tt_q = ttnn.to_torch(q)
        tt_k = ttnn.to_torch(k)
        tt_v = ttnn.to_torch(v)
        print(f"    Q shape: golden={golden_phase2['Q_heads'].shape}, ttnn={tt_q.shape}")
        print(f"    K shape: golden={golden_phase2['K_heads'].shape}, ttnn={tt_k.shape}")
        # TTNN split_query_key_value_and_split_heads returns K transposed:
        #   Q: [batch, heads, seq_padded, head_dim]
        #   K: [batch, heads, head_dim, seq_padded]  (transposed for Q@K^T)
        #   V: [batch, heads, seq_padded, head_dim]
        # Golden Q/K/V are all [batch, heads, seq, head_dim] (non-transposed)
        g_q = golden_phase2["Q_heads"]  # [1, heads, seq, dim]
        g_k = golden_phase2["K_heads"]
        g_v = golden_phase2["V_heads"]
        nh, sl, hd = g_q.shape[1], g_q.shape[2], g_q.shape[3]
        pcc_q = compute_pcc(g_q, tt_q[:, :nh, :sl, :hd])
        # Transpose K back to [batch, heads, seq, head_dim] before comparing
        tt_k_t = tt_k.transpose(-2, -1)
        pcc_k = compute_pcc(g_k, tt_k_t[:, :nh, :sl, :hd])
        pcc_v = compute_pcc(g_v, tt_v[:, :nh, :sl, :hd])
        print_result("Q_heads", pcc_q, threshold)
        print_result("K_heads", pcc_k, threshold)
        print_result("V_heads", pcc_v, threshold)

        # --- Attention scores: Q @ K^T ---
        attn_scores = ttnn.matmul(q, k, memory_config=memory_config, compute_kernel_config=compute_config)
        ttnn.deallocate(k)

        # Scale
        head_dim = q.shape[-1]
        scale = head_dim ** -0.5
        attn_scores = ttnn.mul(attn_scores, scale, memory_config=memory_config)

        tt_scores = ttnn.to_torch(attn_scores)
        g_scores = golden_phase2["attn_scores_raw"]
        pcc_scores = compute_pcc(g_scores, tt_scores[:, :g_scores.shape[1], :g_scores.shape[2], :g_scores.shape[3]])
        print_result("attn_scores (Q@K^T*scale)", pcc_scores, threshold)

        # --- Softmax ---
        attn_probs = _softmax(attn_scores, dim=-1, memory_config=memory_config)
        ttnn.deallocate(attn_scores)

        tt_probs = ttnn.to_torch(attn_probs)
        g_probs = golden_phase2["attn_probs"]
        pcc_probs = compute_pcc(g_probs, tt_probs[:, :g_probs.shape[1], :g_probs.shape[2], :g_probs.shape[3]])
        print_result("attn_probs (softmax)", pcc_probs, threshold)

        # --- Context: probs @ V ---
        context = ttnn.matmul(attn_probs, v, memory_config=memory_config, compute_kernel_config=compute_config)
        ttnn.deallocate(attn_probs)
        ttnn.deallocate(v)
        ttnn.deallocate(q)

        tt_ctx = ttnn.to_torch(context)
        g_ctx = golden_phase2["context_heads"]
        pcc_ctx = compute_pcc(g_ctx, tt_ctx[:, :g_ctx.shape[1], :g_ctx.shape[2], :g_ctx.shape[3]])
        print_result("context_heads (probs@V)", pcc_ctx, threshold)

        # --- Concatenate heads ---
        context = ttnn.transformer.concatenate_heads(context, memory_config=memory_config)

        tt_concat = ttnn.to_torch(context)
        pcc_concat = compute_pcc(golden_phase2["context_concat"], tt_concat[:, :seq_len, :])
        print_result("context_concat", pcc_concat, threshold)

        # --- Output projection ---
        attn_output = ttnn.linear(
            context,
            layer_params["self_attn"]["out_proj_weight"],
            bias=layer_params["self_attn"]["out_proj_bias"],
            memory_config=memory_config,
            compute_kernel_config=compute_config,
        )
        ttnn.deallocate(context)

        tt_out = ttnn.to_torch(attn_output)
        pcc_out = compute_pcc(golden_phase2["out_proj"], tt_out[:, :seq_len, :])
        print_result("out_proj", pcc_out, threshold)

        # --- Residual 1 ---
        hidden_states = ttnn.add(residual, attn_output, memory_config=memory_config)
        ttnn.deallocate(residual)
        ttnn.deallocate(attn_output)

        tt_res1 = ttnn.to_torch(hidden_states)
        pcc_res1 = compute_pcc(golden_phase2["residual1"], tt_res1[:, :seq_len, :])
        print_result("residual1", pcc_res1, threshold)

        # --- Layer Norm 2 ---
        ln2 = ttnn.layer_norm(
            hidden_states,
            weight=layer_params["layer_norm2"]["weight"],
            bias=layer_params["layer_norm2"]["bias"],
            epsilon=config.layer_norm_eps,
            memory_config=memory_config,
        )

        tt_ln2 = ttnn.to_torch(ln2)
        pcc_ln2 = compute_pcc(golden_phase2["layer_norm2"], tt_ln2[:, :seq_len, :])
        print_result("layer_norm2", pcc_ln2, threshold)

        # --- MLP fc1 ---
        fc1 = ttnn.linear(
            ln2,
            layer_params["mlp"]["fc1_weight"],
            bias=layer_params["mlp"]["fc1_bias"],
            memory_config=memory_config,
            compute_kernel_config=compute_config,
        )
        ttnn.deallocate(ln2)

        tt_fc1 = ttnn.to_torch(fc1)
        pcc_fc1 = compute_pcc(golden_phase2["mlp_fc1"], tt_fc1[:, :seq_len, :])
        print_result("mlp_fc1", pcc_fc1, threshold)

        # --- GELU activation (stage 1 uses ttnn.gelu) ---
        gelu_out = ttnn.gelu(fc1, memory_config=memory_config)
        ttnn.deallocate(fc1)

        tt_gelu = ttnn.to_torch(gelu_out)
        pcc_gelu = compute_pcc(golden_phase2["mlp_quickgelu"], tt_gelu[:, :seq_len, :])
        print_result("mlp_quickgelu", pcc_gelu, threshold)

        # --- MLP fc2 ---
        fc2 = ttnn.linear(
            gelu_out,
            layer_params["mlp"]["fc2_weight"],
            bias=layer_params["mlp"]["fc2_bias"],
            memory_config=memory_config,
            compute_kernel_config=compute_config,
        )
        ttnn.deallocate(gelu_out)

        tt_fc2 = ttnn.to_torch(fc2)
        pcc_fc2 = compute_pcc(golden_phase2["mlp_fc2"], tt_fc2[:, :seq_len, :])
        print_result("mlp_fc2", pcc_fc2, threshold)

        # --- Residual 2 / block output ---
        block_out = ttnn.add(hidden_states, fc2, memory_config=memory_config)
        ttnn.deallocate(hidden_states)
        ttnn.deallocate(fc2)

        tt_block = ttnn.to_torch(block_out)
        ttnn.deallocate(block_out)
        pcc_block = compute_pcc(golden_phase2["block_output"], tt_block[:, :seq_len, :])
        print_result("block_output", pcc_block, threshold)

        return pcc_block
    except Exception as e:
        print(f"  [!!] ERROR: {e}")
        traceback.print_exc()
        return 0.0


def validate_phase3(golden, params, config, device):
    """Phase 3: Full vision encoder with per-layer comparison."""
    print_header("Phase 3: Full Vision Encoder (per-layer)")

    pixel_values = golden["inputs"]["pixel_values"]
    golden_phase3 = golden["phase3_vision_encoder"]
    golden_phase1 = golden["phase1_vision_embed"]

    try:
        from clip_vit_ttnn.tt.clip_model import vision_patch_embeddings, _get_encoder_layer_fn

        threshold = stage_threshold(config)
        # Per-layer PCC accumulates error through 12 layers (especially on
        # ttsim with manual softmax). Use a relaxed per-layer threshold for
        # diagnostics; the final pipeline PCC (Phase 5) is what matters.
        layer_threshold = 0.90 if config.stage >= 2 else threshold
        memory_config = config.get_memory_config()
        compute_config = config.get_compute_kernel_config()
        encoder_layer_fn = _get_encoder_layer_fn(config.stage)

        # Step 1: Patch embeddings
        hidden_states = vision_patch_embeddings(pixel_values, params["vision"], config, device)

        # Compare pre-layer-norm input
        golden_embed = golden_phase1["with_position_embed"]
        tt_embed_torch = ttnn.to_torch(hidden_states)
        seq_len = golden_embed.shape[1]
        pcc_embed = compute_pcc(golden_embed, tt_embed_torch[:, :seq_len, :])
        print_result("patch_embeddings", pcc_embed, threshold)

        # Step 2: Pre-layer norm
        hidden_states = ttnn.layer_norm(
            hidden_states,
            weight=params["vision"]["pre_layer_norm"]["weight"],
            bias=params["vision"]["pre_layer_norm"]["bias"],
            epsilon=config.layer_norm_eps,
            memory_config=memory_config,
        )

        # Compare pre-layer-norm output
        golden_ln_pre = golden_phase1["ln_pre_output"]
        tt_ln_pre = ttnn.to_torch(hidden_states)
        pcc_ln = compute_pcc(golden_ln_pre, tt_ln_pre[:, :seq_len, :])
        print_result("pre_layer_norm", pcc_ln, threshold)

        # Step 3: Encoder layers one by one
        for i in range(config.vision_num_layers):
            hidden_states = encoder_layer_fn(
                hidden_states,
                params["vision"]["layers"][i],
                num_heads=config.vision_num_heads,
                config=config,
                causal_mask=None,
            )

            golden_layer = golden_phase3[f"layer_{i}"]
            tt_layer = ttnn.to_torch(hidden_states)
            pcc_layer = compute_pcc(golden_layer, tt_layer[:, :seq_len, :])
            print_result(f"layer_{i:2d}", pcc_layer, layer_threshold)

        # Step 4: Post-layer norm on CLS token
        hidden_torch = ttnn.to_torch(hidden_states)
        ttnn.deallocate(hidden_states)
        cls_output = hidden_torch[:, 0, :].unsqueeze(0) if hidden_torch.dim() == 3 else hidden_torch[:, :1, :]
        cls_output = cls_output.squeeze(1) if cls_output.dim() == 3 else cls_output

        golden_post_ln = golden_phase3["post_layernorm_cls"]
        pcc_cls_raw = compute_pcc(golden_post_ln, cls_output)
        print_result("cls_token (pre-post_ln)", pcc_cls_raw, layer_threshold)

        if cls_output.dim() == 2:
            cls_output_for_ttnn = cls_output.unsqueeze(0)
        else:
            cls_output_for_ttnn = cls_output

        tt_cls = ttnn.from_torch(
            cls_output_for_ttnn,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=memory_config,
        )
        tt_cls = ttnn.layer_norm(
            tt_cls,
            weight=params["vision"]["post_layer_norm"]["weight"],
            bias=params["vision"]["post_layer_norm"]["bias"],
            epsilon=config.layer_norm_eps,
            memory_config=memory_config,
        )

        tt_post_ln = ttnn.to_torch(tt_cls).squeeze(0)
        if tt_post_ln.dim() > 2:
            tt_post_ln = tt_post_ln.squeeze(0)
        pcc_post_ln = compute_pcc(golden_post_ln, tt_post_ln)
        print_result("post_layer_norm", pcc_post_ln, layer_threshold)

        # Step 5: Visual projection
        vision_embed = ttnn.linear(
            tt_cls,
            params["vision"]["visual_projection_weight"],
            memory_config=memory_config,
            compute_kernel_config=compute_config,
        )
        ttnn.deallocate(tt_cls)

        vision_embed_torch = ttnn.to_torch(vision_embed).squeeze(0)
        if vision_embed_torch.dim() > 2:
            vision_embed_torch = vision_embed_torch.squeeze(0)
        ttnn.deallocate(vision_embed)

        golden_proj = golden_phase3["visual_projection"]
        pcc_proj = compute_pcc(golden_proj.flatten(), vision_embed_torch.flatten())
        print_result("visual_projection", pcc_proj, layer_threshold)

        return pcc_proj
    except Exception as e:
        print(f"  [!!] ERROR: {e}")
        traceback.print_exc()
        return 0.0


def validate_phase4(golden, params, config, device):
    """Phase 4: Text encoder (per-text, batch=1)."""
    print_header("Phase 4: Text Encoder")

    golden_phase4 = golden["phase4_text_encoder"]
    input_ids = golden["inputs"]["input_ids"]        # [3, 7]
    attention_mask = golden["inputs"]["attention_mask"]  # [3, 7]

    golden_proj = golden_phase4["text_projection"]     # [3, 512]

    num_texts = input_ids.shape[0]
    pccs = []

    for t in range(num_texts):
        text_name = golden["metadata"]["texts"][t] if "texts" in golden["metadata"] else f"text_{t}"
        ids_single = input_ids[t:t+1]       # [1, seq_len]
        mask_single = attention_mask[t:t+1]  # [1, seq_len]
        golden_single = golden_proj[t:t+1]   # [1, 512]

        try:
            tt_embed, text_embed_torch = run_text_encoder(
                ids_single, mask_single, params["text"], config, device
            )
            ttnn.deallocate(tt_embed)

            pcc = compute_pcc(golden_single.flatten(), text_embed_torch.flatten())
            print_result(f"text[{t}] \"{text_name}\"", pcc, stage_threshold(config))
            print(f"    Golden shape: {golden_single.shape}, TTNN shape: {text_embed_torch.shape}")
            pccs.append(pcc)
        except Exception as e:
            print(f"  [!!] text[{t}] \"{text_name}\" ERROR: {e}")
            traceback.print_exc()
            pccs.append(0.0)

    avg_pcc = sum(pccs) / len(pccs) if pccs else 0.0
    print(f"\n  Average text PCC: {avg_pcc:.6f}")
    return avg_pcc


def validate_phase5(golden, params, config, device):
    """Phase 5: Full pipeline — logits_per_image and predictions."""
    print_header("Phase 5: Full Pipeline")

    pixel_values = golden["inputs"]["pixel_values"]
    input_ids = golden["inputs"]["input_ids"]
    attention_mask = golden["inputs"]["attention_mask"]
    golden_phase5 = golden["phase5_full_pipeline"]

    try:
        # Vision
        tt_vision, vision_embed_torch = run_vision_encoder(
            pixel_values, params["vision"], config, device
        )
        ttnn.deallocate(tt_vision)

        # Text — run each text separately and concat
        text_embeds = []
        num_texts = input_ids.shape[0]
        for t in range(num_texts):
            tt_text, text_embed_torch = run_text_encoder(
                input_ids[t:t+1], attention_mask[t:t+1],
                params["text"], config, device,
            )
            ttnn.deallocate(tt_text)
            text_embeds.append(text_embed_torch)

        text_embed_all = torch.cat(text_embeds, dim=0)  # [N, 512]

        # Similarity
        logits = compute_similarity(vision_embed_torch, text_embed_all, params["logit_scale"])
        probs = logits.softmax(dim=-1)
        predicted_idx = probs.argmax(dim=-1).item()

        golden_logits = golden_phase5["logits_per_image"]
        golden_probs = golden_phase5["probs"]
        golden_idx = golden_phase5["predicted_idx"]

        pcc_logits = compute_pcc(golden_logits, logits)
        pcc_probs = compute_pcc(golden_probs, probs)

        print_result("logits_per_image", pcc_logits, threshold=0.98)
        print_result("probs", pcc_probs, threshold=0.98)
        print(f"  Golden logits: {golden_logits}")
        print(f"  TTNN logits:   {logits}")
        print(f"  Golden probs:  {golden_probs}")
        print(f"  TTNN probs:    {probs}")
        print(f"  Golden pred:   {golden_idx}, TTNN pred: {predicted_idx}  {'MATCH' if golden_idx == predicted_idx else 'MISMATCH'}")

        return pcc_logits
    except Exception as e:
        print(f"  [!!] ERROR: {e}")
        traceback.print_exc()
        return 0.0


def main():
    parser = argparse.ArgumentParser(description="Validate TTNN CLIP against golden reference")
    parser.add_argument("--golden", type=str, default="golden_reference.pt", help="Path to golden_reference.pt")
    parser.add_argument("--stage", type=int, default=1, help="TTNN stage (1, 2, or 3)")
    args = parser.parse_args()

    print("=" * 60)
    print("  CLIP-ViT TTNN Golden Reference Validation")
    print(f"  Golden: {args.golden}")
    print(f"  Stage:  {args.stage}")
    print("=" * 60)

    # Load golden reference
    print("\nLoading golden reference...")
    golden = torch.load(args.golden, map_location="cpu", weights_only=False)
    print(f"  Model: {golden['metadata']['model_name']}")
    print(f"  Generated with torch {golden['metadata']['torch_version']}, transformers {golden['metadata']['transformers_version']}")
    print(f"  Texts: {golden['metadata']['texts']}")
    print(f"  pixel_values: {golden['inputs']['pixel_values'].shape}")
    print(f"  input_ids: {golden['inputs']['input_ids'].shape}")

    # Load HF model for weight extraction
    print("\nLoading HF model for weight extraction...")
    from transformers import CLIPModel
    hf_model = CLIPModel.from_pretrained(golden["metadata"]["model_name"], attn_implementation="eager")
    hf_model.eval()

    # Initialize TTNN device
    print("\nInitializing TTNN device...")
    device = ttnn.open_device(device_id=0)

    # Create config and load weights
    config = CLIPTTNNConfig(stage=args.stage)
    params = load_all_weights(hf_model, device, config)

    # Add logit_scale to params for phase 5
    params["logit_scale"] = hf_model.logit_scale.data.clone()

    results = {}

    # Phase 1: Vision patch embeddings
    results["phase1"] = validate_phase1(golden, params, config, device)

    # Phase 2: Vision block 0 internals
    results["phase2"] = validate_phase2(golden, params, config, device)

    # Phase 3: Full vision encoder
    results["phase3"] = validate_phase3(golden, params, config, device)

    # Phase 4: Text encoder
    results["phase4"] = validate_phase4(golden, params, config, device)

    # Phase 5: Full pipeline
    results["phase5"] = validate_phase5(golden, params, config, device)

    # Summary
    threshold = stage_threshold(config)
    # Phase 3 measures per-layer accumulated error; use relaxed threshold
    phase3_threshold = 0.90 if config.stage >= 2 else threshold
    print_header(f"Summary (Stage {args.stage}, threshold={threshold})")
    for phase, pcc in results.items():
        t = phase3_threshold if phase == "phase3" else threshold
        status = pcc_status(pcc, t)
        print(f"  {phase}: PCC = {pcc:.6f}  ({status} @ {t})")

    # Close device
    print("\nClosing TTNN device...")
    ttnn.close_device(device)
    print("Done.")


if __name__ == "__main__":
    main()
