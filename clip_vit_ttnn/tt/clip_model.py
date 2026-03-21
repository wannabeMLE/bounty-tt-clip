# SPDX-License-Identifier: Apache-2.0
#
# CLIP-ViT Full Model Implementation in TTNN
# Supports 3 optimization stages:
#   Stage 1: Functional (DRAM, HiFi4, no fusion)
#   Stage 2: Optimized (L1 interleaved, LoFi, bfloat8_b weights)
#   Stage 3: Deep optimization (SDPA, program configs)
#
# Reference: OWL-ViT TTNN implementation in tt-metal

import os
import time
import torch
import torch.nn.functional as F
import ttnn
from typing import Dict, List, Optional, Tuple

from .weight_loader import CLIPTTNNConfig


# ---------------------------------------------------------------------------
# ttsim compatibility: ttnn.softmax uses SFPLOADMACRO which is not supported
# in the simulator. Use a manual decomposition when running under ttsim.
# ---------------------------------------------------------------------------
_ON_TTSIM = bool(os.environ.get("TT_METAL_SIMULATOR", ""))
_USE_MANUAL_SOFTMAX = _ON_TTSIM


def _quick_gelu(x, memory_config=None):
    """QuickGELU activation: x * sigmoid(1.702 * x).

    CLIP uses QuickGELU, not standard GELU. The difference matters for PCC.
    """
    scaled = ttnn.mul(x, 1.702, memory_config=memory_config)
    gate = ttnn.sigmoid(scaled, memory_config=memory_config)
    ttnn.deallocate(scaled)
    result = ttnn.multiply(x, gate, memory_config=memory_config)
    ttnn.deallocate(gate)
    return result


def _tick(timing_dict, device, key, t0_holder):
    """Record elapsed time for previous op, start timing next op.

    When timing_dict is None, this is a no-op (zero overhead).
    t0_holder is a 1-element list so it's mutable across calls.
    """
    if timing_dict is None:
        return
    ttnn.synchronize_device(device)
    now = time.perf_counter()
    if t0_holder[0] is not None and key is not None:
        timing_dict[key] = (now - t0_holder[0]) * 1000
    t0_holder[0] = now


def _softmax(x, dim=-1, memory_config=None):
    """Softmax that works on both real hardware and ttsim."""
    if not _USE_MANUAL_SOFTMAX:
        return ttnn.softmax(x, dim=dim, memory_config=memory_config)
    # Manual decomposition: max-subtract, exp, sum, reciprocal, multiply
    x_max = ttnn.max(x, dim=dim, keepdim=True)
    x_shifted = ttnn.subtract(x, x_max)
    ttnn.deallocate(x_max)
    x_exp = ttnn.exp(x_shifted)
    ttnn.deallocate(x_shifted)
    x_sum = ttnn.sum(x_exp, dim=dim, keepdim=True)
    x_sum_inv = ttnn.reciprocal(x_sum)
    ttnn.deallocate(x_sum)
    result = ttnn.multiply(x_exp, x_sum_inv)
    ttnn.deallocate(x_exp)
    ttnn.deallocate(x_sum_inv)
    return result


# ---------------------------------------------------------------------------
# Patch Embedding (runs on CPU in Stage 1, transfers result to device)
# ---------------------------------------------------------------------------
def vision_patch_embeddings(
    pixel_values: torch.Tensor,
    params: Dict,
    config: CLIPTTNNConfig,
    device
) -> ttnn.Tensor:
    """Convert image pixels to patch embeddings + CLS token + position embeddings.

    For Stage 1, we run the conv2d on CPU (simpler, avoids conv2d shape issues)
    and transfer the result to device.

    Args:
        pixel_values: [1, 3, 224, 224] float32 tensor (preprocessed image)
        params: vision encoder params dict from weight_loader
        config: model config

    Returns:
        TTNN tensor [1, seq_len_padded, hidden_size] on device in TILE_LAYOUT
    """
    patch_weight = params["patch_embedding_weight"]  # [768, 3, 32, 32]
    patch_bias = params["patch_embedding_bias"]

    with torch.no_grad():
        # Run patch embedding conv on CPU
        patches = F.conv2d(
            pixel_values, patch_weight, bias=patch_bias, stride=config.patch_size
        )  # [1, 768, H/P=7, W/P=7]
        patches = patches.flatten(2).transpose(1, 2)  # [1, num_patches, 768]

        # Prepend CLS token
        cls_token = params["cls_token"].unsqueeze(0).unsqueeze(0)  # [1, 1, 768]
        embeddings = torch.cat([cls_token, patches], dim=1)  # [1, seq_len, 768]

        # Add position embeddings
        position_embeddings = params["position_embeddings"]  # [seq_len, 768]
        embeddings = embeddings + position_embeddings.unsqueeze(0)

    # Transfer to device
    tt_embeddings = ttnn.from_torch(
        embeddings,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=config.get_memory_config(),
    )

    return tt_embeddings  # [1, 64, 768] (seq_len padded to tile boundary), the original [1, 50, 768]


# ---------------------------------------------------------------------------
# Stage 2+: On-Device Patch Embedding (fold + linear)
# ---------------------------------------------------------------------------
def vision_patch_embeddings_stage2(
    pixel_values: torch.Tensor,
    params: Dict,
    config: CLIPTTNNConfig,
    device,
) -> ttnn.Tensor:
    """On-device patch embedding using fold + linear (replaces CPU conv2d).

    CLIP-ViT-B/32: patch_size=32, image=224x224, 49 patches (7x7 grid).
    Steps:
      1. Pad channels 3→4 (tile alignment)
      2. Reshape to [1, 224, 7, 128] (group 32-pixel-wide columns into 4ch * 32px)
      3. fold(stride_h=32, stride_w=1) → [1, 49, 4096] (each patch flattened)
      4. linear(x, proj_weight) → [1, 49, 768]
      5. Prepend CLS token + add position embeddings on device

    On ttsim, fold may not be supported → falls back to CPU path.
    """
    memory_config = config.get_memory_config()
    compute_config = config.get_compute_kernel_config()

    if _ON_TTSIM:
        # fold+linear has concat shape issues on ttsim; use CPU path
        return vision_patch_embeddings(pixel_values, params, config, device)

    # Step 1: Pad channels 3→4 on CPU
    with torch.no_grad():
        x = torch.nn.functional.pad(pixel_values, (0, 0, 0, 0, 0, 1))  # [1, 4, 224, 224]

    # Step 2: Reshape to group patches horizontally: [1, 4*32, 224/32, 224] = [1, 128, 7, 224]
    # Then permute to [1, 224, 7, 128] for fold
    x = x.reshape(1, 128, 7, 224).permute(0, 3, 2, 1).contiguous()  # [1, 224, 7, 128]

    # Transfer to device
    tt_x = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=memory_config,
    )

    # Step 3: Fold — collapses stride_h=32 rows into the patch dimension
    # Input: [1, 224, 7, 128], fold with stride_h=32 stride_w=1
    # Output: [1, 7, 7, 4096] = [1, 49 patches flattened to batch*h, w=1, patch_dim]
    # Actually fold output shape depends on implementation. Let's reshape after.
    try:
        tt_folded = ttnn.fold(tt_x, stride_h=32, stride_w=1)
        ttnn.deallocate(tt_x)
    except Exception:
        # fold not supported — fall back to CPU
        ttnn.deallocate(tt_x)
        return vision_patch_embeddings(pixel_values, params, config, device)

    # Reshape to [1, 49, 4096] for linear
    folded_shape = tt_folded.shape
    tt_folded = ttnn.reshape(tt_folded, (1, config.vision_num_patches, -1))
    tt_folded = ttnn.to_layout(tt_folded, ttnn.TILE_LAYOUT)

    # Step 4: Linear projection [1, 49, 4096] @ [4096, 768] → [1, 49, 768]
    tt_patches = ttnn.linear(
        tt_folded,
        params["patch_linear_weight"],
        bias=params.get("patch_linear_bias"),
        memory_config=memory_config,
        compute_kernel_config=compute_config,
    )
    ttnn.deallocate(tt_folded)

    # Step 5: Prepend CLS token
    # CLS token is [1, 1, 768] on device, patches are [1, 49, 768]
    cls_token_tt = params["cls_token_tt"]  # [1, 1, 768] on device
    tt_embed = ttnn.concat([cls_token_tt, tt_patches], dim=1, memory_config=memory_config)
    ttnn.deallocate(tt_patches)

    # Step 6: Add position embeddings [1, 50, 768]
    pos_embed_tt = params["position_embeddings_tt"]  # [1, 50, 768] on device
    tt_embed = ttnn.add(tt_embed, pos_embed_tt, memory_config=memory_config)

    return tt_embed  # [1, 64, 768] (padded to tile boundary)


# ---------------------------------------------------------------------------
# Text Embedding (runs on CPU, transfers result to device)
# ---------------------------------------------------------------------------

def text_embeddings(
    input_ids: torch.Tensor,
    params: Dict,
    config: CLIPTTNNConfig,
    device,
) -> ttnn.Tensor:
    """Compute text token + position embeddings.

    Args:
        input_ids: [1, seq_len] int64 tensor
        params: text encoder params dict

    Returns:
        TTNN tensor [1, seq_len_padded, hidden_size] on device
    """
    with torch.no_grad():
        token_weight = params["token_embedding_weight"]  # [vocab_size, 512]
        position_weight = params["position_embedding_weight"]  # [77, 512]

        token_embeds = F.embedding(input_ids, token_weight)  # [1, seq_len, 512]
        seq_len = input_ids.shape[1]
        position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
        position_embeds = F.embedding(position_ids, position_weight)

        embeddings = token_embeds + position_embeds

    tt_embeddings = ttnn.from_torch(
        embeddings,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=config.get_memory_config(),
    )

    return tt_embeddings  # [1, 96, 512] (seq_len padded to tile boundary), the original [1, 77, 512]


# ---------------------------------------------------------------------------
# Causal Attention Mask (for text encoder)
# ---------------------------------------------------------------------------

def create_causal_mask(seq_len: int, config: CLIPTTNNConfig, device) -> ttnn.Tensor:
    """Create causal attention mask for text encoder.

    Returns a [1, num_heads, seq_len, seq_len] mask where masked positions are -inf.
    The logical shape matches attention_scores exactly; TTNN handles tile padding
    internally. Expanded to num_heads because ttsim does not support subtile broadcast.
    """
    mask = torch.full((seq_len, seq_len), float("-inf"))
    mask = torch.triu(mask, diagonal=1)

    # Expand to [1, num_heads, S, S] to match attention_scores shape exactly
    num_heads = config.text_num_heads
    mask = mask.unsqueeze(0).unsqueeze(0).expand(1, num_heads, -1, -1).contiguous()

    tt_mask = ttnn.from_torch(
        mask,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=config.get_memory_config(),
    )
    return tt_mask


# ---------------------------------------------------------------------------
# Stage 1: Basic Encoder Layer (DRAM, no fusion, no sharding)
# ---------------------------------------------------------------------------

def encoder_layer_stage1(
    hidden_states: ttnn.Tensor,
    layer_params: Dict,
    num_heads: int,
    config: CLIPTTNNConfig,
    causal_mask: Optional[ttnn.Tensor] = None,
    is_vision: bool = True,
    device=None,
    timing_dict: Optional[Dict] = None,
) -> ttnn.Tensor:
    """Run a single transformer encoder layer (Stage 1: basic, DRAM).

    Works for both vision and text encoder layers.
    Pass timing_dict={} and device to enable per-op profiling.
    """
    memory_config = ttnn.DRAM_MEMORY_CONFIG
    compute_config = config.get_compute_kernel_config()

    t0 = [None]
    _tick(timing_dict, device, None, t0)

    # --- Layer Norm 1 ---
    residual = hidden_states
    hidden_states = ttnn.layer_norm(
        hidden_states,
        weight=layer_params["layer_norm1"]["weight"],
        bias=layer_params["layer_norm1"]["bias"],
        epsilon=config.layer_norm_eps,
        memory_config=memory_config,
    )
    _tick(timing_dict, device, "layer_norm1", t0)

    # --- Self-Attention ---
    # QKV projection + split heads
    qkv = ttnn.linear(
        hidden_states,
        layer_params["self_attn"]["qkv_weight"],
        bias=layer_params["self_attn"]["qkv_bias"],
        memory_config=memory_config,
        compute_kernel_config=compute_config,
    )
    ttnn.deallocate(hidden_states)

    q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(
        qkv, num_heads=num_heads
    )
    ttnn.deallocate(qkv)
    _tick(timing_dict, device, "qkv_linear", t0)

    # Attention scores: Q @ K^T + scale + mask
    head_dim = q.shape[-1]
    attention_scores = ttnn.matmul(q, k, memory_config=memory_config, compute_kernel_config=compute_config)
    ttnn.deallocate(k)

    scale = head_dim ** -0.5
    attention_scores = ttnn.mul(attention_scores, scale, memory_config=memory_config)

    if causal_mask is not None:
        attention_scores = ttnn.add(attention_scores, causal_mask, memory_config=memory_config)
    _tick(timing_dict, device, "attn_scores", t0)

    # Softmax
    attention_probs = _softmax(attention_scores, dim=-1, memory_config=memory_config)
    ttnn.deallocate(attention_scores)
    _tick(timing_dict, device, "softmax", t0)

    # Attention context: probs @ V
    context = ttnn.matmul(attention_probs, v, memory_config=memory_config, compute_kernel_config=compute_config)
    ttnn.deallocate(attention_probs)
    ttnn.deallocate(v)
    ttnn.deallocate(q)
    _tick(timing_dict, device, "attn_context", t0)

    # Concatenate heads
    context = ttnn.transformer.concatenate_heads(context, memory_config=memory_config)
    _tick(timing_dict, device, "concat_heads", t0)

    # Output projection
    attn_output = ttnn.linear(
        context,
        layer_params["self_attn"]["out_proj_weight"],
        bias=layer_params["self_attn"]["out_proj_bias"],
        memory_config=memory_config,
        compute_kernel_config=compute_config,
    )
    ttnn.deallocate(context)
    _tick(timing_dict, device, "out_proj", t0)

    # Residual connection
    hidden_states = ttnn.add(residual, attn_output, memory_config=memory_config)
    ttnn.deallocate(residual)
    ttnn.deallocate(attn_output)
    _tick(timing_dict, device, "residual1_add", t0)

    # --- Layer Norm 2 ---
    residual = hidden_states
    hidden_states = ttnn.layer_norm(
        hidden_states,
        weight=layer_params["layer_norm2"]["weight"],
        bias=layer_params["layer_norm2"]["bias"],
        epsilon=config.layer_norm_eps,
        memory_config=memory_config,
    )
    _tick(timing_dict, device, "layer_norm2", t0)

    # --- MLP ---
    # fc1
    mlp = ttnn.linear(
        hidden_states,
        layer_params["mlp"]["fc1_weight"],
        bias=layer_params["mlp"]["fc1_bias"],
        memory_config=memory_config,
        compute_kernel_config=compute_config,
    )
    ttnn.deallocate(hidden_states)
    _tick(timing_dict, device, "fc1_linear", t0)

    # QuickGELU activation: x * sigmoid(1.702 * x)
    mlp = _quick_gelu(mlp, memory_config=memory_config)
    _tick(timing_dict, device, "quick_gelu", t0)

    # fc2
    mlp = ttnn.linear(
        mlp,
        layer_params["mlp"]["fc2_weight"],
        bias=layer_params["mlp"]["fc2_bias"],
        memory_config=memory_config,
        compute_kernel_config=compute_config,
    )
    _tick(timing_dict, device, "fc2_linear", t0)

    # Residual connection
    hidden_states = ttnn.add(residual, mlp, memory_config=memory_config)
    ttnn.deallocate(residual)
    ttnn.deallocate(mlp)
    _tick(timing_dict, device, "residual2_add", t0)

    return hidden_states


# ---------------------------------------------------------------------------
# Stage 2: Optimized Encoder Layer (L1, LoFi, GELU fusion, core_grid)
# ---------------------------------------------------------------------------

def encoder_layer_stage2(
    hidden_states: ttnn.Tensor,
    layer_params: Dict,
    num_heads: int,
    config: CLIPTTNNConfig,
    causal_mask: Optional[ttnn.Tensor] = None,
    is_vision: bool = True,
    device=None,
    timing_dict: Optional[Dict] = None,
) -> ttnn.Tensor:
    """Run a single transformer encoder layer (Stage 2: L1 + LoFi + fusion).

    Optimizations vs Stage 1:
    - bfloat8_b weights for linear ops (2x smaller, faster matmul)
    - L1 interleaved memory for all intermediates (~16x bandwidth vs DRAM)
    - LoFi/HiFi2 math fidelity (faster compute)
    - Program cache enabled (reuse compiled kernels across layers)

    Pass timing_dict={} and device to enable per-op profiling.
    """
    compute_config = config.get_compute_kernel_config()
    memory_config = config.get_memory_config()  # L1 interleaved for stage 2

    t0 = [None]
    _tick(timing_dict, device, None, t0)

    # --- Layer Norm 1 ---
    residual = hidden_states
    hidden_states = ttnn.layer_norm(
        hidden_states,
        weight=layer_params["layer_norm1"]["weight"],
        bias=layer_params["layer_norm1"]["bias"],
        epsilon=config.layer_norm_eps,
        memory_config=memory_config,
    )
    _tick(timing_dict, device, "layer_norm1", t0)

    # --- Self-Attention ---
    # QKV projection + split heads
    qkv = ttnn.linear(
        hidden_states,
        layer_params["self_attn"]["qkv_weight"],
        bias=layer_params["self_attn"]["qkv_bias"],
        memory_config=memory_config,
        compute_kernel_config=compute_config,
    )
    ttnn.deallocate(hidden_states)

    q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(
        qkv, num_heads=num_heads
    )
    ttnn.deallocate(qkv)
    _tick(timing_dict, device, "qkv_linear", t0)

    # Attention scores: Q @ K^T + scale + mask
    head_dim = q.shape[-1]
    attention_scores = ttnn.matmul(
        q, k, memory_config=memory_config, compute_kernel_config=compute_config
    )
    ttnn.deallocate(k)

    scale = head_dim ** -0.5
    attention_scores = ttnn.mul(attention_scores, scale, memory_config=memory_config)

    if causal_mask is not None:
        attention_scores = ttnn.add(
            attention_scores, causal_mask, memory_config=memory_config
        )
    _tick(timing_dict, device, "attn_scores", t0)

    # Softmax
    if _USE_MANUAL_SOFTMAX:
        attention_probs = _softmax(
            attention_scores, dim=-1, memory_config=memory_config
        )
        if attention_probs is not attention_scores:
            ttnn.deallocate(attention_scores)
    else:
        attention_probs = ttnn.softmax_in_place(attention_scores)
    _tick(timing_dict, device, "softmax", t0)

    # Attention context: probs @ V
    context = ttnn.matmul(
        attention_probs, v,
        memory_config=memory_config,
        compute_kernel_config=compute_config,
    )
    ttnn.deallocate(attention_probs)
    ttnn.deallocate(v)
    ttnn.deallocate(q)
    _tick(timing_dict, device, "attn_context", t0)

    # Concatenate heads
    context = ttnn.transformer.concatenate_heads(
        context, memory_config=memory_config
    )
    _tick(timing_dict, device, "concat_heads", t0)

    # Output projection
    attn_output = ttnn.linear(
        context,
        layer_params["self_attn"]["out_proj_weight"],
        bias=layer_params["self_attn"]["out_proj_bias"],
        memory_config=memory_config,
        compute_kernel_config=compute_config,
    )
    ttnn.deallocate(context)
    _tick(timing_dict, device, "out_proj", t0)

    # Residual connection
    hidden_states = ttnn.add(residual, attn_output, memory_config=memory_config)
    ttnn.deallocate(residual)
    ttnn.deallocate(attn_output)
    _tick(timing_dict, device, "residual1_add", t0)

    # --- Layer Norm 2 ---
    residual = hidden_states
    hidden_states = ttnn.layer_norm(
        hidden_states,
        weight=layer_params["layer_norm2"]["weight"],
        bias=layer_params["layer_norm2"]["bias"],
        epsilon=config.layer_norm_eps,
        memory_config=memory_config,
    )
    _tick(timing_dict, device, "layer_norm2", t0)

    # --- MLP with QuickGELU ---
    # NOTE: HEIGHT_SHARDED was tested but causes 14x matmul regression at this
    # tensor size ([1,64,768] = 2 tile-rows). Sharding restricts matmul to 2 cores
    # while L1 interleaved uses the full core grid. See test_reshard_cost.py.
    mlp = ttnn.linear(
        hidden_states,
        layer_params["mlp"]["fc1_weight"],
        bias=layer_params["mlp"]["fc1_bias"],
        memory_config=memory_config,
        compute_kernel_config=compute_config,
    )
    ttnn.deallocate(hidden_states)
    _tick(timing_dict, device, "fc1_linear", t0)

    # QuickGELU: x * sigmoid(1.702 * x)
    mlp = _quick_gelu(mlp, memory_config=memory_config)
    _tick(timing_dict, device, "quick_gelu", t0)

    # fc2
    mlp = ttnn.linear(
        mlp,
        layer_params["mlp"]["fc2_weight"],
        bias=layer_params["mlp"]["fc2_bias"],
        memory_config=memory_config,
        compute_kernel_config=compute_config,
    )
    _tick(timing_dict, device, "fc2_linear", t0)

    # Residual connection
    hidden_states = ttnn.add(residual, mlp, memory_config=memory_config)
    ttnn.deallocate(residual)
    ttnn.deallocate(mlp)
    _tick(timing_dict, device, "residual2_add", t0)

    return hidden_states


# ---------------------------------------------------------------------------
# Stage 3: Deep Optimized Encoder Layer (SDPA, program configs)
# ---------------------------------------------------------------------------

def _get_sdpa_config(config: CLIPTTNNConfig):
    """Create SDPA program config for FlashAttention."""
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=config.core_grid,
        q_chunk_size=32,
        k_chunk_size=32,
        exp_approx_mode=True,
    )


def _get_sdpa_compute_config():
    """Compute kernel config for SDPA (HiFi2 recommended)."""
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )


def encoder_layer_stage3(
    hidden_states: ttnn.Tensor,
    layer_params: Dict,
    num_heads: int,
    config: CLIPTTNNConfig,
    causal_mask: Optional[ttnn.Tensor] = None,
    is_causal: bool = False,
    is_vision: bool = True,
) -> ttnn.Tensor:
    """Run a single transformer encoder layer (Stage 3: SDPA + full optimization)."""
    memory_config = ttnn.L1_MEMORY_CONFIG
    compute_config = config.get_compute_kernel_config()
    full_grid = config.get_full_grid()

    # --- Layer Norm 1 ---
    residual = hidden_states
    hidden_states = ttnn.layer_norm(
        hidden_states,
        weight=layer_params["layer_norm1"]["weight"],
        bias=layer_params["layer_norm1"]["bias"],
        epsilon=config.layer_norm_eps,
        memory_config=memory_config,
    )

    # --- Self-Attention with SDPA ---
    qkv = ttnn.linear(
        hidden_states,
        layer_params["self_attn"]["qkv_weight"],
        bias=layer_params["self_attn"]["qkv_bias"],
        memory_config=memory_config,
        core_grid=full_grid,
        compute_kernel_config=compute_config,
    )
    ttnn.deallocate(hidden_states)

    # SDPA expects K in [B, H, S, D] format (not transposed)
    q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(
        qkv, num_heads=num_heads, transpose_key=False
    )
    ttnn.deallocate(qkv)

    # Fused scaled dot-product attention
    head_dim = q.shape[-1]
    scale = head_dim ** -0.5

    context = ttnn.transformer.scaled_dot_product_attention(
        q, k, v,
        is_causal=is_causal,
        attn_mask=causal_mask if (not is_causal and causal_mask is not None) else None,
        scale=scale,
        program_config=_get_sdpa_config(config),
        compute_kernel_config=_get_sdpa_compute_config(),
        memory_config=memory_config,
    )
    ttnn.deallocate(q)
    ttnn.deallocate(k)
    ttnn.deallocate(v)

    # Concatenate heads
    context = ttnn.transformer.concatenate_heads(context, memory_config=memory_config)

    # Output projection
    attn_output = ttnn.linear(
        context,
        layer_params["self_attn"]["out_proj_weight"],
        bias=layer_params["self_attn"]["out_proj_bias"],
        memory_config=memory_config,
        core_grid=full_grid,
        compute_kernel_config=compute_config,
    )
    ttnn.deallocate(context)

    # Residual
    hidden_states = ttnn.add(residual, attn_output, memory_config=memory_config)
    ttnn.deallocate(residual)
    ttnn.deallocate(attn_output)

    # --- Layer Norm 2 ---
    residual = hidden_states
    hidden_states = ttnn.layer_norm(
        hidden_states,
        weight=layer_params["layer_norm2"]["weight"],
        bias=layer_params["layer_norm2"]["bias"],
        epsilon=config.layer_norm_eps,
        memory_config=memory_config,
    )

    # --- MLP with QuickGELU ---
    mlp = ttnn.linear(
        hidden_states,
        layer_params["mlp"]["fc1_weight"],
        bias=layer_params["mlp"]["fc1_bias"],
        memory_config=memory_config,
        core_grid=full_grid,
        compute_kernel_config=compute_config,
    )
    ttnn.deallocate(hidden_states)

    mlp = _quick_gelu(mlp, memory_config=memory_config)

    mlp = ttnn.linear(
        mlp,
        layer_params["mlp"]["fc2_weight"],
        bias=layer_params["mlp"]["fc2_bias"],
        memory_config=memory_config,
        core_grid=full_grid,
        compute_kernel_config=compute_config,
    )

    hidden_states = ttnn.add(residual, mlp, memory_config=memory_config)
    ttnn.deallocate(residual)
    ttnn.deallocate(mlp)

    return hidden_states


# ---------------------------------------------------------------------------
# Encoder dispatch: selects the right layer function based on stage
# ---------------------------------------------------------------------------

def _get_encoder_layer_fn(stage: int):
    """Return the encoder layer function for the given optimization stage."""
    if stage == 1:
        return encoder_layer_stage1
    elif stage == 2:
        return encoder_layer_stage2
    elif stage == 3:
        return encoder_layer_stage3
    else:
        raise ValueError(f"Unknown stage: {stage}")


# ---------------------------------------------------------------------------
# Vision Encoder (full pipeline)
# ---------------------------------------------------------------------------

def run_vision_encoder(
    pixel_values: torch.Tensor,
    params: Dict,
    config: CLIPTTNNConfig,
    device,
    layer_timings: Optional[List[Dict]] = None,
) -> Tuple[ttnn.Tensor, torch.Tensor]:
    """Run the full CLIP vision encoder.

    Args:
        pixel_values: [1, 3, 224, 224] preprocessed image
        params: vision encoder params from weight_loader
        config: model config
        device: TTNN device
        layer_timings: optional list of dicts (one per layer) for per-op profiling

    Returns:
        (vision_embedding, vision_embedding_torch):
            vision_embedding: TTNN tensor [1, 512] (projected, on device)
            vision_embedding_torch: torch tensor [1, 512] (for similarity computation)
    """
    memory_config = config.get_memory_config()
    compute_config = config.get_compute_kernel_config()
    encoder_layer_fn = _get_encoder_layer_fn(config.stage)

    # Step 1: Patch embeddings (CPU conv2d for all stages)
    # NOTE: fold+linear on-device path exists (vision_patch_embeddings_stage2) but has
    # PCC -0.02 due to reshape ordering issues. Keep CPU conv2d until fixed.
    hidden_states = vision_patch_embeddings(pixel_values, params, config, device)

    # Step 2: Pre-layer norm
    hidden_states = ttnn.layer_norm(
        hidden_states,
        weight=params["pre_layer_norm"]["weight"],
        bias=params["pre_layer_norm"]["bias"],
        epsilon=config.layer_norm_eps,
        memory_config=memory_config,
    )

    # Step 3: Encoder layers
    for i in range(config.vision_num_layers):
        layer_td = layer_timings[i] if layer_timings is not None else None
        hidden_states = encoder_layer_fn(
            hidden_states,
            params["layers"][i],
            num_heads=config.vision_num_heads,
            config=config,
            causal_mask=None,  # Vision encoder: no causal mask
            is_vision=True,
            device=device if layer_td is not None else None,
            timing_dict=layer_td,
        )

    # Step 4: Post-layer norm (on CLS token)
    # Move back to torch to extract CLS token (index 0)
    hidden_torch = ttnn.to_torch(hidden_states)
    ttnn.deallocate(hidden_states)

    # Extract CLS token [1, 768] — first token in sequence
    cls_output = hidden_torch[:, 0, :].unsqueeze(0) if hidden_torch.dim() == 3 else hidden_torch[:, :1, :]
    cls_output = cls_output.squeeze(1) if cls_output.dim() == 3 else cls_output

    # Reshape to [1, 1, 768] for TTNN
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

    # Post-layer norm
    tt_cls = ttnn.layer_norm(
        tt_cls,
        weight=params["post_layer_norm"]["weight"],
        bias=params["post_layer_norm"]["bias"],
        epsilon=config.layer_norm_eps,
        memory_config=memory_config,
    )

    # Step 5: Visual projection [768] -> [512]
    vision_embed = ttnn.linear(
        tt_cls,
        params["visual_projection_weight"],
        memory_config=memory_config,
        compute_kernel_config=compute_config,
    )
    ttnn.deallocate(tt_cls)

    # Convert to torch for similarity computation
    vision_embed_torch = ttnn.to_torch(vision_embed).squeeze(0)
    if vision_embed_torch.dim() > 2:
        vision_embed_torch = vision_embed_torch.squeeze(0)

    return vision_embed, vision_embed_torch


# ---------------------------------------------------------------------------
# Text Encoder (full pipeline)
# ---------------------------------------------------------------------------

def run_text_encoder(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    params: Dict,
    config: CLIPTTNNConfig,
    device,
    layer_timings: Optional[List[Dict]] = None,
) -> Tuple[ttnn.Tensor, torch.Tensor]:
    """Run the full CLIP text encoder.

    Args:
        input_ids: [1, seq_len] token IDs
        attention_mask: [1, seq_len] attention mask
        params: text encoder params from weight_loader
        config: model config
        device: TTNN device
        layer_timings: optional list of dicts (one per layer) for per-op profiling

    Returns:
        (text_embedding, text_embedding_torch):
            text_embedding: TTNN tensor [1, 512] (projected, on device)
            text_embedding_torch: torch tensor [1, 512]
    """
    memory_config = config.get_memory_config()
    compute_config = config.get_compute_kernel_config()
    encoder_layer_fn = _get_encoder_layer_fn(config.stage)
    seq_len = input_ids.shape[1]

    # Step 1: Token + position embeddings (on CPU, transferred to device)
    hidden_states = text_embeddings(input_ids, params, config, device)

    # Step 2: Create causal mask
    is_causal = (config.stage >= 3)  # Stage 3 uses SDPA is_causal flag
    causal_mask = None
    if config.stage < 3:
        causal_mask = create_causal_mask(seq_len, config, device)

    # Step 3: Encoder layers
    for i in range(config.text_num_layers):
        layer_td = layer_timings[i] if layer_timings is not None else None
        if config.stage == 3:
            hidden_states = encoder_layer_fn(
                hidden_states,
                params["layers"][i],
                num_heads=config.text_num_heads,
                config=config,
                causal_mask=None,
                is_causal=True,
                is_vision=False,
            )
        else:
            hidden_states = encoder_layer_fn(
                hidden_states,
                params["layers"][i],
                num_heads=config.text_num_heads,
                config=config,
                causal_mask=causal_mask,
                is_vision=False,
                device=device if layer_td is not None else None,
                timing_dict=layer_td,
            )

    # Deallocate causal mask
    if causal_mask is not None:
        ttnn.deallocate(causal_mask)

    # Step 4: Final layer norm
    hidden_states = ttnn.layer_norm(
        hidden_states,
        weight=params["final_layer_norm"]["weight"],
        bias=params["final_layer_norm"]["bias"],
        epsilon=config.layer_norm_eps,
        memory_config=memory_config,
    )

    # Step 5: Pool from EOS token position
    hidden_torch = ttnn.to_torch(hidden_states)
    ttnn.deallocate(hidden_states)

    # EOS token is at the position of the highest token ID
    eos_indices = input_ids.argmax(dim=-1)
    if hidden_torch.dim() == 4:
        hidden_torch = hidden_torch.squeeze(0)
    pooled = hidden_torch[torch.arange(hidden_torch.shape[0]), eos_indices]  # [1, 512]

    # Reshape for TTNN
    pooled_for_ttnn = pooled.unsqueeze(0) if pooled.dim() == 2 else pooled
    tt_pooled = ttnn.from_torch(
        pooled_for_ttnn,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=memory_config,
    )

    # Step 6: Text projection [512] -> [512]
    text_embed = ttnn.linear(
        tt_pooled,
        params["text_projection_weight"],
        memory_config=memory_config,
        compute_kernel_config=compute_config,
    )
    ttnn.deallocate(tt_pooled)

    # Convert to torch
    text_embed_torch = ttnn.to_torch(text_embed).squeeze(0)
    if text_embed_torch.dim() > 2:
        text_embed_torch = text_embed_torch.squeeze(0)

    return text_embed, text_embed_torch


# ---------------------------------------------------------------------------
# Similarity Computation
# ---------------------------------------------------------------------------

def compute_similarity(
    vision_embed_torch: torch.Tensor,
    text_embed_torch: torch.Tensor,
    logit_scale: torch.Tensor,
) -> torch.Tensor:
    """Compute cosine similarity between vision and text embeddings.

    Args:
        vision_embed_torch: [1, 512] vision embedding
        text_embed_torch: [N, 512] text embeddings (N texts)
        logit_scale: scalar learned temperature parameter

    Returns:
        logits_per_image: [1, N] similarity scores
    """
    # L2 normalize
    vision_embed = vision_embed_torch / vision_embed_torch.norm(p=2, dim=-1, keepdim=True)
    text_embed = text_embed_torch / text_embed_torch.norm(p=2, dim=-1, keepdim=True)

    # Cosine similarity with temperature
    logit_scale_val = logit_scale.exp()
    logits_per_image = logit_scale_val * (vision_embed @ text_embed.T)

    return logits_per_image


# ---------------------------------------------------------------------------
# Full CLIP Model
# ---------------------------------------------------------------------------

class CLIPModelTTNN:
    """Full CLIP model wrapping vision encoder, text encoder, and similarity."""

    def __init__(self, hf_model, device, config: CLIPTTNNConfig):
        self.config = config
        self.device = device

        # Load all weights
        from .weight_loader import load_all_weights
        self.params = load_all_weights(hf_model, device, config)

        print(f"[CLIP-TTNN] Loaded weights for stage {config.stage}")
        print(f"  Vision: {config.vision_num_layers} layers, hidden={config.vision_hidden_size}")
        print(f"  Text:   {config.text_num_layers} layers, hidden={config.text_hidden_size}")
        _sim = _ON_TTSIM
        print(f"  Memory: {'DRAM (ttsim)' if _sim else 'L1' if config.stage >= 2 else 'DRAM'}")
        print(f"  Math:   {'HiFi2 (ttsim)' if (_sim and config.stage >= 2) else 'LoFi' if config.stage >= 2 else 'HiFi4'}")
        print(f"  Weights: {'bfloat16 (ttsim)' if _sim else 'bfloat8_b' if config.stage >= 2 else 'bfloat16'}")
        print(f"  GELU:   QuickGELU (3 ops)")
        print(f"  Sharding: {'off (ttsim)' if _sim else 'L1 interleaved' if config.stage >= 2 else 'off'}")
        print(f"  Patch embed: {'CPU (ttsim)' if _sim else 'CPU conv2d'}")
        print(f"  SDPA:   {'Yes' if config.stage >= 3 else 'No'}")

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode image to embedding. Returns [1, 512] torch tensor."""
        _, embed_torch = run_vision_encoder(
            pixel_values, self.params["vision"], self.config, self.device
        )
        return embed_torch

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode text to embedding. Returns [N, 512] torch tensor."""
        _, embed_torch = run_text_encoder(
            input_ids, attention_mask, self.params["text"], self.config, self.device
        )
        return embed_torch

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run full CLIP forward pass.

        Returns logits_per_image [1, N] similarity scores.
        """
        vision_embed = self.encode_image(pixel_values)
        text_embed = self.encode_text(input_ids, attention_mask)
        logits = compute_similarity(vision_embed, text_embed, self.params["logit_scale"])
        return logits

    def set_stage(self, stage: int):
        """Switch optimization stage (requires reloading weights if sharding changes)."""
        self.config.stage = stage
        print(f"[CLIP-TTNN] Switched to stage {stage}")
