# SPDX-License-Identifier: Apache-2.0
#
# CLIP-ViT Weight Loader
# Loads HuggingFace CLIP weights and converts them to TTNN format.
# Handles QKV fusion, weight transposition, and tile-aligned padding.

import torch
import ttnn
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class CLIPTTNNConfig:
    """Configuration for CLIP-ViT on Tenstorrent hardware."""

    # Vision encoder (ViT-B/32)
    vision_hidden_size: int = 768  # Embedding dim for each vision token
    vision_num_heads: int = 12  # Number of attention heads in vision self-attention
    vision_head_dim: int = 64  # Dim per attention head (hidden_size / num_heads)
    vision_intermediate_size: int = 3072  # MLP hidden dim inside each transformer block
    vision_num_layers: int = 12  # Number of vision transformer encoder layers
    patch_size: int = 32  # Size of each image patch in pixels (32x32)
    image_size: int = 224  # Input image resolution (224x224)
    vision_num_patches: int = 49  # Total patches: (224/32)^2 = 49
    vision_seq_len: int = 50  # num_patches + 1 (+1 for CLS token)
    vision_seq_len_padded: int = 64  # Seq len padded to tile boundary (multiple of 32) for TTNN

    # Text encoder
    text_hidden_size: int = 512  # Embedding dim for each text token
    text_num_heads: int = 8  # Number of attention heads in text self-attention
    text_head_dim: int = 64  # Dim per attention head (hidden_size / num_heads)
    text_intermediate_size: int = 2048  # MLP hidden dim in text transformer blocks
    text_num_layers: int = 12  # Number of text transformer encoder layers
    text_max_position_embeddings: int = 77  # Max text sequence length (CLIP's fixed 77-token limit)
    text_seq_len_padded: int = 96  # Padded to tile boundary (ceil(77/32)*32) for TTNN
    vocab_size: int = 49408  # Size of the BPE tokenizer vocabulary

    # Projection
    projection_dim: int = 512  # Shared embedding space dim for vision-text cosine similarity

    # Hardware
    layer_norm_eps: float = 1e-5  # Epsilon for LayerNorm numerical stability
    core_grid: tuple = (8, 7)  # Wormhole B0 compute core grid (8x7 = 56 cores)

    # Optimization stage: 1=basic DRAM+HiFi4, 2=L1+LoFi+fusion, 3=SDPA+full optimization
    stage: int = 1

    @classmethod
    def from_huggingface(cls, hf_config) -> "CLIPTTNNConfig":
        """Create config from HuggingFace CLIPConfig."""
        v = hf_config.vision_config
        t = hf_config.text_config
        image_size = v.image_size
        patch_size = v.patch_size
        num_patches = (image_size // patch_size) ** 2
        seq_len = num_patches + 1  # +1 for CLS token

        def pad_to_tile(n, tile=32):
            return ((n + tile - 1) // tile) * tile

        return cls(
            vision_hidden_size=v.hidden_size,
            vision_num_heads=v.num_attention_heads,
            vision_head_dim=v.hidden_size // v.num_attention_heads,
            vision_intermediate_size=v.intermediate_size,
            vision_num_layers=v.num_hidden_layers,
            patch_size=patch_size,
            image_size=image_size,
            vision_num_patches=num_patches,
            vision_seq_len=seq_len,
            vision_seq_len_padded=pad_to_tile(seq_len),
            text_hidden_size=t.hidden_size,
            text_num_heads=t.num_attention_heads,
            text_head_dim=t.hidden_size // t.num_attention_heads,
            text_intermediate_size=t.intermediate_size,
            text_num_layers=t.num_hidden_layers,
            text_max_position_embeddings=t.max_position_embeddings,
            text_seq_len_padded=pad_to_tile(t.max_position_embeddings),
            vocab_size=t.vocab_size,
            projection_dim=hf_config.projection_dim,
        )

    def get_compute_kernel_config(self):
        """Return compute kernel config based on optimization stage."""
        if self.stage >= 2:
            return ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.LoFi,
                math_approx_mode=True,
                fp32_dest_acc_en=False,
            )
        return ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
        )

    def get_memory_config(self):
        """Return memory config based on optimization stage."""
        if self.stage >= 2:
            return ttnn.L1_MEMORY_CONFIG
        return ttnn.DRAM_MEMORY_CONFIG

    def get_full_grid(self):
        """Return full compute core grid."""
        return ttnn.CoreGrid(y=self.core_grid[1], x=self.core_grid[0])


def _to_ttnn_weight(tensor, device, dtype=ttnn.bfloat16):
    """Convert a PyTorch tensor to a TTNN tensor on device in TILE_LAYOUT."""
    return ttnn.from_torch(
        tensor.unsqueeze(0).unsqueeze(0) if tensor.dim() == 1 else tensor,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )


def _fuse_qkv(q_proj, k_proj, v_proj):
    """Fuse separate Q, K, V projections into a single QKV weight+bias.

    PyTorch linear stores weights as [out_features, in_features].
    We concatenate along dim=0 to get [3*out, in], then transpose for TTNN.
    """
    qkv_weight = torch.cat([q_proj.weight, k_proj.weight, v_proj.weight], dim=0)
    qkv_bias = torch.cat([q_proj.bias, k_proj.bias, v_proj.bias], dim=0)
    return qkv_weight, qkv_bias


def load_vision_encoder_weights(hf_model, device, config: CLIPTTNNConfig) -> Dict:
    """Load and convert vision encoder weights from HuggingFace CLIP model.

    Returns a dict with:
      - "embeddings": patch embedding, cls_token, position_embeddings
      - "layers": list of per-layer weight dicts
      - "final_layer_norm": weight and bias
      - "visual_projection": weight
    """
    vision = hf_model.vision_model
    dtype = ttnn.bfloat16

    params = {}

    # --- Embeddings (these stay on CPU for Stage 1 patch embedding) ---
    params["patch_embedding_weight"] = vision.embeddings.patch_embedding.weight.data.clone()  # [768, 3, 32, 32]
    params["patch_embedding_bias"] = (  # [768] or None
        vision.embeddings.patch_embedding.bias.data.clone()
        if vision.embeddings.patch_embedding.bias is not None
        else None
    )
    params["cls_token"] = vision.embeddings.class_embedding.data.clone()  # [768]
    params["position_embeddings"] = vision.embeddings.position_embedding.weight.data.clone()  # [50, 768]

    # --- Pre-layer norm (CLIP ViT uses pre_layernorm before encoder) ---
    params["pre_layer_norm"] = {
        "weight": _to_ttnn_weight(vision.pre_layrnorm.weight.data, device, dtype),  # [1, 1, 768]
        "bias": _to_ttnn_weight(vision.pre_layrnorm.bias.data, device, dtype),  # [1, 1, 768]
    }

    # --- Encoder layers ---
    params["layers"] = []
    for i in range(config.vision_num_layers):
        layer = vision.encoder.layers[i]
        attn = layer.self_attn

        # Fuse QKV
        qkv_weight, qkv_bias = _fuse_qkv(attn.q_proj, attn.k_proj, attn.v_proj)

        layer_params = {
            "layer_norm1": {
                "weight": _to_ttnn_weight(layer.layer_norm1.weight.data, device, dtype),  # [1, 1, 768]
                "bias": _to_ttnn_weight(layer.layer_norm1.bias.data, device, dtype),  # [1, 1, 768]
            },
            "self_attn": {
                "qkv_weight": _to_ttnn_weight(qkv_weight.T.contiguous(), device, dtype),  # [768, 2304]
                "qkv_bias": _to_ttnn_weight(qkv_bias, device, dtype),  # [1, 1, 2304]
                "out_proj_weight": _to_ttnn_weight(attn.out_proj.weight.T.contiguous(), device, dtype),  # [768, 768]
                "out_proj_bias": _to_ttnn_weight(attn.out_proj.bias, device, dtype),  # [1, 1, 768]
            },
            "layer_norm2": {
                "weight": _to_ttnn_weight(layer.layer_norm2.weight.data, device, dtype),  # [1, 1, 768]
                "bias": _to_ttnn_weight(layer.layer_norm2.bias.data, device, dtype),  # [1, 1, 768]
            },
            "mlp": {
                "fc1_weight": _to_ttnn_weight(layer.mlp.fc1.weight.T.contiguous(), device, dtype),  # [768, 3072]
                "fc1_bias": _to_ttnn_weight(layer.mlp.fc1.bias, device, dtype),  # [1, 1, 3072]
                "fc2_weight": _to_ttnn_weight(layer.mlp.fc2.weight.T.contiguous(), device, dtype),  # [3072, 768]
                "fc2_bias": _to_ttnn_weight(layer.mlp.fc2.bias, device, dtype),  # [1, 1, 768]
            },
        }
        params["layers"].append(layer_params)

    # --- Post-encoder layer norm ---
    params["post_layer_norm"] = {
        "weight": _to_ttnn_weight(vision.post_layernorm.weight.data, device, dtype),  # [1, 1, 768]
        "bias": _to_ttnn_weight(vision.post_layernorm.bias.data, device, dtype),  # [1, 1, 768]
    }

    # --- Visual projection ---
    params["visual_projection_weight"] = _to_ttnn_weight(
        hf_model.visual_projection.weight.T.contiguous(), device, dtype  # [768, 512]
    )

    return params


def load_text_encoder_weights(hf_model, device, config: CLIPTTNNConfig) -> Dict:
    """Load and convert text encoder weights from HuggingFace CLIP model.

    Returns a dict with:
      - "token_embedding": weight (stays on CPU for embedding lookup)
      - "position_embeddings": weight (stays on CPU)
      - "layers": list of per-layer weight dicts
      - "final_layer_norm": weight and bias
      - "text_projection": weight
    """
    text = hf_model.text_model
    dtype = ttnn.bfloat16

    params = {}

    # --- Embeddings (stay on CPU for lookup) ---
    params["token_embedding_weight"] = text.embeddings.token_embedding.weight.data.clone()  # [49408, 512]
    params["position_embedding_weight"] = text.embeddings.position_embedding.weight.data.clone()  # [77, 512]

    # --- Encoder layers ---
    params["layers"] = []
    for i in range(config.text_num_layers):
        layer = text.encoder.layers[i]
        attn = layer.self_attn

        # Fuse QKV
        qkv_weight, qkv_bias = _fuse_qkv(attn.q_proj, attn.k_proj, attn.v_proj)

        layer_params = {
            "layer_norm1": {
                "weight": _to_ttnn_weight(layer.layer_norm1.weight.data, device, dtype),  # [1, 1, 512]
                "bias": _to_ttnn_weight(layer.layer_norm1.bias.data, device, dtype),  # [1, 1, 512]
            },
            "self_attn": {
                "qkv_weight": _to_ttnn_weight(qkv_weight.T.contiguous(), device, dtype),  # [512, 1536]
                "qkv_bias": _to_ttnn_weight(qkv_bias, device, dtype),  # [1, 1, 1536]
                "out_proj_weight": _to_ttnn_weight(attn.out_proj.weight.T.contiguous(), device, dtype),  # [512, 512]
                "out_proj_bias": _to_ttnn_weight(attn.out_proj.bias, device, dtype),  # [1, 1, 512]
            },
            "layer_norm2": {
                "weight": _to_ttnn_weight(layer.layer_norm2.weight.data, device, dtype),  # [1, 1, 512]
                "bias": _to_ttnn_weight(layer.layer_norm2.bias.data, device, dtype),  # [1, 1, 512]
            },
            "mlp": {
                "fc1_weight": _to_ttnn_weight(layer.mlp.fc1.weight.T.contiguous(), device, dtype),  # [512, 2048]
                "fc1_bias": _to_ttnn_weight(layer.mlp.fc1.bias, device, dtype),  # [1, 1, 2048]
                "fc2_weight": _to_ttnn_weight(layer.mlp.fc2.weight.T.contiguous(), device, dtype),  # [2048, 512]
                "fc2_bias": _to_ttnn_weight(layer.mlp.fc2.bias, device, dtype),  # [1, 1, 512]
            },
        }
        params["layers"].append(layer_params)

    # --- Final layer norm ---
    params["final_layer_norm"] = {
        "weight": _to_ttnn_weight(text.final_layer_norm.weight.data, device, dtype),  # [1, 1, 512]
        "bias": _to_ttnn_weight(text.final_layer_norm.bias.data, device, dtype),  # [1, 1, 512]
    }

    # --- Text projection ---
    params["text_projection_weight"] = _to_ttnn_weight(
        hf_model.text_projection.weight.T.contiguous(), device, dtype  # [512, 512]
    )

    return params


def  load_all_weights(hf_model, device, config: CLIPTTNNConfig) -> Dict:
    """Load all CLIP weights for both encoders."""
    return {
        "vision": load_vision_encoder_weights(hf_model, device, config),
        "text": load_text_encoder_weights(hf_model, device, config),
        "logit_scale": hf_model.logit_scale.data.clone(),  # scalar []
    }
