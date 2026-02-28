# SPDX-License-Identifier: Apache-2.0
#
# PyTorch Reference Implementation for CLIP-ViT
# Used as ground truth for validating the TTNN implementation.

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer
from typing import Tuple, Dict, Optional


MODEL_NAME = "openai/clip-vit-base-patch32"


def _to_tensor(output):
    """Extract tensor from model output (handles transformers 5.x API change)."""
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "pooler_output"):
        return output.pooler_output
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    raise TypeError(f"Unexpected output type: {type(output)}")


def load_clip_model(model_name: str = MODEL_NAME):
    """Load HuggingFace CLIP model and processor."""
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()
    return model, processor


def encode_image_pytorch(model, processor, image: Image.Image) -> torch.Tensor:
    """Encode a single image using the PyTorch reference model.

    Returns the normalized image embedding [1, 512].
    """
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        image_features = _to_tensor(model.get_image_features(**inputs))
    # L2 normalize
    image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
    return image_features


def encode_text_pytorch(model, processor, texts: list) -> torch.Tensor:
    """Encode text strings using the PyTorch reference model.

    Returns the normalized text embeddings [N, 512].
    """
    inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        text_features = _to_tensor(model.get_text_features(**inputs))
    text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
    return text_features


def compute_similarity_pytorch(model, processor, image: Image.Image, texts: list) -> torch.Tensor:
    """Compute image-text similarity scores using the PyTorch reference.

    Returns logits_per_image [1, N] where N = number of text inputs.
    """
    inputs = processor(text=texts, images=image, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.logits_per_image


def get_vision_encoder_intermediates(model, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Run vision encoder and capture intermediate outputs for per-layer validation.

    Args:
        model: HuggingFace CLIPModel
        pixel_values: Preprocessed image tensor [1, 3, 224, 224]

    Returns dict with keys:
        - "embeddings": output of patch embedding + CLS + position [1, 50, 768]
        - "pre_layer_norm": output after pre-layernorm
        - "layer_0" .. "layer_11": output of each encoder layer [1, 50, 768]
        - "post_layer_norm": output after post-layernorm
        - "pooled": CLS token output [1, 768]
        - "projected": after visual projection [1, 512]
    """
    vision = model.vision_model
    intermediates = {}

    with torch.no_grad():
        # Patch embedding + CLS + position
        embeddings = vision.embeddings(pixel_values)
        intermediates["embeddings"] = embeddings.clone()

        # Pre-layernorm
        hidden = vision.pre_layrnorm(embeddings)
        intermediates["pre_layer_norm"] = hidden.clone()

        # Encoder layers
        for i, layer in enumerate(vision.encoder.layers):
            hidden = layer(hidden, attention_mask=None, causal_attention_mask=None)[0]
            intermediates[f"layer_{i}"] = hidden.clone()

        # Post-layernorm on CLS token
        pooled = hidden[:, 0, :]  # CLS token
        pooled = vision.post_layernorm(pooled)
        intermediates["pooled"] = pooled.clone()

        # Visual projection
        projected = model.visual_projection(pooled)
        intermediates["projected"] = projected.clone()

    return intermediates


def get_text_encoder_intermediates(
    model, input_ids: torch.Tensor, attention_mask: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """Run text encoder and capture intermediate outputs for per-layer validation.

    Returns dict with keys:
        - "embeddings": token + position embeddings [1, seq_len, 512]
        - "layer_0" .. "layer_11": output of each encoder layer
        - "final_layer_norm": after final layer norm
        - "pooled": EOS token output [1, 512]
        - "projected": after text projection [1, 512]
    """
    text = model.text_model
    intermediates = {}

    with torch.no_grad():
        # Embeddings
        embeddings = text.embeddings(input_ids=input_ids, position_ids=None)
        intermediates["embeddings"] = embeddings.clone()

        # Build causal mask
        seq_len = input_ids.shape[1]
        causal_mask = torch.full((seq_len, seq_len), float("-inf"))
        causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, S, S]

        # Encoder layers
        hidden = embeddings
        for i, layer in enumerate(text.encoder.layers):
            hidden = layer(hidden, attention_mask=None, causal_attention_mask=causal_mask)[0]
            intermediates[f"layer_{i}"] = hidden.clone()

        # Final layer norm
        hidden = text.final_layer_norm(hidden)
        intermediates["final_layer_norm"] = hidden.clone()

        # Pool: take features from EOS token (highest token ID position)
        eos_indices = input_ids.argmax(dim=-1)  # EOS = highest token id
        pooled = hidden[torch.arange(hidden.shape[0]), eos_indices]
        intermediates["pooled"] = pooled.clone()

        # Text projection
        projected = model.text_projection(pooled)
        intermediates["projected"] = projected.clone()

    return intermediates


def compute_pcc(tensor_a: torch.Tensor, tensor_b: torch.Tensor) -> float:
    """Compute Pearson Correlation Coefficient between two tensors.

    Returns a value between -1 and 1. For correctness, should be > 0.99 (Stage 1),
    > 0.98 (Stage 2), or > 0.97 (Stage 3).
    """
    a = tensor_a.detach().float().flatten()
    b = tensor_b.detach().float().flatten()

    if a.shape != b.shape:
        min_len = min(len(a), len(b))
        a = a[:min_len]
        b = b[:min_len]

    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


if __name__ == "__main__":
    print("Loading CLIP model...")
    model, processor = load_clip_model()

    # Sample inference
    from urllib.request import urlopen
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(urlopen(url))

    texts = ["a photo of a cat", "a photo of a dog", "two cats on a couch"]
    similarity = compute_similarity_pytorch(model, processor, image, texts)

    probs = similarity.softmax(dim=-1)
    print("\nImage-text similarity:")
    for i, text in enumerate(texts):
        print(f"  '{text}': {probs[0, i].item():.4f}")
