# CLIP-ViT CPU Test — no Tenstorrent hardware needed
# Tests the PyTorch reference model against COCO val2017 images.
#
# Usage:
#   python test_cpu.py                  # 10 random COCO images
#   python test_cpu.py --num_images 50  # 50 images
#   python test_cpu.py --local_dir ./coco_images  # use pre-downloaded images

import argparse
import time
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

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

# COCO-style candidate labels for zero-shot classification
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


def load_coco_from_hub(num_images: int):
    """Load COCO val2017 images from HuggingFace datasets."""
    from datasets import load_dataset

    print(f"Loading COCO val2017 from HuggingFace hub ({num_images} images)...")
    ds = load_dataset(
        "detection-datasets/coco",
        split="val",
        streaming=True,
    )

    images = []
    annotations = []
    for i, sample in enumerate(ds):
        if i >= num_images:
            break
        img = sample["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        images.append(img)

        # Get ground truth categories from the sample
        if "objects" in sample and "category" in sample["objects"]:
            cats = sample["objects"]["category"]
            annotations.append(cats)
        else:
            annotations.append([])

        if (i + 1) % 10 == 0:
            print(f"  Loaded {i + 1}/{num_images} images...")

    print(f"  Loaded {len(images)} images total.")
    return images, annotations


def load_coco_from_local(local_dir: str, num_images: int):
    """Load images from a local directory."""
    import os

    images = []
    filenames = []
    for f in sorted(os.listdir(local_dir)):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            img = Image.open(os.path.join(local_dir, f)).convert("RGB")
            images.append(img)
            filenames.append(f)
            if len(images) >= num_images:
                break

    print(f"Loaded {len(images)} images from {local_dir}")
    return images, filenames


def zero_shot_classify(model, processor, image, candidate_labels, top_k=5):
    """Run zero-shot classification on a single image."""
    prompts = [f"a photo of a {label}" for label in candidate_labels]
    inputs = processor(text=prompts, images=image, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits_per_image[0]  # [num_labels]
    probs = logits.softmax(dim=-1)

    top_indices = probs.argsort(descending=True)[:top_k]
    results = [(candidate_labels[i], probs[i].item()) for i in top_indices]
    return results


def main():
    parser = argparse.ArgumentParser(description="CLIP-ViT CPU test with COCO dataset")
    parser.add_argument("--num_images", type=int, default=10, help="Number of images to test")
    parser.add_argument("--local_dir", type=str, default=None, help="Local directory with images (skip download)")
    parser.add_argument("--top_k", type=int, default=5, help="Top-K predictions to show")
    args = parser.parse_args()

    print("=" * 60)
    print("  CLIP-ViT CPU Test (PyTorch reference)")
    print(f"  Model: {MODEL_NAME}")
    print("=" * 60)

    # Load model
    print("\nLoading model...")
    t0 = time.perf_counter()
    model = CLIPModel.from_pretrained(MODEL_NAME)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model.eval()
    print(f"  Model loaded in {time.perf_counter() - t0:.1f}s")

    # Pre-encode text labels (one-time cost)
    print(f"\nPre-encoding {len(COCO_CLASSES)} COCO class prompts...")
    prompts = [f"a photo of a {c}" for c in COCO_CLASSES]
    text_inputs = processor(text=prompts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        text_features = _to_tensor(model.get_text_features(**text_inputs))
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
    print("  Done.")

    # Load images
    print()
    if args.local_dir:
        images, meta = load_coco_from_local(args.local_dir, args.num_images)
        has_annotations = False
    else:
        images, meta = load_coco_from_hub(args.num_images)
        has_annotations = True

    if not images:
        print("No images loaded. Exiting.")
        return

    # Classify each image
    print(f"\nRunning zero-shot classification ({len(COCO_CLASSES)} classes)...")
    print("-" * 60)

    total_time = 0
    for idx, image in enumerate(images):
        t0 = time.perf_counter()

        # Encode image
        img_inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            img_features = _to_tensor(model.get_image_features(**img_inputs))
            img_features = img_features / img_features.norm(p=2, dim=-1, keepdim=True)

        # Compute similarity
        logit_scale = model.logit_scale.exp()
        logits = (img_features @ text_features.T) * logit_scale
        probs = logits.softmax(dim=-1)[0]

        elapsed = time.perf_counter() - t0
        total_time += elapsed

        top_indices = probs.argsort(descending=True)[:args.top_k]

        print(f"\nImage {idx + 1}/{len(images)} ({elapsed*1000:.0f}ms)")

        if has_annotations and meta[idx]:
            gt_ids = meta[idx]
            gt_names = [COCO_CLASSES[i] if i < len(COCO_CLASSES) else f"id:{i}" for i in gt_ids]
            unique_gt = list(dict.fromkeys(gt_names))[:5]
            print(f"  Ground truth: {', '.join(unique_gt)}")

        print(f"  Top-{args.top_k} predictions:")
        for rank, i in enumerate(top_indices):
            label = COCO_CLASSES[i]
            prob = probs[i].item()
            print(f"    {rank+1}. {label:20s} {prob:.4f} ({prob*100:.1f}%)")

    # Summary
    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    print(f"  Images tested: {len(images)}")
    print(f"  Total time:    {total_time:.2f}s")
    print(f"  Avg per image: {total_time/len(images)*1000:.0f}ms")
    print(f"  Model:         {MODEL_NAME}")

if __name__ == "__main__":
    main()
