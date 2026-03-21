"""Run a single vision + text encoder pass for Tracy profiling."""
import torch
import ttnn
from clip_vit_ttnn.tt.weight_loader import load_all_weights, CLIPTTNNConfig
from clip_vit_ttnn.tt.clip_model import run_vision_encoder, run_text_encoder
from transformers import CLIPModel, CLIPProcessor

dispatch_config = ttnn.DispatchCoreConfig(type=ttnn.DispatchCoreType.WORKER)
device = ttnn.open_device(device_id=0, dispatch_core_config=dispatch_config)
device.enable_program_cache()

config = CLIPTTNNConfig(stage=2)
hf_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
params = load_all_weights(hf_model, device, config)

pixel_values = torch.randn(1, 3, 224, 224)
text_inputs = processor(text=["a photo of a cat"], return_tensors="pt", padding=True)

# Warmup (compile + cache)
for _ in range(3):
    tt_v, _ = run_vision_encoder(pixel_values, params["vision"], config, device)
    ttnn.deallocate(tt_v)
    tt_t, _ = run_text_encoder(
        text_inputs["input_ids"], text_inputs["attention_mask"],
        params["text"], config, device,
    )
    ttnn.deallocate(tt_t)
ttnn.synchronize_device(device)

# Profiled run
tt_v, _ = run_vision_encoder(pixel_values, params["vision"], config, device)
ttnn.synchronize_device(device)
ttnn.deallocate(tt_v)

tt_t, _ = run_text_encoder(
    text_inputs["input_ids"], text_inputs["attention_mask"],
    params["text"], config, device,
)
ttnn.synchronize_device(device)
ttnn.deallocate(tt_t)

ttnn.close_device(device)
print("Tracy profiling complete")
