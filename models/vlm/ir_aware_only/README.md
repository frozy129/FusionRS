---
base_model: Qwen/Qwen2.5-VL-7B-Instruct
library_name: peft
pipeline_tag: image-text-to-text
tags:
- remote-sensing
- infrared
- vision-language
- lora
---

# FusionRS VLM — IR-aware-only

LoRA adapter trained on FusionRS IR-aware captions, which describe observable grayscale intensity, contrast, boundary, texture, and spatial cues.

- Base model: `Qwen/Qwen2.5-VL-7B-Instruct`
- Adapter type: LoRA, rank 16, alpha 32
- Weight SHA-256: `d41a44445c6a11d1fc2437c1119ad5d46b03ec3da57c84c131f61baf49a7f693`
- Evaluation: see `results/caption_vqa/` and `results/real_thermal_vqa/`

