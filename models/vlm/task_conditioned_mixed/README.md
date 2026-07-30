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

# FusionRS VLM — Task-conditioned mixed

LoRA adapter trained with task-conditioned conventional and IR-aware caption supervision. This is the primary generative FusionRS adapter.

- Base model: `Qwen/Qwen2.5-VL-7B-Instruct`
- Adapter type: LoRA, rank 16, alpha 32
- Weight SHA-256: `b738d3558f3d9afd1d4a172327df30d64dcaf088d028cbd790db5a1536e52231`
- Evaluation: see `results/caption_vqa/` and `results/real_thermal_vqa/`

