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

# FusionRS VLM — Original-only

LoRA adapter trained on FusionRS conventional scene captions. It is released as the original-caption control for the Caption/VQA experiments.

- Base model: `Qwen/Qwen2.5-VL-7B-Instruct`
- Adapter type: LoRA, rank 16, alpha 32
- Weight SHA-256: `a5d2bb53c66813033dd36b75a0cc7ec347c0077d2d563c884d090c669b22f8a9`
- Evaluation: see `results/caption_vqa/` and `results/real_thermal_vqa/`

