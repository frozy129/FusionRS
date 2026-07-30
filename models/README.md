# Models

Large weights are stored outside GitHub and verified against [`MODEL_MANIFEST.json`](MODEL_MANIFEST.json).

## Released artifacts

- `clip/fusionrs_rc4_trimodal_580k_seed42`: FusionRS CLIP model exported as `model.safetensors`.
- `vlm/original_only`: Qwen2.5-VL-7B LoRA trained with conventional captions.
- `vlm/ir_aware_only`: Qwen2.5-VL-7B LoRA trained with IR-aware captions.
- `vlm/task_conditioned_mixed`: Qwen2.5-VL-7B LoRA trained with task-conditioned mixed supervision.

The VLM files are adapters and require `Qwen/Qwen2.5-VL-7B-Instruct` plus a compatible `transformers`/`peft` environment. Only final seed-42 adapters are listed. Interrupted or incomplete checkpoints are not release artifacts.

Example:

```python
from peft import PeftModel
from transformers import Qwen2_5_VLForConditionalGeneration

base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    torch_dtype="auto",
    device_map="auto",
)
model = PeftModel.from_pretrained(base, "/path/to/adapter")
```

