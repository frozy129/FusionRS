# Code

- `dataset/`: build, filter, reconstruct, audit, and validate dataset metadata.
- `clip/`: train FusionRS CLIP variants and evaluate retrieval/downstream tasks.
- `vlm/`: build caption-training manifests and fine-tune Qwen2.5-VL adapters.
- `benchmark/`: run Caption/VQA inference, score predictions, compare systems, and validate complete suites.
- `inference/`: source mirror for lightweight FusionRS CLIP loading and inference.
- `tools/`: release sanitization and manifest generation.

All public commands accept data/model/output paths through command-line arguments or environment variables. Machine-specific launch scripts, credentials, cached models, and scheduler logs are excluded.

The installable `fusionrs` package is also exposed at the repository root:

```bash
python -m pip install -e .
fusionrs-infer --help
```
