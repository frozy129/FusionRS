# Hugging Face upload plan

## Dataset repository

Upload from `FusionRS-hub-assets/dataset/`:

- rights-cleared public 600K index and summary;
- Dataset Card, Datasheet, Croissant metadata, schemas, split/configuration files;
- source notices, rights records, checksums, and reconstruction code.

Do not upload upstream RGB bytes, generated infrared-style bytes, source captions without confirmed redistribution rights, or machine-local paths.

## Model repository

Upload from `FusionRS-hub-assets/models/`:

- canonical FusionRS CLIP `model.safetensors` and processor/tokenizer configuration;
- final `original_only`, `ir_aware_only`, and `task_conditioned_mixed` VLM adapters;
- per-model cards and `MODEL_MANIFEST.json`.

After upload, download each artifact once and verify its SHA-256 against the manifest before linking it from GitHub.

## Benchmark repository

Upload from `FusionRS-hub-assets/benchmark/`:

- fixed JSONL manifests;
- accepted Caption/VQA CSV exports;
- annotation provenance and benchmark card;
- scoring scripts, sanitized predictions, metrics, and checksums.

Images are linked by relative locators and remain subject to source-specific distribution terms.

