# FusionRS

Official repository for **FusionRS: A Large-Scale RGB–Infrared Remote Sensing Dataset for Dual-Modal Vision–Language Foundation Models**.

FusionRS provides resources for RGB–infrared-style remote-sensing vision-language learning. The release is organized into five independently usable parts:

| Part | Contents |
|---|---|
| [`dataset/`](dataset/) | Dataset card, Datasheet, Croissant metadata, public index, schemas, generation configuration, rights records, and reconstruction tools |
| [`code/`](code/) | Dataset construction, CLIP training/evaluation, VLM training, inference, benchmark scoring, and validation |
| [`results/`](results/) | Retrieval, ablation, downstream transfer, caption/VQA, real-thermal VQA, paired tests, and validation artifacts |
| [`models/`](models/) | Model cards, loading metadata, configurations, exact file sizes, and SHA-256 hashes |
| [`benchmark/`](benchmark/) | Fixed caption/VQA manifests, real-thermal VQA manifest, annotation provenance, protocols, and scorers |

## Release layout

GitHub hosts source code, documentation, compact manifests, and auditable results. Large dataset assets and model weights are prepared for separate Hugging Face dataset/model repositories. This avoids duplicating large binaries in Git history and keeps versioning explicit.

The current public dataset package is the rights-cleared metadata release: public identifiers, split/group metadata, reconstruction configuration, checksums, and documentation. Third-party source imagery and generated image bytes are not redistributed through this repository. See [`dataset/PUBLIC_UPLOAD_ALLOWLIST_20260727.md`](dataset/PUBLIC_UPLOAD_ALLOWLIST_20260727.md).

## Quick start

```bash
git clone https://github.com/frozy129/FusionRS.git
cd FusionRS
python -m pip install -r requirements.txt
python code/dataset/scripts/verify_public_redacted_index.py \
  --index dataset/manifests/fusionrs_public_index_v3_rc4_20260727.jsonl.gz
```

Benchmark scoring examples and required model-specific commands are documented in [`benchmark/README.md`](benchmark/README.md). Exact released-weight hashes are listed in [`models/MODEL_MANIFEST.json`](models/MODEL_MANIFEST.json).
The release design is compared with related official repositories in [`docs/OPEN_SOURCE_BENCHMARK_PRACTICES.md`](docs/OPEN_SOURCE_BENCHMARK_PRACTICES.md).

## Reproducibility

- Canonical data split: `rc4`, with RGB, infrared-style, and caption-linked records assigned by joint group.
- Caption benchmark: 998 image-task records over 499 infrared-style images.
- Synthetic VQA candidate: 1,993 records over 500 images; AI-assisted provenance is retained in every release.
- Real-thermal VQA: 3,223 questions over 579 HIT-UAV images, derived from official bounding-box annotations.
- Public predictions contain record IDs, prompts, references/answers, predictions, and status, with machine-local paths removed.

## License and citation

Code is released under the repository [`LICENSE`](LICENSE). Dataset components remain subject to their source-specific terms; see [`dataset/THIRD_PARTY_NOTICES.md`](dataset/THIRD_PARTY_NOTICES.md) and [`dataset/licenses/`](dataset/licenses/).

Please cite the paper using [`CITATION.cff`](CITATION.cff).
