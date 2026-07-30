# Open-source release practices for related remote-sensing datasets and benchmarks

Official repositories reviewed:

| Project | Public release pattern |
|---|---|
| [RS5M / GeoRSCLIP](https://github.com/om-ai-lab/RS5M) | Dataset download, label-only option, dataloader, inference/evaluation code, model checkpoints, and license |
| [RemoteCLIP](https://github.com/ChenDelong1999/RemoteCLIP) | Model checkpoints, loading/demo notebooks, retrieval code, dataset links, and environment instructions |
| [GeoChat](https://github.com/mbzuai-oryx/GeoChat) | Code, models, dataset, training instructions, evaluation scripts, and model zoo |
| [VRSBench](https://github.com/lx709/VRSBench) | Dataset/train/eval JSON, annotation-generation code and prompts, baselines, models, scoring code, and metrics |
| [TEOChat](https://github.com/ermongroup/TEOChat) | Dataset, model, training/validation, inference, demo, and license |

The common release pattern is:

1. GitHub stores code, documentation, environment files, small manifests, fixed splits, prompts, scorers, examples, and checksums.
2. Hugging Face or another dataset host stores large dataset shards and model weights.
3. A benchmark release includes the exact task schema, fixed test manifest, annotation provenance, evaluation script, baseline commands, row-level outputs, aggregate metrics, and version/hash metadata.
4. Dataset cards document composition, source licenses, preprocessing, intended use, limitations, and takedown/contact procedures.
5. Model cards document the base model, training objective, loading instructions, checkpoint hash, evaluation protocol, and license dependencies.

FusionRS follows this pattern while applying a stricter redistribution gate: only rights-cleared metadata and artifacts are published, and third-party image bytes remain under their original source terms.

