# FusionRS benchmark

The benchmark package separates fixed data records from model outputs:

- `manifests/caption_eval_manifest.jsonl`: 998 Caption records over 499 infrared-style images, with up to three AI-assisted references from GPT-5.6 Sol, Terra, and Luna.
- `manifests/vqa_eval_manifest.jsonl`: 1,993 synthetic infrared-style VQA records over 500 images.
- `manifests/hit_uav_vqa_eval_manifest.jsonl`: 3,223 real-thermal VQA records over 579 HIT-UAV images, derived from official bounding-box labels.
- `annotations/accepted_caption_references.csv`: all 2,956 accepted Caption reference rows.
- `annotations/synthetic_vqa_candidate.csv`: all 1,993 accepted synthetic VQA rows with IR-observability fields.
- `annotations/`: annotation provenance and accepted CSV exports.
- `protocol/`: frozen task definitions, denominators, positives, and metrics.

All questions in the paired RGB/IR annotation workflow are required to be answerable from the infrared view. RGB is contextual evidence for annotators, not a required input at benchmark inference time.

## Evaluation

```bash
python code/benchmark/run_fusionrs_benchmark_qwen.py \
  --caption-manifest benchmark/manifests/caption_eval_manifest.jsonl \
  --vqa-manifest benchmark/manifests/vqa_eval_manifest.jsonl \
  --image-root /path/to/benchmark-assets \
  --model-path Qwen/Qwen2.5-VL-7B-Instruct \
  --adapter-path /path/to/adapter \
  --model-tag my_model \
  --output predictions/my_model.jsonl

python code/benchmark/score_fusionrs_benchmark.py \
  --input predictions/my_model.jsonl \
  --output results/my_model.metrics.json
```

The synthetic Caption/VQA package records its AI-assisted provenance. The synthetic VQA portion remains a candidate benchmark until its final review and balance gate is completed; it should not yet be presented as a finalized public leaderboard. The real-thermal HIT-UAV VQA suite is separately frozen and validated.

Benchmark images are not redistributed in this GitHub repository. Obtain source data under the original dataset terms and resolve the relative locators in the manifests.
