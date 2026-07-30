# Experimental results

This directory preserves the compact evidence behind the paper tables.

- `clip/retrieval/`: validation checkpoint selection and final test retrieval, including per-source results.
- `clip/ablations/`: backbone/capacity comparisons.
- `clip/downstream/`: HIT-UAV, SIRST-V2, IRSTD-1K, and paired RGB–thermal transfer results.
- `caption_vqa/`: model metrics, audits, paired comparisons, and sanitized row-level predictions.
- `real_thermal_vqa/`: HIT-UAV real sensor-captured thermal VQA metrics and predictions.
- `validation/`: frozen-suite validation and canonical split hashes.

Prediction files retain benchmark IDs, questions/prompts, references or accepted answers, model tags, predictions, and execution status. Absolute server paths and adapter locations have been removed. Metrics should be interpreted with the protocols and frozen denominators in [`../benchmark/`](../benchmark/).

