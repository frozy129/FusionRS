# Public upload allowlist — 2026-07-27

Do **not** upload this working directory wholesale.

## Upload now

- `fusionrs_public_index_v3_rc4_20260727.jsonl.gz` and its summary;
- source code under `scripts/`, except private credentials or local launchers;
- schemas, configs, fixed split metadata, checksums, Datasheet, Croissant
  metadata, benchmark protocol, rights audit, third-party notices, takedown
  policy, and source links;
- model/checkpoint hashes and evaluation code that contain no third-party bytes.

## Do not upload now

- `fusionrs_dataset_index_v3_rc4_20260726.jsonl.gz` because it contains source
  captions;
- `manifests/iraware_strict_v3_45913.jsonl.gz` and the removed-caption
  manifests until the annotation-rights gate passes;
- upstream RGB files, translated infrared-style files, copied benchmark images,
  model credentials, OpenRouter logs containing secrets, or machine-local
  paths;
- `benchmark_eval_ready_20260727` as a final benchmark until human correction,
  negative-question balancing, and frozen-scoring gates pass.

## Final check

Run:

```bash
python scripts/verify_public_redacted_index.py \
  --index fusionrs_public_index_v3_rc4_20260727.jsonl.gz
```

Blank human-review fields are not approvals.
