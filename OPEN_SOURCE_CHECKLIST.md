# FusionRS open-source checklist

## GitHub release

- [x] Dataset card, Datasheet, Croissant metadata, schemas, source notices, and takedown policy
- [x] Rights-cleared public dataset index and split audits
- [x] Dataset construction, training, inference, scoring, and validation code
- [x] Fixed benchmark manifests and annotation provenance
- [x] CLIP/VLM metrics, sanitized raw predictions, paired comparisons, and validation reports
- [x] Model configurations, model cards, exact sizes, and SHA-256 hashes
- [x] Repository-wide checksum manifest
- [ ] Clean-machine reproduction test
- [ ] Create tagged GitHub release after final review

## Hugging Face dataset release

- [x] Public metadata package staged
- [ ] Confirm redistribution rights for every image/text component
- [ ] Upload only rights-cleared data shards
- [ ] Add dataset viewer configuration and split statistics
- [ ] Link immutable dataset revision from GitHub

## Hugging Face model release

- [x] Canonical CLIP weight staged and hashed
- [x] Three final VLM adapters staged and hashed
- [ ] Upload weights, model cards, base-model requirements, and loading examples
- [ ] Verify uploaded file hashes against `models/MODEL_MANIFEST.json`
- [ ] Link immutable model revisions from GitHub

## Benchmark release

- [x] Frozen Caption and synthetic VQA manifests
- [x] Frozen HIT-UAV real-thermal VQA manifest
- [x] Scoring code, public baseline predictions, model predictions, metrics, and audits
- [ ] Complete the remaining synthetic VQA review/balance gate before presenting it as a finalized leaderboard
- [ ] Publish any benchmark images only when their redistribution terms explicitly permit it

