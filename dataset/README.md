# Dataset

This directory contains the public, rights-cleared metadata release of FusionRS.

## Included

- `manifests/fusionrs_public_index_v3_rc4_20260727.jsonl.gz`: redacted canonical index;
- `configs/`: generation, filtering, and split configurations;
- `schemas/`: machine-readable record schemas;
- `audit/`: public-index and joint-split verification;
- `DATASET_CARD.md`, `DATASHEET.md`, and `croissant.json`;
- `licenses/`, `THIRD_PARTY_NOTICES.md`, and `TAKEDOWN_POLICY.md`.

The public index intentionally contains stable identifiers, source locators, split/group metadata, and hashes rather than machine-local paths or copied third-party bytes. Reconstruction and verification utilities are in [`../code/dataset/scripts/`](../code/dataset/scripts/).

The internal full index, source captions pending redistribution review, upstream RGB files, generated infrared-style files, and copied benchmark images are not part of the current GitHub release. The authoritative allowlist is [`PUBLIC_UPLOAD_ALLOWLIST_20260727.md`](PUBLIC_UPLOAD_ALLOWLIST_20260727.md).
