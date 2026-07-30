# FusionRS dataset card

## Description

FusionRS contains 600,000 aligned RGB, infrared-style, and text records assembled from five public remote-sensing sources. The infrared-style view is translated from RGB with DiffV2IR to provide paired cross-modal supervision at scale.

## Canonical split

| Split | Records | Unique evaluation representatives |
|---|---:|---:|
| Train | 580,000 | — |
| Validation | 10,000 | 9,538 |
| Test | 10,000 | 9,553 |

All related RGB images, infrared-style images, and caption-equivalent records are joined before split assignment. Each connected component belongs wholly to one split. Eight empty or unusable training captions are excluded from CLIP training, leaving 579,992 usable triplets.

## Text supervision

Every usable record has a conventional scene caption. A filtered set of 45,913 training records also carries IR-aware text describing observable grayscale intensity, contrast, boundaries, texture, and spatial structure. The filtering configuration and audit metadata are included in this release.

## Tasks

- RGB↔text and infrared-style↔text retrieval;
- RGB↔infrared-style paired-view retrieval;
- dual-modal representation pretraining;
- infrared-style Caption and VQA;
- transfer to real sensor-captured thermal benchmarks.

The fixed retrieval split, positive definitions, aggregation rules, and scorers are documented in [`../benchmark/`](../benchmark/).

## Public distribution

The current GitHub package contains the rights-cleared public index, source locators, split/group metadata, schemas, configurations, checksums, and reconstruction tools. Users obtain upstream sources under their original terms. Source-specific notices are recorded in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) and [`licenses/`](licenses/).

## Scope

Infrared-style images are translated views aligned with RGB scene content; they are intended for representation learning and are not calibrated temperature or emissivity measurements. Real sensor-captured thermal benchmarks are therefore reported separately. FusionRS is intended for remote-sensing research and not for safety-critical sensing or operational targeting.

