# Datasheet for FusionRS

## Document status

This document describes the public metadata release corresponding to the
600,000-record canonical `rc4` index. Large assets are distributed separately
after source-specific rights verification.

## Motivation

FusionRS was created to study RGB--infrared-style--text representation learning
in remote sensing at a scale not available from paired sensor-captured
RGB--thermal corpora. It supports controlled cross-view retrieval, data-scale
and supervision ablations, transfer to separately identified sensor-captured
benchmarks, and an infrared-style Caption/VQA protocol. Infrared-style images
are translated from RGB and are not physical temperature, emissivity, or
material measurements.

## Composition

The canonical index contains 600,000 records:

| Source | Total | Train | Validation | Test |
|---|---:|---:|---:|---:|
| RS5M | 488,033 | 472,682 | 7,670 | 7,681 |
| SkyScript | 65,266 | 64,150 | 558 | 558 |
| NWPU-Captions | 31,186 | 30,094 | 556 | 536 |
| RSICD | 10,824 | 9,383 | 716 | 725 |
| RSITMD | 4,691 | 3,691 | 500 | 500 |
| **Total** | **600,000** | **580,000** | **10,000** | **10,000** |

Each record identifies an upstream RGB item, its translated infrared-style
output key, split and duplicate-component membership, and available text
supervision. There are 599,992 usable source captions across all splits and
45,913 retained IR-aware captions in the training partition. The latter were
machine generated and then filtered; they are not human-authored ground truth.

## Collection and construction

Source records were indexed from RS5M, SkyScript, NWPU-Captions, RSICD, and
RSITMD. RGB views were translated using the recorded DiffV2IR configuration.
IR-aware captions were generated with a vision-language model conditioned on
the translated infrared-style view and source semantics. Accepted annotations
were filtered and versioned with their generation provenance. Public text
distribution follows the source-specific rights records.

## Preprocessing and quality control

Captions are normalized using Unicode NFKC, case folding, punctuation removal,
and whitespace collapsing for grouping. RGB and infrared-style perceptual-hash
components and normalized-caption equality are joined before split assignment.
No connected group may cross partitions. Evaluation uses one representative
per held-out group, giving 9,538 validation and 9,553 test representatives.

IR-aware captions pass format, observable-IR-cue, lexical consistency, and
physical-overclaim filters. The released audits document the retained records
and benchmark annotation provenance.

## Distribution and licensing

FusionRS uses an index-only release. It does not redistribute upstream RGB
images or translated infrared-style image files. Users obtain each source
through its official route and materialize indexed views locally under the
applicable terms. Translated derivatives are not redistributed where derivative
rights are unconfirmed.

There is no blanket FusionRS license covering all upstream images, captions,
translated derivatives, generated annotations, code, and model weights.
`THIRD_PARTY_NOTICES.md` and `licenses/sources.json` record the available
source-specific evidence and release treatment. Source-caption text must be
removed from the public manifest for any source whose text-redistribution right
is not established, leaving identifiers and reconstruction metadata instead.
The root software license does not override upstream data terms.

## Intended uses

Intended uses include research on remote-sensing representation learning,
RGB--infrared-style alignment, retrieval, controlled synthetic-to-real
evaluation, and infrared-style language grounding. Sensor-captured benchmarks
must be clearly distinguished from translated infrared-style data.

The dataset is not intended for calibrated thermal measurement, individual
identification, persistent person or vehicle tracking, operational target
localization or selection, autonomous surveillance, weapon-related decision
making, or other safety-critical uses.

## Limitations and risks

FusionRS cannot recover thermal physics absent from RGB. Its source distribution
is imbalanced toward RS5M. A 200-vs.-200 image-statistic audit against HIT-UAV
found similar mean brightness but substantial differences in contrast, dynamic
range, entropy, edge strength, and Laplacian variance. Machine-generated
captions may contain residual semantic or linguistic errors. Remote-sensing and
UAV imagery also present surveillance and military dual-use risks.

## Maintenance, corrections, and takedown

Releases use versioned manifests and file checksums. Rights, privacy, corrupted
record, duplicate-leakage, or unsafe-content reports are handled under
`TAKEDOWN_POLICY.md`; verified changes are recorded in versioned release notes.
Reports can be submitted through the public repository issue tracker.

## Reproducibility artifacts

The package contains the public canonical index, source-rights metadata, JSON
schemas, fixed group-aware split configuration, generation configurations,
reconstruction scripts, checksums, machine-readable Croissant metadata, frozen
Caption/VQA manifests, and released scorers. Synthetic VQA keeps its
AI-assisted provenance and is versioned separately from the validated
real-thermal HIT-UAV VQA suite.
