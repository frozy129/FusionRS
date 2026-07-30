# Third-party data and model notices

FusionRS does not apply one blanket license to upstream images, translated
derivatives, captions, code, and model weights. This is a research audit, not
legal advice.

| Asset | Verified evidence | Public treatment now |
|---|---|---|
| RS5M | The repository MIT license covers software and associated documentation; RS5M aggregates heterogeneous upstream image-text sources | Identifiers and reconstruction metadata only |
| SkyScript | Repository code is MIT; imagery follows ten Earth Engine collection terms; caption content is derived from OpenStreetMap tags under ODbL | Identifiers and reconstruction metadata only until row-level notices and ODbL treatment are implemented |
| NWPU-Captions | Base RESISC45 images are CC BY-NC 4.0; the NWPU-Captions repository has no explicit caption license | Identifiers and reconstruction metadata only |
| RSICD | No explicit redistribution or derivative-work license located; README lists third-party imagery providers | Identifiers and reconstruction metadata only |
| RSITMD | Public access/download statement but no explicit redistribution or derivative-work license located | Identifiers and reconstruction metadata only |
| DiffV2IR code | Apache-2.0 in the checked source tree | Link and attribution; users obtain code/checkpoint |
| OpenAI CLIP code | MIT | Follow upstream code and weight terms |

Infrared-style outputs are derivatives of upstream RGB images. Translation
does not remove upstream restrictions. No upstream image byte or translated
image byte may enter the public release unless the corresponding source and
derivative rights are confirmed.

Source-caption strings must also be removed where text redistribution is not
established. Because the generated IR-aware annotations were conditioned on
both source imagery and source captions and retain substantial source
vocabulary, they remain blocked pending source-specific clearance or a
clean-room rewrite. Hashes used for internal split construction are provenance
metadata, not a substitute for distribution permission.

The detailed research record and links are in `licenses/RIGHTS_AUDIT_20260727.md`;
the machine-readable status is in `licenses/sources.json`.
