# FusionRS source-rights audit — 2026-07-27

Status: research audit, not legal advice. Public release remains blocked until
the authors complete the actions below.

## Decision

The safe current release is **identifiers, split/group metadata, reconstruction
scripts, configurations, and checksums only**. Do not publish upstream RGB
bytes, translated infrared-style bytes, uncleared source-caption strings, or
IR-aware captions that substantially retain uncleared source text.

| Source | Finding | Current public decision |
|---|---|---|
| RS5M | The top-level MIT text licenses software and associated documentation, not the heterogeneous upstream LAION/COYO/remote-sensing items | Block image, translated image, source caption, and derived caption text |
| SkyScript | The ten imagery collections generally permit reuse subject to collection-specific attribution/change notices; captions originate from OpenStreetMap tags under ODbL | Conditional in principle, but block until every row has a source code and an automatic attribution/ODbL notice |
| NWPU-Captions | Base RESISC45 imagery is CC BY-NC 4.0; the later caption repository has no explicit license | Base-image derivatives only under noncommercial attribution/change terms; block caption and derived-caption text |
| RSICD | No explicit redistribution/derivative license located; README identifies Google Earth, Baidu Map, MapABC, and Tianditu imagery | Block image and text redistribution |
| RSITMD | Access/download language is not an explicit redistribution/derivative license | Block image and text redistribution |

## Why the repository licenses are insufficient

- RS5M and SkyScript use MIT language for “software and associated
  documentation.” That wording does not prove ownership of, or grant blanket
  rights to, every collected image and caption.
- RS5M contains heterogeneous upstream material. FusionRS has 488,033 RS5M
  records, dominated by LAION/COYO-derived subsets, so item-level provenance is
  necessary before any bytes or captions are redistributed.
- SkyScript has 65,266 FusionRS records across ten Earth Engine collections.
  The imagery terms are source-specific, and the OpenStreetMap-derived caption
  database adds ODbL attribution/share-alike obligations.
- NWPU in this project is **NWPU-Captions**, not merely the class-label
  RESISC45 release. The image license cannot be assumed to license the added
  sentence annotations.

## Public-manifest rule

Allowed now:

- `sample_id`, `source_dataset`, and official source record/locator;
- fixed split and opaque image/group identifiers;
- reconstruction configuration, model identifier, and output naming;
- checksums, counts, schemas, scripts, Datasheet, Croissant metadata, and
  source-specific rights codes.

Remove until cleared:

- `caption` and any other upstream caption string;
- upstream or translated image bytes;
- IR-aware caption strings for blocked sources unless counsel/rights holders
  confirm release or the text is independently rewritten and checked;
- any field whose value embeds machine-local paths, credentials, or private
  storage locations.

`normalized_caption_key` may be retained for internal leakage auditing but
should not be advertised as a released caption substitute; an opaque
release-specific group ID is preferable.

## Source-specific evidence

- RS5M repository and license:
  <https://github.com/om-ai-lab/RS5M>,
  <https://raw.githubusercontent.com/om-ai-lab/RS5M/main/LICENSE>
- SkyScript repository, imagery map, and license:
  <https://github.com/wangzhecheng/SkyScript>,
  <https://raw.githubusercontent.com/wangzhecheng/SkyScript/main/image_sources.py>,
  <https://raw.githubusercontent.com/wangzhecheng/SkyScript/main/LICENSE>
- NWPU-Captions and RESISC45 terms:
  <https://github.com/HaiyanHuang98/NWPU-Captions>,
  <https://gcheng-nwpu.github.io/>
- RSICD and RSITMD:
  <https://github.com/201528014227051/RSICD_optimal>,
  <https://github.com/xiaoyuan1996/AMFMN/tree/master/RSITMD>
- Creative Commons BY-NC 4.0:
  <https://creativecommons.org/licenses/by-nc/4.0/legalcode.en>
- OpenStreetMap/ODbL:
  <https://www.openstreetmap.org/copyright>,
  <https://opendatacommons.org/licenses/odbl/1-0/>
- Earth Engine source terms:
  <https://developers.google.com/earth-engine/datasets/catalog/USDA_NAIP_DOQQ>,
  <https://developers.google.com/earth-engine/datasets/catalog/Germany_Brandenburg_orthos_20cm>,
  <https://developers.google.com/earth-engine/datasets/catalog/Finland_SMK_V_50cm>,
  <https://developers.google.com/earth-engine/datasets/catalog/Switzerland_SWISSIMAGE_orthos_10cm>,
  <https://developers.google.com/earth-engine/datasets/catalog/Spain_PNOA_PNOA10>,
  <https://developers.google.com/earth-engine/datasets/catalog/SKYSAT_GEN-A_PUBLIC_ORTHO_RGB>,
  <https://developers.google.com/earth-engine/datasets/catalog/SKYSAT_GEN-A_PUBLIC_ORTHO_MULTISPECTRAL>,
  <https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED>,
  <https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C02_T1_TOA>,
  <https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC09_C02_T1_TOA>.

## Release gates

1. Ask the RS5M, NWPU-Captions, RSICD, and RSITMD maintainers for explicit
   written permission covering caption redistribution and generated
   annotations.
2. Add a row-level `rights_code` and upstream-subsource field.
3. Generate all attribution, change, noncommercial, and ODbL notices from that
   field.
4. Build a public redacted manifest and verify that blocked text/image fields
   are absent.
5. Keep the full 600K internal training manifest private until these gates
   pass.
