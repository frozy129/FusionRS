# RC4 public verification record

Verification date: 2026-10-03 (Asia/Shanghai)

## Public revisions

- Main repository verification tag: `v1.0.0-rc4-main` (immutable commit recorded below).
- Model release: [`FusionRS-CLIP v1.0.0-rc4`](https://github.com/frozy129/FusionRS-CLIP/releases/tag/v1.0.0-rc4), commit `f776df3472753dddbb3a53379504cbf08beb62b1`.

## Reproducibility checks

The public redacted-index validator passes on the RC4 index:

```text
rows: 600000
status: pass
blocked fields absent: caption, class_name, image_bytes, ir_path,
  normalized_caption_key, rgb_path
```

The model release was verified from the RC4 package with:

```text
release verifier: status=ok, files_verified=37
CPU image shape: (2, 512)
CPU text shape: (1, 512)
unit smoke tests: 2/2 passed
```

The public model release exposes the portable CLIP weights, the source-aware 600K index, the 45,913-record IR-aware caption manifest, reconstruction metadata, and SHA-256 records. The caption manifest was anonymously downloaded through Git LFS media storage, decompressed successfully, and contained 45,913 rows. The model media endpoint returned the expected 605,156,676-byte artifact.

## Data and claim boundaries

The upstream RGB and generated infrared-style image bytes are not copied into this repository because the five source datasets have source-specific redistribution terms. Users obtain source data under those terms and use the documented reconstruction path.

The synthetic VQA file remains a candidate benchmark pending a balance and final-review gate. The current release does not claim that the generated captions or synthetic VQA labels are human-verified ground truth. Real-sensor transfer results remain task- and sensor-dependent.
