# HIT-UAV Object Presence

- samples: 2898
- positive samples: 2866
- all-negative samples: 32
- classes: person, bicycle, car, other vehicle

| model | checkpoint | mAP | Top1 presence hit |
|---|---:|---:|---:|
| openai_clip_vit_b32_0k | pretrained-base | 36.4693 | 45.8937 |
| dual_text_580k_seed42 | latest.pt | 39.8828 | 31.7460 |
| georsclip_vit_b32_0k | latest.pt | 34.3633 | 37.7847 |
| grayscale_580k_seed42 | latest.pt | 39.2365 | 32.9538 |
| remoteclip_vit_b32_0k | latest.pt | 38.7128 | 44.1684 |
| rgb_duplicate_580k_seed42 | latest.pt | 37.0539 | 31.1249 |
| rgb_only_580k_seed42 | latest.pt | 37.0145 | 32.6432 |
| trimodal_580k_seed2026 | latest.pt | 41.1147 | 42.0980 |
| trimodal_580k_seed3407 | latest.pt | 40.7715 | 36.5769 |
| trimodal_580k_seed42 | latest.pt | 40.2058 | 29.2961 |
