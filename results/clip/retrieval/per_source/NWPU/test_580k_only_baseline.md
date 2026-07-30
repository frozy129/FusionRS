# 580k-only CLIP baseline on clean held-out test

- test_pair_jsonl: `<machine-local-path-removed>`
- samples: 536
- baseline: `NWPU_joint_rgb_ir_phash4_caption_disjoint`

| model | checkpoint | Mean Recall | direction | R@1 | R@5 | R@10 |
|---|---:|---:|---:|---:|---:|---:|
| base_0k | latest.pt | 10.7743 | IR->original_caption | 2.0522 | 10.2612 | 15.6716 |
| base_0k | latest.pt | 10.7743 | original_caption->IR | 2.2388 | 10.2612 | 16.9776 |
| base_0k | latest.pt | 10.7743 | RGB->IR | 5.0373 | 16.7910 | 22.7612 |
| base_0k | latest.pt | 10.7743 | IR->RGB | 3.1716 | 8.7687 | 15.2985 |
| dual_text_580k_seed42 | latest.pt | 56.0479 | IR->original_caption | 22.2015 | 58.3955 | 73.1343 |
| dual_text_580k_seed42 | latest.pt | 56.0479 | original_caption->IR | 18.6567 | 50.7463 | 71.2687 |
| dual_text_580k_seed42 | latest.pt | 56.0479 | RGB->IR | 42.7239 | 70.3358 | 79.2910 |
| dual_text_580k_seed42 | latest.pt | 56.0479 | IR->RGB | 40.8582 | 66.2313 | 78.7313 |
| georsclip_vit_b32_0k | latest.pt | 20.4291 | IR->original_caption | 3.7313 | 16.9776 | 26.6791 |
| georsclip_vit_b32_0k | latest.pt | 20.4291 | original_caption->IR | 5.5970 | 21.8284 | 35.8209 |
| georsclip_vit_b32_0k | latest.pt | 20.4291 | RGB->IR | 11.7537 | 30.5970 | 44.0299 |
| georsclip_vit_b32_0k | latest.pt | 20.4291 | IR->RGB | 6.5299 | 16.7910 | 24.8134 |
| grayscale_580k_seed42 | latest.pt | 26.6014 | IR->original_caption | 6.7164 | 24.8134 | 34.8881 |
| grayscale_580k_seed42 | latest.pt | 26.6014 | original_caption->IR | 6.9030 | 23.6940 | 38.9925 |
| grayscale_580k_seed42 | latest.pt | 26.6014 | RGB->IR | 18.6567 | 36.5672 | 47.0149 |
| grayscale_580k_seed42 | latest.pt | 26.6014 | IR->RGB | 14.7388 | 29.6642 | 36.5672 |
| remoteclip_vit_b32_0k | latest.pt | 11.9714 | IR->original_caption | 2.4254 | 8.5821 | 14.9254 |
| remoteclip_vit_b32_0k | latest.pt | 11.9714 | original_caption->IR | 2.7985 | 10.0746 | 14.7388 |
| remoteclip_vit_b32_0k | latest.pt | 11.9714 | RGB->IR | 8.5821 | 22.5746 | 30.7836 |
| remoteclip_vit_b32_0k | latest.pt | 11.9714 | IR->RGB | 2.2388 | 10.0746 | 15.8582 |
| rgb_duplicate_580k_seed42 | latest.pt | 18.9832 | IR->original_caption | 4.2910 | 16.9776 | 26.8657 |
| rgb_duplicate_580k_seed42 | latest.pt | 18.9832 | original_caption->IR | 5.0373 | 22.2015 | 34.3284 |
| rgb_duplicate_580k_seed42 | latest.pt | 18.9832 | RGB->IR | 11.0075 | 29.6642 | 39.1791 |
| rgb_duplicate_580k_seed42 | latest.pt | 18.9832 | IR->RGB | 5.4104 | 13.0597 | 19.7761 |
| rgb_only_580k_seed42 | latest.pt | 18.5012 | IR->original_caption | 4.6642 | 17.9104 | 25.5597 |
| rgb_only_580k_seed42 | latest.pt | 18.5012 | original_caption->IR | 4.1045 | 21.8284 | 34.5149 |
| rgb_only_580k_seed42 | latest.pt | 18.5012 | RGB->IR | 9.1418 | 28.1716 | 36.7537 |
| rgb_only_580k_seed42 | latest.pt | 18.5012 | IR->RGB | 6.3433 | 13.6194 | 19.4030 |
| trimodal_100k_seed42 | latest.pt | 64.2102 | IR->original_caption | 16.6045 | 44.7761 | 61.1940 |
| trimodal_100k_seed42 | latest.pt | 64.2102 | original_caption->IR | 12.5000 | 39.1791 | 55.4104 |
| trimodal_100k_seed42 | latest.pt | 64.2102 | RGB->IR | 77.2388 | 93.0970 | 97.5746 |
| trimodal_100k_seed42 | latest.pt | 64.2102 | IR->RGB | 80.7836 | 94.9627 | 97.2015 |
| trimodal_300k_seed42 | latest.pt | 68.5323 | IR->original_caption | 18.4701 | 53.9179 | 67.5373 |
| trimodal_300k_seed42 | latest.pt | 68.5323 | original_caption->IR | 15.2985 | 45.1493 | 62.6866 |
| trimodal_300k_seed42 | latest.pt | 68.5323 | RGB->IR | 81.1567 | 96.0821 | 99.0672 |
| trimodal_300k_seed42 | latest.pt | 68.5323 | IR->RGB | 87.1269 | 97.3881 | 98.5075 |
| trimodal_50k_seed42 | latest.pt | 60.0902 | IR->original_caption | 13.2463 | 37.1269 | 57.0896 |
| trimodal_50k_seed42 | latest.pt | 60.0902 | original_caption->IR | 10.0746 | 33.7687 | 50.0000 |
| trimodal_50k_seed42 | latest.pt | 60.0902 | RGB->IR | 73.1343 | 90.6716 | 95.1493 |
| trimodal_50k_seed42 | latest.pt | 60.0902 | IR->RGB | 75.0000 | 90.8582 | 94.9627 |
| trimodal_580k_seed2026 | latest.pt | 71.9216 | IR->original_caption | 22.0149 | 59.1418 | 73.1343 |
| trimodal_580k_seed2026 | latest.pt | 71.9216 | original_caption->IR | 17.7239 | 52.2388 | 69.4030 |
| trimodal_580k_seed2026 | latest.pt | 71.9216 | RGB->IR | 85.6343 | 97.5746 | 99.2537 |
| trimodal_580k_seed2026 | latest.pt | 71.9216 | IR->RGB | 90.2985 | 97.5746 | 99.0672 |
| trimodal_580k_seed3407 | latest.pt | 72.1549 | IR->original_caption | 23.5075 | 57.6493 | 74.4403 |
| trimodal_580k_seed3407 | latest.pt | 72.1549 | original_caption->IR | 19.2164 | 50.9328 | 71.0821 |
| trimodal_580k_seed3407 | latest.pt | 72.1549 | RGB->IR | 86.3806 | 97.2015 | 99.0672 |
| trimodal_580k_seed3407 | latest.pt | 72.1549 | IR->RGB | 89.1791 | 98.1343 | 99.0672 |
| trimodal_580k_seed42 | latest.pt | 71.5019 | IR->original_caption | 22.0149 | 57.0896 | 73.8806 |
| trimodal_580k_seed42 | latest.pt | 71.5019 | original_caption->IR | 17.1642 | 51.8657 | 69.5896 |
| trimodal_580k_seed42 | latest.pt | 71.5019 | RGB->IR | 85.4478 | 97.7612 | 99.0672 |
| trimodal_580k_seed42 | latest.pt | 71.5019 | IR->RGB | 86.9403 | 98.3209 | 98.8806 |
