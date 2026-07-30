# 580k-only CLIP baseline on clean held-out test

- test_pair_jsonl: `<machine-local-path-removed>`
- samples: 9553
- baseline: `joint_b32_l14_50k`

| model | checkpoint | Mean Recall | direction | R@1 | R@5 | R@10 |
|---|---:|---:|---:|---:|---:|---:|
| b32_50k_seed2026 | latest.pt | 55.1450 | IR->original_caption | 10.1539 | 26.3582 | 35.6747 |
| b32_50k_seed2026 | latest.pt | 55.1450 | original_caption->IR | 10.3528 | 26.0232 | 35.6956 |
| b32_50k_seed2026 | latest.pt | 55.1450 | RGB->IR | 76.0285 | 89.7728 | 93.1645 |
| b32_50k_seed2026 | latest.pt | 55.1450 | IR->RGB | 76.3530 | 89.4169 | 92.7457 |
| b32_50k_seed3407 | latest.pt | 55.0700 | IR->original_caption | 10.5098 | 26.1279 | 36.0515 |
| b32_50k_seed3407 | latest.pt | 55.0700 | original_caption->IR | 10.4993 | 25.8872 | 35.3606 |
| b32_50k_seed3407 | latest.pt | 55.0700 | RGB->IR | 75.9761 | 89.2913 | 93.1226 |
| b32_50k_seed3407 | latest.pt | 55.0700 | IR->RGB | 76.1332 | 89.1762 | 92.7039 |
| b32_50k_seed42 | latest.pt | 55.2165 | IR->original_caption | 10.7401 | 25.9918 | 36.0410 |
| b32_50k_seed42 | latest.pt | 55.2165 | original_caption->IR | 10.4470 | 25.8767 | 35.3502 |
| b32_50k_seed42 | latest.pt | 55.2165 | RGB->IR | 76.5414 | 89.8880 | 93.3215 |
| b32_50k_seed42 | latest.pt | 55.2165 | IR->RGB | 76.4472 | 89.2390 | 92.7143 |
| base_b32_0k | latest.pt | 6.4456 | IR->original_caption | 1.1515 | 2.8682 | 4.5117 |
| base_b32_0k | latest.pt | 6.4456 | original_caption->IR | 1.3504 | 4.1139 | 6.0923 |
| base_b32_0k | latest.pt | 6.4456 | RGB->IR | 8.4685 | 16.1729 | 20.1193 |
| base_b32_0k | latest.pt | 6.4456 | IR->RGB | 2.1669 | 4.3128 | 6.0191 |
| base_l14_0k | latest.pt | 9.3016 | IR->original_caption | 2.0412 | 5.2340 | 7.7358 |
| base_l14_0k | latest.pt | 9.3016 | original_caption->IR | 2.2611 | 5.9039 | 8.5209 |
| base_l14_0k | latest.pt | 9.3016 | RGB->IR | 10.9390 | 20.1612 | 24.8090 |
| base_l14_0k | latest.pt | 9.3016 | IR->RGB | 4.4070 | 8.5104 | 11.0960 |
| l14_50k_seed2026 | latest.pt | 61.5496 | IR->original_caption | 16.3195 | 36.6691 | 47.4929 |
| l14_50k_seed2026 | latest.pt | 61.5496 | original_caption->IR | 15.8589 | 34.8896 | 46.2996 |
| l14_50k_seed2026 | latest.pt | 61.5496 | RGB->IR | 81.6393 | 92.5573 | 95.4988 |
| l14_50k_seed2026 | latest.pt | 61.5496 | IR->RGB | 83.2409 | 92.9132 | 95.2162 |
| l14_50k_seed3407 | latest.pt | 61.6569 | IR->original_caption | 16.3928 | 36.1666 | 47.2103 |
| l14_50k_seed3407 | latest.pt | 61.6569 | original_caption->IR | 16.0683 | 34.9314 | 46.0693 |
| l14_50k_seed3407 | latest.pt | 61.6569 | RGB->IR | 81.9010 | 92.6411 | 95.5511 |
| l14_50k_seed3407 | latest.pt | 61.6569 | IR->RGB | 84.0783 | 93.3110 | 95.5616 |
| l14_50k_seed42 | latest.pt | 61.5845 | IR->original_caption | 16.4556 | 36.2190 | 47.1266 |
| l14_50k_seed42 | latest.pt | 61.5845 | original_caption->IR | 15.9740 | 35.2873 | 45.7029 |
| l14_50k_seed42 | latest.pt | 61.5845 | RGB->IR | 81.6811 | 92.6097 | 95.4151 |
| l14_50k_seed42 | latest.pt | 61.5845 | IR->RGB | 83.9003 | 93.2587 | 95.3836 |
