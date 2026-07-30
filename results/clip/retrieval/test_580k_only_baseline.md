# 580k-only CLIP baseline on clean held-out test

- test_pair_jsonl: `<machine-local-path-removed>`
- samples: 9553
- baseline: `joint_rgb_ir_phash4_caption_disjoint`

| model | checkpoint | Mean Recall | direction | R@1 | R@5 | R@10 |
|---|---:|---:|---:|---:|---:|---:|
| base_0k | latest.pt | 6.4290 | IR->original_caption | 1.1410 | 2.8577 | 4.5012 |
| base_0k | latest.pt | 6.4290 | original_caption->IR | 1.3504 | 4.1034 | 6.1028 |
| base_0k | latest.pt | 6.4290 | RGB->IR | 8.4371 | 16.1520 | 20.0775 |
| base_0k | latest.pt | 6.4290 | IR->RGB | 2.1773 | 4.3023 | 5.9458 |
| dual_text_580k_seed42 | latest.pt | 53.8487 | IR->original_caption | 20.9777 | 45.1900 | 56.6419 |
| dual_text_580k_seed42 | latest.pt | 53.8487 | original_caption->IR | 20.8312 | 44.2898 | 56.4535 |
| dual_text_580k_seed42 | latest.pt | 53.8487 | RGB->IR | 56.9036 | 74.1861 | 80.2994 |
| dual_text_580k_seed42 | latest.pt | 53.8487 | IR->RGB | 50.6647 | 66.7644 | 72.9823 |
| georsclip_vit_b32_0k | latest.pt | 9.4246 | IR->original_caption | 1.2561 | 4.0301 | 6.2284 |
| georsclip_vit_b32_0k | latest.pt | 9.4246 | original_caption->IR | 3.3288 | 8.5104 | 12.0067 |
| georsclip_vit_b32_0k | latest.pt | 9.4246 | RGB->IR | 12.2789 | 22.9771 | 28.8182 |
| georsclip_vit_b32_0k | latest.pt | 9.4246 | IR->RGB | 2.1355 | 4.8990 | 6.6262 |
| grayscale_580k_seed42 | latest.pt | 21.2490 | IR->original_caption | 4.3756 | 12.2998 | 17.6489 |
| grayscale_580k_seed42 | latest.pt | 21.2490 | original_caption->IR | 7.3903 | 17.3244 | 23.7936 |
| grayscale_580k_seed42 | latest.pt | 21.2490 | RGB->IR | 21.5011 | 34.7535 | 41.7670 |
| grayscale_580k_seed42 | latest.pt | 21.2490 | IR->RGB | 16.1311 | 26.2117 | 31.7911 |
| remoteclip_vit_b32_0k | latest.pt | 2.9903 | IR->original_caption | 0.2722 | 0.7746 | 1.4550 |
| remoteclip_vit_b32_0k | latest.pt | 2.9903 | original_caption->IR | 0.2617 | 1.0782 | 1.7063 |
| remoteclip_vit_b32_0k | latest.pt | 2.9903 | RGB->IR | 3.7370 | 8.0394 | 10.7191 |
| remoteclip_vit_b32_0k | latest.pt | 2.9903 | IR->RGB | 1.2038 | 2.7845 | 3.8522 |
| rgb_duplicate_580k_seed42 | latest.pt | 14.4405 | IR->original_caption | 2.5228 | 6.9926 | 10.5203 |
| rgb_duplicate_580k_seed42 | latest.pt | 14.4405 | original_caption->IR | 6.0505 | 14.1736 | 19.9100 |
| rgb_duplicate_580k_seed42 | latest.pt | 14.4405 | RGB->IR | 16.1939 | 28.7763 | 35.2141 |
| rgb_duplicate_580k_seed42 | latest.pt | 14.4405 | IR->RGB | 5.4852 | 11.7031 | 15.7437 |
| rgb_only_580k_seed42 | latest.pt | 14.3820 | IR->original_caption | 2.4390 | 6.9821 | 10.7610 |
| rgb_only_580k_seed42 | latest.pt | 14.3820 | original_caption->IR | 6.0400 | 14.0479 | 20.0147 |
| rgb_only_580k_seed42 | latest.pt | 14.3820 | RGB->IR | 16.1415 | 28.5460 | 34.9000 |
| rgb_only_580k_seed42 | latest.pt | 14.3820 | IR->RGB | 5.6527 | 11.5042 | 15.5553 |
| trimodal_100k_seed42 | latest.pt | 58.3604 | IR->original_caption | 12.4254 | 29.7812 | 40.4480 |
| trimodal_100k_seed42 | latest.pt | 58.3604 | original_caption->IR | 12.0800 | 29.2683 | 39.2337 |
| trimodal_100k_seed42 | latest.pt | 58.3604 | RGB->IR | 80.9484 | 92.5050 | 95.3522 |
| trimodal_100k_seed42 | latest.pt | 58.3604 | IR->RGB | 81.0531 | 92.2328 | 94.9963 |
| trimodal_300k_seed42 | latest.pt | 64.4248 | IR->original_caption | 17.5128 | 39.1709 | 50.4135 |
| trimodal_300k_seed42 | latest.pt | 64.4248 | original_caption->IR | 17.1360 | 38.1032 | 49.5028 |
| trimodal_300k_seed42 | latest.pt | 64.4248 | RGB->IR | 86.6011 | 95.9803 | 97.9797 |
| trimodal_300k_seed42 | latest.pt | 64.4248 | IR->RGB | 87.3234 | 95.8233 | 97.5505 |
| trimodal_50k_seed42 | latest.pt | 54.3904 | IR->original_caption | 9.6410 | 24.7357 | 34.4813 |
| trimodal_50k_seed42 | latest.pt | 54.3904 | original_caption->IR | 10.0597 | 24.6938 | 33.5811 |
| trimodal_50k_seed42 | latest.pt | 54.3904 | RGB->IR | 76.3634 | 89.6786 | 93.1226 |
| trimodal_50k_seed42 | latest.pt | 54.3904 | IR->RGB | 75.6412 | 88.7156 | 91.9711 |
| trimodal_580k_seed2026 | latest.pt | 67.8818 | IR->original_caption | 20.8626 | 45.1900 | 56.7256 |
| trimodal_580k_seed2026 | latest.pt | 67.8818 | original_caption->IR | 20.0775 | 43.7768 | 55.6893 |
| trimodal_580k_seed2026 | latest.pt | 67.8818 | RGB->IR | 89.6158 | 97.4354 | 98.7020 |
| trimodal_580k_seed2026 | latest.pt | 67.8818 | IR->RGB | 90.4637 | 97.3830 | 98.6601 |
| trimodal_580k_seed3407 | latest.pt | 67.8373 | IR->original_caption | 20.3392 | 45.1167 | 56.5477 |
| trimodal_580k_seed3407 | latest.pt | 67.8373 | original_caption->IR | 20.1821 | 43.7559 | 55.9301 |
| trimodal_580k_seed3407 | latest.pt | 67.8373 | RGB->IR | 89.7728 | 97.4040 | 98.7648 |
| trimodal_580k_seed3407 | latest.pt | 67.8373 | IR->RGB | 90.4009 | 97.2365 | 98.5973 |
| trimodal_580k_seed42 | latest.pt | 67.6568 | IR->original_caption | 20.2868 | 44.4468 | 56.4639 |
| trimodal_580k_seed42 | latest.pt | 67.6568 | original_caption->IR | 19.8262 | 43.4000 | 55.6056 |
| trimodal_580k_seed42 | latest.pt | 67.6568 | RGB->IR | 89.5949 | 97.5505 | 98.8171 |
| trimodal_580k_seed42 | latest.pt | 67.6568 | IR->RGB | 89.9717 | 97.2888 | 98.6287 |
