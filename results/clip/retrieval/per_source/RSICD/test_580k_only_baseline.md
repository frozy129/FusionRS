# 580k-only CLIP baseline on clean held-out test

- test_pair_jsonl: `<machine-local-path-removed>`
- samples: 707
- baseline: `RSICD_joint_rgb_ir_phash4_caption_disjoint`

| model | checkpoint | Mean Recall | direction | R@1 | R@5 | R@10 |
|---|---:|---:|---:|---:|---:|---:|
| base_0k | latest.pt | 8.3451 | IR->original_caption | 1.4144 | 4.3847 | 9.1938 |
| base_0k | latest.pt | 8.3451 | original_caption->IR | 1.4144 | 6.5064 | 10.1839 |
| base_0k | latest.pt | 8.3451 | RGB->IR | 6.6478 | 15.5587 | 20.5092 |
| base_0k | latest.pt | 8.3451 | IR->RGB | 2.2631 | 8.2037 | 13.8614 |
| dual_text_580k_seed42 | latest.pt | 45.6153 | IR->original_caption | 9.0523 | 32.5318 | 51.9095 |
| dual_text_580k_seed42 | latest.pt | 45.6153 | original_caption->IR | 8.6280 | 29.1372 | 45.4031 |
| dual_text_580k_seed42 | latest.pt | 45.6153 | RGB->IR | 41.7256 | 67.7511 | 78.5007 |
| dual_text_580k_seed42 | latest.pt | 45.6153 | IR->RGB | 34.7949 | 68.1754 | 79.7737 |
| georsclip_vit_b32_0k | latest.pt | 16.4545 | IR->original_caption | 2.1216 | 8.9109 | 14.8515 |
| georsclip_vit_b32_0k | latest.pt | 16.4545 | original_caption->IR | 2.4045 | 10.7496 | 18.5290 |
| georsclip_vit_b32_0k | latest.pt | 16.4545 | RGB->IR | 12.4470 | 32.9562 | 44.1301 |
| georsclip_vit_b32_0k | latest.pt | 16.4545 | IR->RGB | 6.3649 | 18.2461 | 25.7426 |
| grayscale_580k_seed42 | latest.pt | 24.4932 | IR->original_caption | 3.6775 | 14.8515 | 24.1867 |
| grayscale_580k_seed42 | latest.pt | 24.4932 | original_caption->IR | 3.8190 | 14.1443 | 22.2065 |
| grayscale_580k_seed42 | latest.pt | 24.4932 | RGB->IR | 16.4074 | 33.9463 | 43.9887 |
| grayscale_580k_seed42 | latest.pt | 24.4932 | IR->RGB | 19.8020 | 42.9986 | 53.8897 |
| remoteclip_vit_b32_0k | latest.pt | 12.8359 | IR->original_caption | 2.4045 | 8.7694 | 14.1443 |
| remoteclip_vit_b32_0k | latest.pt | 12.8359 | original_caption->IR | 3.8190 | 11.8812 | 17.3975 |
| remoteclip_vit_b32_0k | latest.pt | 12.8359 | RGB->IR | 9.6181 | 20.7921 | 29.9859 |
| remoteclip_vit_b32_0k | latest.pt | 12.8359 | IR->RGB | 5.2334 | 12.0226 | 17.9632 |
| rgb_duplicate_580k_seed42 | latest.pt | 15.5823 | IR->original_caption | 3.3946 | 10.7496 | 19.6605 |
| rgb_duplicate_580k_seed42 | latest.pt | 15.5823 | original_caption->IR | 3.1117 | 12.7298 | 20.2263 |
| rgb_duplicate_580k_seed42 | latest.pt | 15.5823 | RGB->IR | 8.3451 | 20.2263 | 30.2687 |
| rgb_duplicate_580k_seed42 | latest.pt | 15.5823 | IR->RGB | 7.2136 | 19.6605 | 31.4003 |
| rgb_only_580k_seed42 | latest.pt | 15.2051 | IR->original_caption | 3.9604 | 11.1740 | 18.3876 |
| rgb_only_580k_seed42 | latest.pt | 15.2051 | original_caption->IR | 3.1117 | 12.5884 | 20.0849 |
| rgb_only_580k_seed42 | latest.pt | 15.2051 | RGB->IR | 6.9307 | 20.3678 | 28.5714 |
| rgb_only_580k_seed42 | latest.pt | 15.2051 | IR->RGB | 6.3649 | 20.5092 | 30.4102 |
| trimodal_100k_seed42 | latest.pt | 54.8208 | IR->original_caption | 6.3649 | 23.0552 | 35.9264 |
| trimodal_100k_seed42 | latest.pt | 54.8208 | original_caption->IR | 5.6577 | 20.2263 | 32.9562 |
| trimodal_100k_seed42 | latest.pt | 54.8208 | RGB->IR | 73.5502 | 91.6549 | 96.7468 |
| trimodal_100k_seed42 | latest.pt | 54.8208 | IR->RGB | 79.3494 | 94.3423 | 98.0198 |
| trimodal_300k_seed42 | latest.pt | 59.2881 | IR->original_caption | 10.7496 | 27.2984 | 42.5743 |
| trimodal_300k_seed42 | latest.pt | 59.2881 | original_caption->IR | 8.2037 | 27.7228 | 41.0184 |
| trimodal_300k_seed42 | latest.pt | 59.2881 | RGB->IR | 77.9349 | 95.6153 | 97.8784 |
| trimodal_300k_seed42 | latest.pt | 59.2881 | IR->RGB | 85.7143 | 97.7369 | 99.0099 |
| trimodal_50k_seed42 | latest.pt | 52.6992 | IR->original_caption | 5.7992 | 20.9335 | 32.1075 |
| trimodal_50k_seed42 | latest.pt | 52.6992 | original_caption->IR | 4.3847 | 17.5389 | 30.6931 |
| trimodal_50k_seed42 | latest.pt | 52.6992 | RGB->IR | 70.5799 | 90.5233 | 94.9081 |
| trimodal_50k_seed42 | latest.pt | 52.6992 | IR->RGB | 75.1061 | 92.6450 | 97.1711 |
| trimodal_580k_seed2026 | latest.pt | 61.5629 | IR->original_caption | 11.7397 | 33.3805 | 48.6563 |
| trimodal_580k_seed2026 | latest.pt | 61.5629 | original_caption->IR | 9.1938 | 29.5615 | 46.3932 |
| trimodal_580k_seed2026 | latest.pt | 61.5629 | RGB->IR | 80.7638 | 96.4639 | 98.3027 |
| trimodal_580k_seed2026 | latest.pt | 61.5629 | IR->RGB | 86.5629 | 98.1612 | 99.5757 |
| trimodal_580k_seed3407 | latest.pt | 61.6572 | IR->original_caption | 10.8911 | 33.6634 | 49.0806 |
| trimodal_580k_seed3407 | latest.pt | 61.6572 | original_caption->IR | 8.9109 | 30.4102 | 44.8373 |
| trimodal_580k_seed3407 | latest.pt | 61.6572 | RGB->IR | 81.7539 | 96.4639 | 98.7270 |
| trimodal_580k_seed3407 | latest.pt | 61.6572 | IR->RGB | 87.1287 | 98.4441 | 99.5757 |
| trimodal_580k_seed42 | latest.pt | 61.8458 | IR->original_caption | 11.0325 | 32.2489 | 50.2122 |
| trimodal_580k_seed42 | latest.pt | 61.8458 | original_caption->IR | 8.0622 | 29.7030 | 47.3833 |
| trimodal_580k_seed42 | latest.pt | 61.8458 | RGB->IR | 82.8854 | 96.8883 | 99.0099 |
| trimodal_580k_seed42 | latest.pt | 61.8458 | IR->RGB | 87.4116 | 97.7369 | 99.5757 |
