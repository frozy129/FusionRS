# 580k-only CLIP baseline on clean held-out test

- test_pair_jsonl: `<machine-local-path-removed>`
- samples: 9553
- baseline: `joint_multiclip_rc4`

| model | checkpoint | Mean Recall | direction | R@1 | R@5 | R@10 |
|---|---:|---:|---:|---:|---:|---:|
| georsclip_0k | latest.pt | 9.4246 | IR->original_caption | 1.2561 | 4.0301 | 6.2284 |
| georsclip_0k | latest.pt | 9.4246 | original_caption->IR | 3.3288 | 8.5104 | 12.0067 |
| georsclip_0k | latest.pt | 9.4246 | RGB->IR | 12.2789 | 22.9771 | 28.8182 |
| georsclip_0k | latest.pt | 9.4246 | IR->RGB | 2.1355 | 4.8990 | 6.6262 |
| georsclip_fusionrs_580k_seed42 | latest.pt | 69.0848 | IR->original_caption | 23.1550 | 47.5034 | 59.5729 |
| georsclip_fusionrs_580k_seed42 | latest.pt | 69.0848 | original_caption->IR | 22.4641 | 46.4357 | 58.6413 |
| georsclip_fusionrs_580k_seed42 | latest.pt | 69.0848 | RGB->IR | 89.2495 | 97.3411 | 98.6496 |
| georsclip_fusionrs_580k_seed42 | latest.pt | 69.0848 | IR->RGB | 90.0450 | 97.3516 | 98.6078 |
| remoteclip_0k | latest.pt | 2.9903 | IR->original_caption | 0.2722 | 0.7746 | 1.4550 |
| remoteclip_0k | latest.pt | 2.9903 | original_caption->IR | 0.2617 | 1.0782 | 1.7063 |
| remoteclip_0k | latest.pt | 2.9903 | RGB->IR | 3.7370 | 8.0394 | 10.7191 |
| remoteclip_0k | latest.pt | 2.9903 | IR->RGB | 1.2038 | 2.7845 | 3.8522 |
| remoteclip_fusionrs_580k_seed42 | latest.pt | 67.1264 | IR->original_caption | 19.7320 | 43.6303 | 55.3125 |
| remoteclip_fusionrs_580k_seed42 | latest.pt | 67.1264 | original_caption->IR | 19.1144 | 42.0077 | 53.9412 |
| remoteclip_fusionrs_580k_seed42 | latest.pt | 67.1264 | RGB->IR | 89.5635 | 97.3307 | 98.5554 |
| remoteclip_fusionrs_580k_seed42 | latest.pt | 67.1264 | IR->RGB | 90.4009 | 97.3097 | 98.6182 |
