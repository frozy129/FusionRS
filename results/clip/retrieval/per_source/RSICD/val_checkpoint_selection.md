# CLIP checkpoint selection on clean held-out val

- val_pair_jsonl: `<machine-local-path-removed>`
- samples: 700

| model | checkpoint | Mean Recall | direction | R@1 | R@5 | R@10 |
|---|---:|---:|---:|---:|---:|---:|
| base_0k | latest.pt | 8.4524 | IR->original_caption | 1.0000 | 4.5714 | 7.7143 |
| base_0k | latest.pt | 8.4524 | original_caption->IR | 2.4286 | 6.7143 | 10.0000 |
| base_0k | latest.pt | 8.4524 | RGB->IR | 5.7143 | 16.1429 | 23.1429 |
| base_0k | latest.pt | 8.4524 | IR->RGB | 2.4286 | 8.8571 | 12.7143 |
| dual_text_580k_seed42 | latest.pt | 44.4167 | IR->original_caption | 10.0000 | 31.4286 | 52.2857 |
| dual_text_580k_seed42 | latest.pt | 44.4167 | original_caption->IR | 9.4286 | 29.1429 | 45.0000 |
| dual_text_580k_seed42 | latest.pt | 44.4167 | RGB->IR | 36.5714 | 63.5714 | 75.7143 |
| dual_text_580k_seed42 | latest.pt | 44.4167 | IR->RGB | 37.8571 | 63.2857 | 78.7143 |
| georsclip_vit_b32_0k | latest.pt | 16.6905 | IR->original_caption | 2.1429 | 9.7143 | 16.1429 |
| georsclip_vit_b32_0k | latest.pt | 16.6905 | original_caption->IR | 3.8571 | 12.7143 | 21.0000 |
| georsclip_vit_b32_0k | latest.pt | 16.6905 | RGB->IR | 12.5714 | 31.7143 | 42.8571 |
| georsclip_vit_b32_0k | latest.pt | 16.6905 | IR->RGB | 5.7143 | 17.1429 | 24.7143 |
| grayscale_580k_seed42 | latest.pt | 24.8690 | IR->original_caption | 4.4286 | 15.5714 | 26.1429 |
| grayscale_580k_seed42 | latest.pt | 24.8690 | original_caption->IR | 5.0000 | 15.4286 | 23.4286 |
| grayscale_580k_seed42 | latest.pt | 24.8690 | RGB->IR | 17.1429 | 33.7143 | 44.5714 |
| grayscale_580k_seed42 | latest.pt | 24.8690 | IR->RGB | 18.8571 | 40.5714 | 53.5714 |
| remoteclip_vit_b32_0k | latest.pt | 14.0952 | IR->original_caption | 3.5714 | 11.5714 | 19.1429 |
| remoteclip_vit_b32_0k | latest.pt | 14.0952 | original_caption->IR | 2.7143 | 11.0000 | 19.2857 |
| remoteclip_vit_b32_0k | latest.pt | 14.0952 | RGB->IR | 9.1429 | 21.0000 | 30.5714 |
| remoteclip_vit_b32_0k | latest.pt | 14.0952 | IR->RGB | 6.8571 | 13.4286 | 20.8571 |
| rgb_duplicate_580k_seed42 | latest.pt | 15.2500 | IR->original_caption | 2.7143 | 11.4286 | 21.4286 |
| rgb_duplicate_580k_seed42 | latest.pt | 15.2500 | original_caption->IR | 3.5714 | 12.5714 | 19.5714 |
| rgb_duplicate_580k_seed42 | latest.pt | 15.2500 | RGB->IR | 7.0000 | 19.4286 | 29.4286 |
| rgb_duplicate_580k_seed42 | latest.pt | 15.2500 | IR->RGB | 7.7143 | 18.8571 | 29.2857 |
| rgb_only_580k_seed42 | latest.pt | 14.7857 | IR->original_caption | 2.7143 | 12.7143 | 20.4286 |
| rgb_only_580k_seed42 | latest.pt | 14.7857 | original_caption->IR | 3.5714 | 12.2857 | 19.2857 |
| rgb_only_580k_seed42 | latest.pt | 14.7857 | RGB->IR | 7.1429 | 19.0000 | 27.8571 |
| rgb_only_580k_seed42 | latest.pt | 14.7857 | IR->RGB | 6.7143 | 18.0000 | 27.7143 |
| trimodal_100k_seed42 | latest.pt | 55.7976 | IR->original_caption | 7.7143 | 26.1429 | 40.4286 |
| trimodal_100k_seed42 | latest.pt | 55.7976 | original_caption->IR | 6.2857 | 23.7143 | 36.8571 |
| trimodal_100k_seed42 | latest.pt | 55.7976 | RGB->IR | 70.1429 | 91.2857 | 95.1429 |
| trimodal_100k_seed42 | latest.pt | 55.7976 | IR->RGB | 78.7143 | 95.5714 | 97.5714 |
| trimodal_300k_seed42 | latest.pt | 59.1071 | IR->original_caption | 9.4286 | 29.2857 | 46.0000 |
| trimodal_300k_seed42 | latest.pt | 59.1071 | original_caption->IR | 7.2857 | 28.2857 | 43.5714 |
| trimodal_300k_seed42 | latest.pt | 59.1071 | RGB->IR | 74.1429 | 93.2857 | 97.7143 |
| trimodal_300k_seed42 | latest.pt | 59.1071 | IR->RGB | 83.0000 | 97.7143 | 99.5714 |
| trimodal_50k_seed42 | latest.pt | 52.7976 | IR->original_caption | 6.2857 | 21.7143 | 36.1429 |
| trimodal_50k_seed42 | latest.pt | 52.7976 | original_caption->IR | 5.2857 | 20.4286 | 33.4286 |
| trimodal_50k_seed42 | latest.pt | 52.7976 | RGB->IR | 67.4286 | 87.5714 | 93.0000 |
| trimodal_50k_seed42 | latest.pt | 52.7976 | IR->RGB | 73.2857 | 92.8571 | 96.1429 |
| trimodal_580k_seed2026 | latest.pt | 61.6786 | IR->original_caption | 12.4286 | 34.4286 | 51.8571 |
| trimodal_580k_seed2026 | latest.pt | 61.6786 | original_caption->IR | 9.1429 | 29.8571 | 46.5714 |
| trimodal_580k_seed2026 | latest.pt | 61.6786 | RGB->IR | 78.5714 | 95.2857 | 98.2857 |
| trimodal_580k_seed2026 | latest.pt | 61.6786 | IR->RGB | 85.8571 | 98.2857 | 99.5714 |
| trimodal_580k_seed3407 | latest.pt | 61.4286 | IR->original_caption | 11.5714 | 33.7143 | 51.8571 |
| trimodal_580k_seed3407 | latest.pt | 61.4286 | original_caption->IR | 8.7143 | 29.5714 | 46.2857 |
| trimodal_580k_seed3407 | latest.pt | 61.4286 | RGB->IR | 77.7143 | 95.0000 | 98.0000 |
| trimodal_580k_seed3407 | latest.pt | 61.4286 | IR->RGB | 86.0000 | 99.0000 | 99.7143 |
| trimodal_580k_seed42 | latest.pt | 60.7619 | IR->original_caption | 9.8571 | 33.2857 | 48.4286 |
| trimodal_580k_seed42 | latest.pt | 60.7619 | original_caption->IR | 8.1429 | 29.8571 | 44.0000 |
| trimodal_580k_seed42 | latest.pt | 60.7619 | RGB->IR | 78.4286 | 95.4286 | 98.7143 |
| trimodal_580k_seed42 | latest.pt | 60.7619 | IR->RGB | 85.1429 | 98.1429 | 99.7143 |
