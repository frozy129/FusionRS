# CLIP checkpoint selection on clean held-out val

- val_pair_jsonl: `<machine-local-path-removed>`
- samples: 9538

| model | checkpoint | Mean Recall | direction | R@1 | R@5 | R@10 |
|---|---:|---:|---:|---:|---:|---:|
| georsclip_0k | latest.pt | 9.1738 | IR->original_caption | 1.3001 | 3.9631 | 6.0390 |
| georsclip_0k | latest.pt | 9.1738 | original_caption->IR | 3.1873 | 7.9052 | 11.4175 |
| georsclip_0k | latest.pt | 9.1738 | RGB->IR | 12.1828 | 22.4995 | 28.0772 |
| georsclip_0k | latest.pt | 9.1738 | IR->RGB | 2.1808 | 4.6655 | 6.6681 |
| georsclip_fusionrs_580k_seed42 | latest.pt | 68.7889 | IR->original_caption | 22.2374 | 47.3160 | 59.1529 |
| georsclip_fusionrs_580k_seed42 | latest.pt | 68.7889 | original_caption->IR | 22.2583 | 46.2361 | 58.0520 |
| georsclip_fusionrs_580k_seed42 | latest.pt | 68.7889 | RGB->IR | 88.9914 | 97.1692 | 98.5217 |
| georsclip_fusionrs_580k_seed42 | latest.pt | 68.7889 | IR->RGB | 89.6624 | 97.3684 | 98.5007 |
| remoteclip_0k | latest.pt | 3.0317 | IR->original_caption | 0.3040 | 0.9226 | 1.6670 |
| remoteclip_0k | latest.pt | 3.0317 | original_caption->IR | 0.4194 | 1.2267 | 1.9606 |
| remoteclip_0k | latest.pt | 3.0317 | RGB->IR | 3.6590 | 7.9052 | 10.7465 |
| remoteclip_0k | latest.pt | 3.0317 | IR->RGB | 1.1847 | 2.5896 | 3.7953 |
| remoteclip_fusionrs_580k_seed42 | latest.pt | 66.7278 | IR->original_caption | 19.3122 | 42.5875 | 54.3930 |
| remoteclip_fusionrs_580k_seed42 | latest.pt | 66.7278 | original_caption->IR | 18.5783 | 41.4447 | 53.7744 |
| remoteclip_fusionrs_580k_seed42 | latest.pt | 66.7278 | RGB->IR | 89.1801 | 96.9281 | 98.2701 |
| remoteclip_fusionrs_580k_seed42 | latest.pt | 66.7278 | IR->RGB | 90.4068 | 97.3579 | 98.5007 |
