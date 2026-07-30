# Caltech Aerial RGB-T paired retrieval

Exact-pair retrieval on the official synchronized and rectified RGB-thermal test split.

- pairs: 279
- random R@1: 0.3584

| model | checkpoint | Mean R | RGB->T R@1 | T->RGB R@1 |
|---|---:|---:|---:|---:|
| base_0k | latest.pt | 22.8793 | 7.8853 | 7.8853 |
| dual_text_580k_seed42 | latest.pt | 43.4289 | 23.6559 | 15.4122 |
| georsclip_vit_b32_0k | latest.pt | 24.7909 | 13.2616 | 7.8853 |
| grayscale_580k_seed42 | latest.pt | 35.3644 | 16.8459 | 15.7706 |
| remoteclip_vit_b32_0k | latest.pt | 15.2927 | 7.1685 | 5.3763 |
| rgb_duplicate_580k_seed42 | latest.pt | 33.0944 | 15.4122 | 11.8280 |
| trimodal_580k_seed42 | latest.pt | 68.0406 | 40.8602 | 39.0681 |
