# Frozen evaluation protocol

The canonical `rc4` split assigns every connected RGB, infrared-style, and caption-linked component wholly to train, validation, or test. Validation is used for checkpoint selection; test is evaluated after selection.

For image–text retrieval, targets sharing the same normalized-caption group are positives. For paired RGB–IR retrieval, views sharing the same sample ID are positives. Recall@1, Recall@5, and Recall@10 are computed in both directions.

Generative evaluation consumes only the infrared-style or real-thermal image at inference. Caption rows use their complete released reference set. VQA uses normalized exact match against `accepted_answers`. Failed or missing predictions remain in the denominator and are exposed by the output audits.

The synthetic benchmark consists of 998 Caption rows and 1,993 VQA rows. The real-thermal suite consists of 3,223 VQA rows. These denominators must not be changed without publishing a new manifest version and checksum.

