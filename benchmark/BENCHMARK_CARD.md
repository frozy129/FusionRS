# Benchmark card

## Tasks

1. **Infrared-style captioning:** describe visible scene semantics, grayscale intensity, contrast, edges, texture, and spatial structure without relying on RGB-only cues.
2. **Infrared-style VQA:** scene category, object presence, intensity/contrast, and spatial-structure questions answerable from the IR view.
3. **Real-thermal VQA:** binary object-presence questions grounded in HIT-UAV official bounding boxes.
4. **Cross-modal retrieval:** infrared-style image↔text and RGB↔infrared-style paired-view retrieval.

## Annotation provenance

Caption references and synthetic VQA records were generated and cross-checked by GPT-5.6 Sol, Terra, and Luna using paired RGB/IR views. The released records explicitly preserve their AI-assisted status. RGB was used only during annotation; the final questions were filtered for IR answerability.

HIT-UAV VQA labels are deterministically derived from official object bounding-box annotations. The source images remain governed by the HIT-UAV distribution terms.

## Metrics

- Captioning: BLEU, METEOR, ROUGE-L, CIDEr, and semantic similarity as implemented by the released scorer.
- VQA: normalized exact-match accuracy, reported overall and by question type.
- Retrieval: Recall@1/5/10 in both directions, using group-aware positives.
- Statistical comparison: paired record-level tests reported in `results/`.

## Versioning

Every public result must record the benchmark manifest hash, model/checkpoint hash, preprocessing configuration, and scorer version. Results from a different split or denominator must not be mixed into the same table.

