# Annotation provenance

The synthetic Caption/VQA pool was annotated independently by three GPT-5.6 variants: Sol, Terra, and Luna. Every annotator received the paired RGB and infrared-style views. Caption references were retained after format and IR-observability filtering. VQA questions were required to remain answerable from the infrared-style view alone.

The released Caption manifest contains the accepted reference sets and their contributing model identifiers. The synthetic VQA manifest contains the canonical answer, accepted-answer set, question type, and concise IR evidence used during review. These labels are AI-assisted rather than human-authored.

The HIT-UAV real-thermal VQA manifest is generated deterministically from the official test annotations. Each record retains its source image locator, object class, question type, accepted answer, and label source.

