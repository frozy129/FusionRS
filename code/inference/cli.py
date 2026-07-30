from __future__ import annotations

import argparse
import json

from .modeling import FusionRSEncoder


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run FusionRS-CLIP inference")
    parser.add_argument("--model-dir", required=True, help="Local model directory")
    parser.add_argument("--rgb", required=True, help="RGB image path")
    parser.add_argument("--ir", required=True, help="IR or infrared-style image path")
    parser.add_argument("--text", required=True, help="Text description")
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default=None,
        help="Defaults to CUDA when available",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    encoder = FusionRSEncoder.from_pretrained(args.model_dir, device=args.device)
    rgb = encoder.encode_rgb([args.rgb])
    ir = encoder.encode_ir([args.ir])
    text = encoder.encode_text([args.text])
    result = {
        "device": str(encoder.device),
        "rgb_shape": list(rgb.shape),
        "ir_shape": list(ir.shape),
        "text_shape": list(text.shape),
        "scores": {
            "rgb_text": float((rgb[0] * text[0]).sum()),
            "ir_text": float((ir[0] * text[0]).sum()),
            "rgb_ir": float((rgb[0] * ir[0]).sum()),
        },
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
