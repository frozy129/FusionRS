#!/usr/bin/env python3
"""Run deterministic FusionRS Caption and VQA inference with Qwen2.5-VL."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from PIL import Image
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    InstructBlipForConditionalGeneration,
    InstructBlipProcessor,
    Qwen2_5_VLForConditionalGeneration,
)


CAPTION_PROMPTS = {
    "semantic_description": (
        "Describe the scene and visible objects in this infrared-style aerial "
        "image in one concise sentence."
    ),
    "ir_observable_description": (
        "Describe this infrared-style aerial image in one concise sentence. "
        "Report visible scene semantics, grayscale intensity, contrast, edges, "
        "texture, or structure only. Do not infer temperature, heat, material "
        "properties, emissivity, or RGB color."
    ),
}


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def resolve_image(root: Path, relative_path: str) -> str:
    path = Path(relative_path)
    if not path.is_absolute():
        path = root / path
    if not path.is_file():
        raise FileNotFoundError(path)
    return str(path)


def load_done(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("status") == "success" and row.get("record_key"):
                done.add(row["record_key"])
    return done


class QwenRunner:
    def __init__(self, model_path: str, adapter_path: str | None) -> None:
        self.processor = AutoProcessor.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=True
        )
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            local_files_only=True,
            trust_remote_code=True,
        ).eval()
        if adapter_path:
            model = PeftModel.from_pretrained(
                model, adapter_path, is_trainable=False
            ).eval()
        self.model = model

    def infer(self, image_path: str, prompt: str, max_new_tokens: int) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
            )
        trimmed = [
            output[len(input_ids) :]
            for input_ids, output in zip(inputs.input_ids, generated)
        ]
        return " ".join(
            self.processor.batch_decode(
                trimmed, skip_special_tokens=True
            )[0].split()
        )


class InstructBlipRunner:
    def __init__(self, model_path: str) -> None:
        self.processor = InstructBlipProcessor.from_pretrained(
            model_path, local_files_only=True
        )
        self.model = InstructBlipForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            local_files_only=True,
        ).eval()

    def infer(self, image_path: str, prompt: str, max_new_tokens: int) -> str:
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(
            images=image, text=prompt, return_tensors="pt"
        )
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
            )
        return " ".join(
            self.processor.batch_decode(
                generated, skip_special_tokens=True
            )[0].split()
        )


class HFImageTextRunner:
    def __init__(self, model_path: str) -> None:
        self.processor = AutoProcessor.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=True
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            local_files_only=True,
            trust_remote_code=True,
        ).eval()

    def infer(self, image_path: str, prompt: str, max_new_tokens: int) -> str:
        image = Image.open(image_path).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        if hasattr(self.processor, "apply_chat_template"):
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(
                text=text, images=image, return_tensors="pt"
            )
        else:
            inputs = self.processor(
                text=f"USER: <image>\n{prompt}\nASSISTANT:",
                images=image,
                return_tensors="pt",
            )
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        prompt_length = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
            )
        output_ids = generated[0, prompt_length:]
        return " ".join(
            self.processor.decode(output_ids, skip_special_tokens=True).split()
        )


def build_records(
    caption_manifest: Path, vqa_manifest: Path, image_root: Path
) -> list[dict]:
    records: list[dict] = []
    for row in read_jsonl(caption_manifest):
        task = row["task_type"]
        records.append(
            {
                "record_key": f"caption:{row['benchmark_id']}:{task}",
                "benchmark_id": row["benchmark_id"],
                "dataset": row["dataset"],
                "image_path": resolve_image(image_root, row["ir_image"]),
                "evaluation_type": "caption",
                "task_type": task,
                "prompt": CAPTION_PROMPTS[task],
                "references": row["references"],
                "reference_models": row["reference_models"],
                "max_new_tokens": 96,
            }
        )
    for row in read_jsonl(vqa_manifest):
        question = row["question"]
        image_path = row.get("ir_image") or row.get("image_locator")
        if not image_path:
            raise KeyError(f"missing VQA image locator: {row}")
        records.append(
            {
                "record_key": f"vqa:{row['question_id']}",
                "benchmark_id": row.get(
                    "benchmark_id",
                    f"{row.get('dataset', 'unknown')}:{row.get('image_id', '')}",
                ),
                "question_id": row["question_id"],
                "dataset": row.get("dataset", "unknown"),
                "image_path": resolve_image(image_root, image_path),
                "evaluation_type": "vqa",
                "question_type": row["question_type"],
                "question": question,
                "prompt": row.get("prompt")
                or (
                    "Answer the following question using only the visible "
                    "content of this infrared-style aerial image. Give a short "
                    f"answer.\nQuestion: {question}"
                ),
                "canonical_answer": row["canonical_answer"],
                "accepted_answers": row.get("accepted_answers")
                or [row["canonical_answer"]],
                "max_new_tokens": int(row.get("max_new_tokens", 32)),
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-kind",
        choices=("qwen", "instructblip", "hf"),
        default="qwen",
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-path")
    parser.add_argument("--model-tag", required=True)
    parser.add_argument("--caption-manifest", type=Path, required=True)
    parser.add_argument("--vqa-manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args()

    records = build_records(
        args.caption_manifest, args.vqa_manifest, args.image_root
    )
    if args.limit is not None:
        records = records[: args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(args.output)
    if args.model_kind == "qwen":
        runner = QwenRunner(args.model_path, args.adapter_path)
    elif args.model_kind == "instructblip":
        if args.adapter_path:
            raise ValueError("InstructBLIP adapter loading is not supported")
        runner = InstructBlipRunner(args.model_path)
    else:
        if args.adapter_path:
            raise ValueError("generic HF adapter loading is not supported")
        runner = HFImageTextRunner(args.model_path)

    started = time.time()
    errors = 0
    with args.output.open("a", encoding="utf-8") as handle:
        for index, row in enumerate(records, 1):
            if row["record_key"] in done:
                continue
            error = ""
            prediction = ""
            for attempt in range(1, args.retries + 2):
                try:
                    prediction = runner.infer(
                        row["image_path"],
                        row["prompt"],
                        row["max_new_tokens"],
                    )
                    if not prediction:
                        raise RuntimeError("empty prediction")
                    error = ""
                    break
                except Exception as exc:  # Keep raw failure evidence.
                    error = f"attempt={attempt}: {exc!r}"
            status = "success" if prediction else "error"
            errors += status == "error"
            output = {
                **row,
                "model_tag": args.model_tag,
                "adapter_path": args.adapter_path or "",
                "prediction": prediction,
                "status": status,
                "error": error,
            }
            handle.write(json.dumps(output, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            if index % 10 == 0 or status == "error":
                print(
                    json.dumps(
                        {
                            "model": args.model_tag,
                            "completed": index,
                            "total": len(records),
                            "errors": errors,
                        }
                    ),
                    flush=True,
                )

    successful = load_done(args.output)
    expected = {row["record_key"] for row in records}
    missing = expected - successful
    summary = {
        "model_tag": args.model_tag,
        "adapter_path": args.adapter_path or "",
        "expected": len(expected),
        "successful": len(expected & successful),
        "missing_or_error": len(missing),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    if missing:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
