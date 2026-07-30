#!/usr/bin/env python3
"""Full-manifest Qwen2.5-VL LoRA training with assistant-only supervision."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from qwen_vl_utils import process_vision_info
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
)


def find_last_subsequence(values: list[int], needle: list[int]) -> int:
    if not needle:
        raise ValueError("empty subsequence")
    for start in range(len(values) - len(needle), -1, -1):
        if values[start : start + len(needle)] == needle:
            return start
    return -1


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CaptionDataset(Dataset):
    def __init__(self, manifest: Path, image_root: Path):
        self.rows = []
        with manifest.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                row = json.loads(line)
                if row.get("benchmark_split") != "train":
                    raise ValueError(f"{manifest}:{line_number}: non-train row")
                image = image_root / row["image"]
                if not image.is_file():
                    raise FileNotFoundError(image)
                row["resolved_image"] = str(image)
                self.rows.append(row)
        if not self.rows:
            raise ValueError("empty training manifest")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        return self.rows[index]


@dataclass
class AssistantOnlyCollator:
    processor: Any

    def __post_init__(self) -> None:
        self.assistant_header = self.processor.tokenizer.encode(
            "<|im_start|>assistant\n", add_special_tokens=False
        )
        if not self.assistant_header:
            raise ValueError("tokenizer produced an empty assistant header")

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        texts = []
        images = []
        for row in batch:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": row["resolved_image"]},
                        {"type": "text", "text": row["prompt"]},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": row["response"]}],
                },
            ]
            texts.append(
                self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            )
            image_inputs, _ = process_vision_info(messages)
            images.append(image_inputs[0])

        inputs = self.processor(
            text=texts,
            images=images,
            padding=True,
            return_tensors="pt",
        )
        labels = inputs["input_ids"].clone()
        for row_index, token_row in enumerate(inputs["input_ids"].tolist()):
            start = find_last_subsequence(token_row, self.assistant_header)
            if start < 0:
                raise RuntimeError("assistant header not found in tokenized sample")
            response_start = start + len(self.assistant_header)
            labels[row_index, :response_start] = -100
        labels[inputs["attention_mask"] == 0] = -100
        if (labels != -100).sum(dim=1).min().item() == 0:
            raise RuntimeError("a batch row has no assistant response tokens")
        inputs["labels"] = labels
        return inputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model-name", default="Qwen/Qwen2.5-VL-7B-Instruct"
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--max-pixels", type=int, default=401408)
    parser.add_argument("--dry-run-batches", type=int, default=0)
    args = parser.parse_args()

    seed_everything(args.seed)
    dataset = CaptionDataset(args.manifest, args.image_root)
    processor = AutoProcessor.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        max_pixels=args.max_pixels,
    )
    collator = AssistantOnlyCollator(processor)
    if args.dry_run_batches:
        for index in range(min(args.dry_run_batches, len(dataset))):
            batch = collator([dataset[index]])
            trainable = int((batch["labels"] != -100).sum())
            print(
                json.dumps(
                    {
                        "sample_id": dataset[index]["sample_id"],
                        "input_shape": list(batch["input_ids"].shape),
                        "assistant_tokens": trainable,
                    }
                )
            )
        return

    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required for Qwen2.5-VL-7B QLoRA training")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=(
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        ),
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=(
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        ),
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True
    )
    model.enable_input_require_grads()
    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        ),
    )
    model.print_trainable_parameters()
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=10,
        save_steps=args.save_steps,
        save_strategy="steps",
        save_total_limit=2,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,
        remove_unused_columns=False,
        dataloader_num_workers=args.num_workers,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        ddp_find_unused_parameters=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )
    trainer.train()
    final_dir = args.output_dir / "final"
    trainer.save_model(str(final_dir))
    processor.save_pretrained(str(final_dir))
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(
            {
                "manifest": args.manifest.name,
                "rows": len(dataset),
                "epochs": args.epochs,
                "seed": args.seed,
                "assistant_only_supervision": True,
                "model": args.model_name,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
