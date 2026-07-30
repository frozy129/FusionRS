#!/usr/bin/env python3
"""Run FusionRS benchmark with LLaVA-repository model implementations."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

import torch
from PIL import Image

try:
    import transformers.modeling_utils as _modeling_utils
    import transformers.utils.import_utils as _import_utils

    _modeling_utils.check_torch_load_is_safe = lambda: None
    _import_utils.check_torch_load_is_safe = lambda: None
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_fusionrs_benchmark_qwen import build_records, load_done  # noqa: E402


LOCAL_CLIP_TOWER = os.environ.get(
    "FUSIONRS_CLIP_TOWER", "openai/clip-vit-large-patch14-336"
)
LOCAL_H2RS_TOWER = os.environ.get(
    "FUSIONRS_H2RS_TOWER", "H2RSVLM-VHM-weights/vhm_7b_pretrain_vit"
)
LLAVA_REPO = os.environ.get("FUSIONRS_LLAVA_REPO", "LLaVA")
H2RS_REPO = os.environ.get("FUSIONRS_H2RS_REPO", "H2RSVLM-VHM")


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def patch_config(model_path: str, model_kind: str) -> None:
    config_path = Path(model_path) / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["mm_vision_tower"] = (
        LOCAL_H2RS_TOWER if model_kind == "h2rsvlm" else LOCAL_CLIP_TOWER
    )
    if model_kind == "geochat":
        config["model_type"] = "llava_llama"
        config["architectures"] = ["LlavaLlamaForCausalLM"]
    config_path.write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


class LlavaRunner:
    def __init__(self, model_path: str, model_kind: str) -> None:
        patch_config(model_path, model_kind)
        sys.path.insert(0, LLAVA_REPO)
        from llava.constants import (
            DEFAULT_IMAGE_TOKEN,
            DEFAULT_IM_END_TOKEN,
            DEFAULT_IM_START_TOKEN,
            IMAGE_TOKEN_INDEX,
        )
        from llava.conversation import conv_templates
        from llava.mm_utils import process_images, tokenizer_image_token
        from llava.model.builder import load_pretrained_model
        from llava.utils import disable_torch_init

        disable_torch_init()
        model_name = (
            "llava-geochat" if model_kind == "geochat" else Path(model_path).name
        )
        tokenizer, model, image_processor, _ = load_pretrained_model(
            model_path,
            None,
            model_name,
            device_map="cuda:0",
            device="cuda",
        )
        self.tokenizer = tokenizer
        self.model = model.eval()
        self.image_processor = image_processor
        self.process_images = process_images
        self.tokenizer_image_token = tokenizer_image_token
        self.conv_templates = conv_templates
        self.image_token_index = IMAGE_TOKEN_INDEX
        self.default_image_token = DEFAULT_IMAGE_TOKEN
        self.default_im_start_token = DEFAULT_IM_START_TOKEN
        self.default_im_end_token = DEFAULT_IM_END_TOKEN
        self.conv_mode = "llava_v1"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def infer(self, image_path: str, prompt: str, max_new_tokens: int) -> str:
        image = Image.open(image_path).convert("RGB")
        image_sizes = [image.size]
        image_tensor = self.process_images(
            [image], self.image_processor, self.model.config
        )
        if isinstance(image_tensor, list):
            image_tensor = torch.stack(image_tensor)
        image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)

        image_token = self.default_image_token
        if getattr(self.model.config, "mm_use_im_start_end", False):
            image_token = (
                self.default_im_start_token
                + image_token
                + self.default_im_end_token
            )
        conversation = self.conv_templates[self.conv_mode].copy()
        conversation.append_message(
            conversation.roles[0], image_token + "\n" + prompt.strip()
        )
        conversation.append_message(conversation.roles[1], None)
        input_ids = self.tokenizer_image_token(
            conversation.get_prompt(),
            self.tokenizer,
            self.image_token_index,
            return_tensors="pt",
        ).unsqueeze(0).to(self.model.device)
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=image_sizes,
                do_sample=False,
                num_beams=1,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        prediction = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return clean(prediction)


class H2RSRunner:
    def __init__(self, model_path: str) -> None:
        patch_config(model_path, "h2rsvlm")
        sys.path.insert(0, H2RS_REPO)
        from vhm.constants import (
            DEFAULT_IMAGE_TOKEN,
            DEFAULT_IM_END_TOKEN,
            DEFAULT_IM_START_TOKEN,
        )
        from vhm.conversation import conv_templates
        from vhm.mm_utils import (
            get_model_name_from_path,
            process_images,
            tokenizer_image_token,
        )
        from vhm.model.builder import load_pretrained_model

        model_name = get_model_name_from_path(model_path)
        tokenizer, model, image_processor, _ = load_pretrained_model(
            model_path,
            None,
            model_name,
            device_map="cuda:0",
            device="cuda",
        )
        self.tokenizer = tokenizer
        self.model = model.eval()
        self.image_processor = image_processor
        self.process_images = process_images
        self.tokenizer_image_token = tokenizer_image_token
        self.conv_templates = conv_templates
        self.default_image_token = DEFAULT_IMAGE_TOKEN
        self.default_im_start_token = DEFAULT_IM_START_TOKEN
        self.default_im_end_token = DEFAULT_IM_END_TOKEN
        self.conv_mode = "v1"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def infer(self, image_path: str, prompt: str, max_new_tokens: int) -> str:
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.process_images(
            [image], self.image_processor, self.model.config
        )
        if isinstance(image_tensor, list):
            image_tensor = torch.stack(image_tensor)
        device = next(self.model.parameters()).device
        image_tensor = image_tensor.to(device=device, dtype=self.model.dtype)
        image_token = self.default_image_token
        if getattr(self.model.config, "mm_use_im_start_end", False):
            image_token = (
                self.default_im_start_token
                + image_token
                + self.default_im_end_token
            )
        conversation = self.conv_templates[self.conv_mode].copy()
        conversation.append_message(
            conversation.roles[0], image_token + "\n" + prompt.strip()
        )
        conversation.append_message(conversation.roles[1], None)
        input_ids = self.tokenizer_image_token(
            conversation.get_prompt(),
            self.tokenizer,
            return_tensors="pt",
        ).unsqueeze(0).to(device)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                images=image_tensor,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=False,
            )
        prediction = self.tokenizer.decode(
            output_ids[0, input_ids.shape[1] :], skip_special_tokens=True
        )
        return clean(prediction)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-kind",
        required=True,
        choices=("llava15", "llava16", "geochat", "h2rsvlm"),
    )
    parser.add_argument("--model-path", required=True)
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
    runner = (
        H2RSRunner(args.model_path)
        if args.model_kind == "h2rsvlm"
        else LlavaRunner(args.model_path, args.model_kind)
    )

    started = time.time()
    errors = 0
    with args.output.open("a", encoding="utf-8") as handle:
        for index, row in enumerate(records, 1):
            if row["record_key"] in done:
                continue
            prediction = ""
            error = ""
            for attempt in range(1, args.retries + 2):
                try:
                    prediction = runner.infer(
                        row["image_path"], row["prompt"], row["max_new_tokens"]
                    )
                    if not prediction:
                        raise RuntimeError("empty prediction")
                    error = ""
                    break
                except Exception as exc:
                    error = (
                        f"attempt={attempt}: {exc!r}\n"
                        f"{traceback.format_exc()}"
                    )
            status = "success" if prediction else "error"
            errors += status == "error"
            handle.write(
                json.dumps(
                    {
                        **row,
                        "model_tag": args.model_tag,
                        "prediction": prediction,
                        "status": status,
                        "error": error,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
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
        "expected": len(expected),
        "successful": len(expected & successful),
        "missing_or_error": len(missing),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    if missing:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
