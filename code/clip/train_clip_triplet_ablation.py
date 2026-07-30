#!/usr/bin/env python3
import argparse
import json
import math
import random
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from eval_clip_retrieval import to_feature_tensor


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                if row.get("rgb_path") and row.get("ir_path") and row.get("caption"):
                    rows.append(row)
    return rows


def open_image(path):
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return Image.new("RGB", (224, 224))


def open_ir_view(row, ir_view):
    if ir_view == "translated":
        return open_image(row["ir_path"])
    rgb = open_image(row["rgb_path"])
    if ir_view == "grayscale":
        return rgb.convert("L").convert("RGB")
    if ir_view == "rgb_duplicate":
        return rgb
    raise ValueError(ir_view)


class TripletDataset(Dataset):
    def __init__(self, path, preprocess=None, ir_view="translated"):
        self.rows = read_jsonl(path)
        self.preprocess = preprocess
        self.ir_view = ir_view

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        if self.preprocess is None:
            return row
        return (
            self.preprocess(open_image(row["rgb_path"])),
            self.preprocess(open_ir_view(row, self.ir_view)),
            row["caption"],
        )


def norm(x):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def contrastive_loss(a, b, logit_scale):
    logits = logit_scale.exp() * norm(a) @ norm(b).t()
    labels = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (
        nn.functional.cross_entropy(logits, labels)
        + nn.functional.cross_entropy(logits.t(), labels)
    )


def combine_losses(setting, rgb_ir_weight, rgb_text, ir_text, rgb_ir):
    if setting == "rgb_text":
        return rgb_text
    if setting == "ir_text":
        return ir_text
    if setting == "dual_text":
        return 0.5 * (rgb_text + ir_text)
    if setting == "trimodal":
        return (rgb_text + ir_text + rgb_ir_weight * rgb_ir) / (2.0 + rgb_ir_weight)
    raise ValueError(setting)


def collate_hf(batch, processor, ir_view):
    rgb = processor(images=[open_image(row["rgb_path"]) for row in batch], return_tensors="pt")
    ir = processor(images=[open_ir_view(row, ir_view) for row in batch], return_tensors="pt")
    text = processor(
        text=[row["caption"] for row in batch], return_tensors="pt", padding=True, truncation=True
    )
    return rgb, ir, text


def load_model(base_ckpt_path, base_model_dir, device):
    if base_model_dir:
        from transformers import CLIPModel, CLIPProcessor

        processor = CLIPProcessor.from_pretrained(base_model_dir, local_files_only=True)
        model = CLIPModel.from_pretrained(base_model_dir, local_files_only=True)
        model.to(device)
        return "hf", model, processor, {"model_dir": base_model_dir, "initialization": "pretrained_base"}

    ckpt = torch.load(base_ckpt_path, map_location="cpu")
    base_args = ckpt.get("args", {})
    if "model_dir" in base_args:
        from transformers import CLIPModel, CLIPProcessor

        processor = CLIPProcessor.from_pretrained(base_args["model_dir"], local_files_only=True)
        model = CLIPModel.from_pretrained(base_args["model_dir"], local_files_only=True)
        model.load_state_dict(ckpt["model"], strict=True)
        model.to(device)
        return "hf", model, processor, base_args

    import open_clip

    model_name = base_args.get("model", "ViT-B-32")
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=None, device=device)
    model.load_state_dict(ckpt["model"], strict=True)
    tokenizer = open_clip.get_tokenizer(model_name)
    return "openclip", model, (preprocess, tokenizer), base_args


def make_loader(
    path,
    mode,
    helper,
    batch_size,
    workers,
    shuffle,
    drop_last,
    generator=None,
    ir_view="translated",
):
    if mode == "hf":
        dataset = TripletDataset(path, ir_view=ir_view)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=workers,
            pin_memory=True,
            drop_last=drop_last,
            collate_fn=lambda batch: collate_hf(batch, helper, ir_view),
            generator=generator,
        )
    else:
        dataset = TripletDataset(path, helper[0], ir_view)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=workers,
            pin_memory=True,
            drop_last=drop_last,
            generator=generator,
        )
    return dataset, loader


def move_dict(batch, device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def encode_batch(mode, model, helper, batch, device):
    if mode == "hf":
        rgb, ir, text = batch
        rgb = move_dict(rgb, device)
        ir = move_dict(ir, device)
        text = move_dict(text, device)
        rgb_features = to_feature_tensor(model.get_image_features(**rgb))
        ir_features = to_feature_tensor(model.get_image_features(**ir))
        text_features = to_feature_tensor(model.get_text_features(**text))
        return rgb_features, ir_features, text_features

    rgb, ir, captions = batch
    rgb = rgb.to(device, non_blocking=True)
    ir = ir.to(device, non_blocking=True)
    tokens = helper[1](list(captions)).to(device, non_blocking=True)
    return model.encode_image(rgb), model.encode_image(ir), model.encode_text(tokens)


def component_losses(model, rgb, ir, text):
    return {
        "rgb_text": contrastive_loss(rgb, text, model.logit_scale),
        "ir_text": contrastive_loss(ir, text, model.logit_scale),
        "rgb_ir": contrastive_loss(rgb, ir, model.logit_scale),
    }


def mean(values):
    return sum(values) / max(1, len(values))


def main():
    ap = argparse.ArgumentParser()
    base = ap.add_mutually_exclusive_group(required=True)
    base.add_argument("--base-ckpt")
    base.add_argument("--base-model-dir")
    ap.add_argument("--train-jsonl", required=True)
    ap.add_argument("--val-jsonl", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument(
        "--loss-setting", required=True, choices=["rgb_text", "ir_text", "dual_text", "trimodal"]
    )
    ap.add_argument("--rgb-ir-weight", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--val-batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=20260710)
    ap.add_argument("--precision", default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument(
        "--ir-view",
        default="translated",
        choices=["translated", "grayscale", "rgb_duplicate"],
        help="Image source for the second visual view.",
    )
    ap.add_argument(
        "--no-save-best",
        action="store_true",
        help="Keep only latest.pt to reduce storage for controlled sweeps.",
    )
    args = ap.parse_args()
    if args.rgb_ir_weight < 0:
        raise SystemExit("--rgb-ir-weight must be non-negative")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    mode, model, helper, base_args = load_model(args.base_ckpt, args.base_model_dir, device)
    train_ds, train_loader = make_loader(
        args.train_jsonl,
        mode,
        helper,
        args.batch_size,
        args.workers,
        True,
        True,
        train_generator,
        args.ir_view,
    )
    val_ds, val_loader = make_loader(
        args.val_jsonl,
        mode,
        helper,
        args.val_batch_size,
        args.workers,
        False,
        False,
        ir_view=args.ir_view,
    )
    if not train_ds or not val_ds:
        raise SystemExit("empty triplet dataset")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_loader)

    def lr_lambda(step):
        if step < args.warmup:
            return max(1e-8, step / max(1, args.warmup))
        progress = (step - args.warmup) / max(1, total_steps - args.warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    use_amp = args.precision in {"fp16", "bf16"} and device.type == "cuda"
    amp_dtype = torch.float16 if args.precision == "fp16" else torch.bfloat16
    scaler = GradScaler(enabled=(args.precision == "fp16" and device.type == "cuda"))
    output = Path(args.output_dir) / args.name
    output.mkdir(parents=True, exist_ok=True)
    meta = {
        **vars(args),
        "train_triplets": len(train_ds),
        "val_triplets": len(val_ds),
        "total_steps": total_steps,
        "base_args": base_args,
    }
    print(json.dumps(meta, ensure_ascii=False), flush=True)

    best = float("inf")
    step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp, dtype=amp_dtype):
                rgb, ir, text = encode_batch(mode, model, helper, batch, device)
                components = component_losses(model, rgb, ir, text)
                loss = combine_losses(args.loss_setting, args.rgb_ir_weight, **components)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            step += 1
            if step % 25 == 0:
                print(
                    f"epoch={epoch} step={step}/{total_steps} loss={loss.item():.4f} "
                    f"rgb_text={components['rgb_text'].item():.4f} "
                    f"ir_text={components['ir_text'].item():.4f} rgb_ir={components['rgb_ir'].item():.4f} "
                    f"lr={scheduler.get_last_lr()[0]:.2e}",
                    flush=True,
                )

        model.eval()
        val_losses = []
        val_components = {"rgb_text": [], "ir_text": [], "rgb_ir": []}
        with torch.inference_mode():
            for batch in val_loader:
                with autocast(enabled=use_amp, dtype=amp_dtype):
                    rgb, ir, text = encode_batch(mode, model, helper, batch, device)
                    components = component_losses(model, rgb, ir, text)
                    val_loss = combine_losses(args.loss_setting, args.rgb_ir_weight, **components)
                val_losses.append(float(val_loss.item()))
                for key, value in components.items():
                    val_components[key].append(float(value.item()))
        val_loss = mean(val_losses)
        val_summary = {key: mean(values) for key, values in val_components.items()}
        print(
            f"epoch={epoch} val_loss={val_loss:.4f} "
            f"val_rgb_text={val_summary['rgb_text']:.4f} "
            f"val_ir_text={val_summary['ir_text']:.4f} val_rgb_ir={val_summary['rgb_ir']:.4f}",
            flush=True,
        )

        save_args = vars(args).copy()
        if "model_dir" in base_args:
            save_args["model_dir"] = base_args["model_dir"]
        if "model" in base_args:
            save_args["model"] = base_args["model"]
        save_args["base_args"] = base_args
        checkpoint = {
            "model": model.state_dict(),
            "epoch": epoch,
            "step": step,
            "val_loss": val_loss,
            "val_components": val_summary,
            "args": save_args,
        }
        torch.save(checkpoint, output / "latest.pt")
        if val_loss < best:
            best = val_loss
            if not args.no_save_best:
                torch.save(checkpoint, output / "best.pt")

    (output / "summary.json").write_text(
        json.dumps({**meta, "best_val_loss": best, "last_step": step}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("DONE", output, flush=True)


if __name__ == "__main__":
    main()
