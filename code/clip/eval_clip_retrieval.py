#!/usr/bin/env python3
import argparse
import json
import os
import time
from pathlib import Path

import torch
from PIL import Image


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_samples(pair_jsonl):
    by_id = {}
    order = []
    for row in load_jsonl(pair_jsonl):
        sid = row.get("sample_id") or row.get("id")
        if not sid:
            continue
        if sid not in by_id:
            by_id[sid] = {
                "sample_id": sid,
                "caption": row.get("text") or row.get("caption") or "",
                "rgb_image": None,
                "ir_image": None,
            }
            order.append(sid)
        if row.get("text"):
            by_id[sid]["caption"] = row["text"]
        pair_type = row.get("pair_type", "")
        image = row.get("image")
        if pair_type == "ir_caption":
            by_id[sid]["ir_image"] = image
        elif pair_type == "rgb_caption":
            by_id[sid]["rgb_image"] = image
    samples = []
    dropped = 0
    for sid in order:
        item = by_id[sid]
        if item["caption"] and item["rgb_image"] and item["ir_image"]:
            samples.append(item)
        else:
            dropped += 1
    return samples, dropped


def pil_open(path):
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return Image.new("RGB", (224, 224))


def l2norm(x):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def to_feature_tensor(output):
    if torch.is_tensor(output):
        return output
    for attr in ("image_embeds", "text_embeds", "pooler_output", "last_hidden_state"):
        if hasattr(output, attr):
            value = getattr(output, attr)
            if torch.is_tensor(value):
                if attr == "last_hidden_state":
                    return value[:, 0]
                return value
    if isinstance(output, (tuple, list)):
        for value in output:
            if torch.is_tensor(value) and value.ndim == 2:
                return value
    raise TypeError(f"Cannot extract feature tensor from {type(output)!r}")


def recall_at(query, gallery, chunk_size=1024):
    query = query.float()
    gallery = gallery.float()
    target = torch.arange(query.shape[0])
    hits = {1: 0, 5: 0, 10: 0}
    max_k = min(max(hits), gallery.shape[0])
    for start in range(0, query.shape[0], chunk_size):
        end = min(start + chunk_size, query.shape[0])
        sims = query[start:end] @ gallery.t()
        topk = sims.topk(max_k, dim=1).indices.cpu()
        tgt = target[start:end].view(-1, 1)
        eq = topk.eq(tgt)
        for k in hits:
            hits[k] += eq[:, :k].any(dim=1).sum().item()
    n = query.shape[0]
    return {f"R@{k}": hits[k] / n for k in (1, 5, 10)}


def normalized_text_key(text):
    return " ".join(str(text).lower().split())


def recall_at_multi_positive(query, gallery, query_keys, gallery_keys, chunk_size=1024):
    if len(query_keys) != query.shape[0] or len(gallery_keys) != gallery.shape[0]:
        raise ValueError("feature and positive-key counts do not match")
    group_ids = {}
    query_groups = []
    gallery_groups = []
    for key in [*query_keys, *gallery_keys]:
        if key not in group_ids:
            group_ids[key] = len(group_ids)
    query_groups = torch.tensor([group_ids[key] for key in query_keys], dtype=torch.long)
    gallery_groups = torch.tensor([group_ids[key] for key in gallery_keys], dtype=torch.long)
    query = query.float()
    gallery = gallery.float()
    hits = {1: 0, 5: 0, 10: 0}
    max_k = min(max(hits), gallery.shape[0])
    for start in range(0, query.shape[0], chunk_size):
        end = min(start + chunk_size, query.shape[0])
        sims = query[start:end] @ gallery.t()
        topk = sims.topk(max_k, dim=1).indices.cpu()
        matched = gallery_groups[topk].eq(query_groups[start:end].view(-1, 1))
        for k in hits:
            hits[k] += matched[:, :k].any(dim=1).sum().item()
    n = query.shape[0]
    return {f"R@{k}": hits[k] / n for k in (1, 5, 10)}


class HFClipRunner:
    def __init__(self, ckpt_path, device, precision):
        from transformers import CLIPModel, CLIPProcessor

        ckpt = torch.load(ckpt_path, map_location="cpu")
        args = ckpt.get("args", {})
        model_dir = args["model_dir"]
        self.processor = CLIPProcessor.from_pretrained(model_dir, local_files_only=True)
        self.model = CLIPModel.from_pretrained(model_dir, local_files_only=True)
        self.model.load_state_dict(ckpt["model"], strict=True)
        self.model.to(device).eval()
        self.device = device
        self.precision = precision

    def amp(self):
        if self.device.type != "cuda" or self.precision == "fp32":
            return torch.autocast("cpu", enabled=False)
        dtype = torch.bfloat16 if self.precision == "bf16" else torch.float16
        return torch.autocast("cuda", dtype=dtype)

    @torch.inference_mode()
    def encode_images(self, paths, batch_size):
        feats = []
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start : start + batch_size]
            images = [pil_open(p) for p in batch_paths]
            inputs = self.processor(images=images, return_tensors="pt")
            inputs = {k: v.to(self.device, non_blocking=True) for k, v in inputs.items()}
            with self.amp():
                out = self.model.get_image_features(**inputs)
            out = to_feature_tensor(out)
            feats.append(l2norm(out).cpu())
            print(f"  images {min(start + batch_size, len(paths))}/{len(paths)}", flush=True)
        return torch.cat(feats, dim=0)

    @torch.inference_mode()
    def encode_texts(self, texts, batch_size):
        feats = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            inputs = self.processor(text=batch, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device, non_blocking=True) for k, v in inputs.items()}
            with self.amp():
                out = self.model.get_text_features(**inputs)
            out = to_feature_tensor(out)
            feats.append(l2norm(out).cpu())
            print(f"  texts {min(start + batch_size, len(texts))}/{len(texts)}", flush=True)
        return torch.cat(feats, dim=0)


class OpenClipRunner:
    def __init__(self, ckpt_path, device, precision):
        import open_clip

        ckpt = torch.load(ckpt_path, map_location="cpu")
        args = ckpt.get("args", {})
        self.model_name = args.get("model", "ViT-B-32")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.model_name, pretrained=None, device=device
        )
        self.model.load_state_dict(ckpt["model"], strict=True)
        self.model.to(device).eval()
        self.tokenizer = open_clip.get_tokenizer(self.model_name)
        self.device = device
        self.precision = precision

    def amp(self):
        if self.device.type != "cuda" or self.precision == "fp32":
            return torch.autocast("cpu", enabled=False)
        dtype = torch.bfloat16 if self.precision == "bf16" else torch.float16
        return torch.autocast("cuda", dtype=dtype)

    @torch.inference_mode()
    def encode_images(self, paths, batch_size):
        feats = []
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start : start + batch_size]
            images = torch.stack([self.preprocess(pil_open(p)) for p in batch_paths]).to(
                self.device, non_blocking=True
            )
            with self.amp():
                out = self.model.encode_image(images)
            feats.append(l2norm(out).cpu())
            print(f"  images {min(start + batch_size, len(paths))}/{len(paths)}", flush=True)
        return torch.cat(feats, dim=0)

    @torch.inference_mode()
    def encode_texts(self, texts, batch_size):
        feats = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            tokens = self.tokenizer(batch).to(self.device, non_blocking=True)
            with self.amp():
                out = self.model.encode_text(tokens)
            feats.append(l2norm(out).cpu())
            print(f"  texts {min(start + batch_size, len(texts))}/{len(texts)}", flush=True)
        return torch.cat(feats, dim=0)


def load_runner(ckpt_path, device, precision):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    args = ckpt.get("args", {})
    del ckpt
    if "model_dir" in args:
        return HFClipRunner(ckpt_path, device, precision)
    return OpenClipRunner(ckpt_path, device, precision)


def pct(x):
    return round(100.0 * x, 4)


def mean_recall(metrics):
    vals = []
    for direction in metrics.values():
        vals.extend([direction["R@1"], direction["R@5"], direction["R@10"]])
    return sum(vals) / max(1, len(vals))


def evaluate_checkpoint(model_dir, checkpoint_name, samples, args, device):
    captions = [s["caption"] for s in samples]
    ir_paths = [s["ir_image"] for s in samples]
    rgb_paths = [s["rgb_image"] for s in samples]
    ckpt_path = model_dir / checkpoint_name

    print(f"MODEL {model_dir.name} checkpoint={checkpoint_name}", flush=True)
    runner = load_runner(str(ckpt_path), device, args.precision)
    print(" encode IR", flush=True)
    ir_feat = runner.encode_images(ir_paths, args.image_batch_size)
    print(" encode RGB", flush=True)
    rgb_feat = runner.encode_images(rgb_paths, args.image_batch_size)
    print(" encode text", flush=True)
    txt_feat = runner.encode_texts(captions, args.text_batch_size)
    del runner
    if device.type == "cuda":
        torch.cuda.empty_cache()

    text_recall = recall_at
    text_recall_args = ()
    if args.multi_positive_text:
        caption_keys = [normalized_text_key(text) for text in captions]
        text_recall = recall_at_multi_positive
        text_recall_args = (caption_keys, caption_keys)
    metrics = {
        "IR->original_caption": text_recall(
            ir_feat, txt_feat, *text_recall_args, args.sim_chunk_size
        ),
        "original_caption->IR": text_recall(
            txt_feat, ir_feat, *text_recall_args, args.sim_chunk_size
        ),
        "RGB->IR": recall_at(rgb_feat, ir_feat, args.sim_chunk_size),
        "IR->RGB": recall_at(ir_feat, rgb_feat, args.sim_chunk_size),
    }
    return {
        "model": model_dir.name,
        "checkpoint_name": checkpoint_name,
        "checkpoint": str(ckpt_path),
        "text_retrieval_protocol": (
            "normalized-caption multi-positive"
            if args.multi_positive_text
            else "single-positive diagonal"
        ),
        "mean_recall": mean_recall(metrics),
        "metrics": metrics,
    }


def write_outputs(results, output_dir, stem="clip_retrieval_results"):
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{stem}.json"
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    tsv_path = output_dir / f"{stem}.tsv"
    with tsv_path.open("w", encoding="utf-8") as f:
        f.write("model\tcheckpoint_name\tmean_recall\tdirection\tR@1\tR@5\tR@10\n")
        for row in results["results"]:
            for direction, metrics in row["metrics"].items():
                f.write(
                    f"{row['model']}\t{row.get('checkpoint_name', '')}\t"
                    f"{pct(row.get('mean_recall', 0.0)):.4f}\t{direction}\t"
                    f"{pct(metrics['R@1']):.4f}\t{pct(metrics['R@5']):.4f}\t{pct(metrics['R@10']):.4f}\n"
                )

    md_path = output_dir / f"{stem}.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write(f"# {results.get('title', 'CLIP retrieval')}\n\n")
        if "pair_jsonl" in results:
            f.write(f"- pair_jsonl: `{results['pair_jsonl']}`\n")
        if "val_pair_jsonl" in results:
            f.write(f"- val_pair_jsonl: `{results['val_pair_jsonl']}`\n")
        if "test_pair_jsonl" in results:
            f.write(f"- test_pair_jsonl: `{results['test_pair_jsonl']}`\n")
        f.write(f"- samples: {results['num_samples']}\n")
        if "checkpoint_name" in results:
            f.write(f"- checkpoint: `{results['checkpoint_name']}`\n")
        if "baseline_name" in results:
            f.write(f"- baseline: `{results['baseline_name']}`\n")
        f.write("\n| model | checkpoint | Mean Recall | direction | R@1 | R@5 | R@10 |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for row in results["results"]:
            for direction, metrics in row["metrics"].items():
                f.write(
                    f"| {row['model']} | {row.get('checkpoint_name', '')} | "
                    f"{pct(row.get('mean_recall', 0.0)):.4f} | {direction} | {pct(metrics['R@1']):.4f} | "
                    f"{pct(metrics['R@5']):.4f} | {pct(metrics['R@10']):.4f} |\n"
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-jsonl")
    parser.add_argument("--val-pair-jsonl")
    parser.add_argument("--test-pair-jsonl")
    parser.add_argument("--models-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint-name", default="best.pt")
    parser.add_argument("--select-checkpoints", nargs="*", default=["best.pt", "latest.pt", "final.pt"])
    parser.add_argument("--baseline-name", default="580k-only")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--image-batch-size", type=int, default=256)
    parser.add_argument("--text-batch-size", type=int, default=512)
    parser.add_argument("--sim-chunk-size", type=int, default=1024)
    parser.add_argument(
        "--multi-positive-text",
        action="store_true",
        help="Treat samples with identical normalized captions as mutual positives for text retrieval.",
    )
    parser.add_argument("--only-model", action="append", default=[])
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True

    checkpoint_names = list(dict.fromkeys([args.checkpoint_name, *args.select_checkpoints]))
    model_dirs = sorted(
        [p for p in Path(args.models_root).iterdir() if any((p / name).exists() for name in checkpoint_names)]
    )
    if args.only_model:
        keep = set(args.only_model)
        model_dirs = [p for p in model_dirs if p.name in keep]
    if not model_dirs:
        raise SystemExit("No checkpoints found.")

    if args.val_pair_jsonl and args.test_pair_jsonl:
        val_samples, val_dropped = build_samples(args.val_pair_jsonl)
        test_samples, test_dropped = build_samples(args.test_pair_jsonl)
        print(
            json.dumps(
                {
                    "val_samples": len(val_samples),
                    "val_dropped": val_dropped,
                    "test_samples": len(test_samples),
                    "test_dropped": test_dropped,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

        selection = {
            "title": "CLIP checkpoint selection on clean held-out val",
            "val_pair_jsonl": args.val_pair_jsonl,
            "models_root": args.models_root,
            "candidate_checkpoints": args.select_checkpoints,
            "num_samples": len(val_samples),
            "dropped_samples": val_dropped,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "results": [],
            "selected": {},
        }
        selected = {}
        for model_dir in model_dirs:
            candidates = [name for name in args.select_checkpoints if (model_dir / name).exists()]
            best_row = None
            for checkpoint_name in candidates:
                row = evaluate_checkpoint(model_dir, checkpoint_name, val_samples, args, device)
                selection["results"].append(row)
                write_outputs(selection, Path(args.output_dir), "val_checkpoint_selection")
                print(json.dumps(row, ensure_ascii=False, indent=2), flush=True)
                if best_row is None or row["mean_recall"] > best_row["mean_recall"]:
                    best_row = row
            if best_row is None:
                raise RuntimeError(f"No candidate checkpoint for {model_dir.name}")
            selected[model_dir.name] = best_row["checkpoint_name"]
            selection["selected"][model_dir.name] = {
                "checkpoint_name": best_row["checkpoint_name"],
                "mean_recall": best_row["mean_recall"],
            }
            write_outputs(selection, Path(args.output_dir), "val_checkpoint_selection")

        baseline = {
            "title": "580k-only CLIP baseline on clean held-out test",
            "baseline_name": args.baseline_name,
            "test_pair_jsonl": args.test_pair_jsonl,
            "models_root": args.models_root,
            "selected_checkpoints": selected,
            "num_samples": len(test_samples),
            "dropped_samples": test_dropped,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "results": [],
        }
        for model_dir in model_dirs:
            row = evaluate_checkpoint(model_dir, selected[model_dir.name], test_samples, args, device)
            baseline["results"].append(row)
            write_outputs(baseline, Path(args.output_dir), "test_580k_only_baseline")
            print(json.dumps(row, ensure_ascii=False, indent=2), flush=True)

        write_outputs(selection, Path(args.output_dir), "val_checkpoint_selection")
        write_outputs(baseline, Path(args.output_dir), "test_580k_only_baseline")
        print("DONE", args.output_dir, flush=True)
        return

    if not args.pair_jsonl:
        raise SystemExit("--pair-jsonl is required unless --val-pair-jsonl and --test-pair-jsonl are used.")

    samples, dropped = build_samples(args.pair_jsonl)
    print(json.dumps({"samples": len(samples), "dropped": dropped}, ensure_ascii=False), flush=True)

    all_results = {
        "title": "CLIP retrieval",
        "pair_jsonl": args.pair_jsonl,
        "models_root": args.models_root,
        "checkpoint_name": args.checkpoint_name,
        "num_samples": len(samples),
        "dropped_samples": dropped,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": [],
    }

    for model_dir in model_dirs:
        row = evaluate_checkpoint(model_dir, args.checkpoint_name, samples, args, device)
        all_results["results"].append(row)
        write_outputs(all_results, Path(args.output_dir))
        print(json.dumps(row, ensure_ascii=False, indent=2), flush=True)

    write_outputs(all_results, Path(args.output_dir))
    print("DONE", args.output_dir, flush=True)


if __name__ == "__main__":
    main()
