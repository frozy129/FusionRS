#!/usr/bin/env python3
import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_clip_retrieval import load_jsonl, load_runner  # noqa: E402


def row_id(row):
    return row.get("sample_id") or row.get("id")


def class_name(row):
    value = row.get("class_name") or row.get("label") or row.get("category")
    if isinstance(value, list):
        value = value[0] if value else ""
    return str(value or "").strip()


def ir_path(row):
    return row.get("ir_path") or row.get("ir_image") or row.get("image")


def label_text(name):
    return name.replace("_", " ").replace("-", " ").strip()


def load_samples(path):
    samples = []
    seen = set()
    dropped = 0
    for row in load_jsonl(path):
        sid = row_id(row)
        cname = class_name(row)
        image = ir_path(row)
        if not (sid and cname and image):
            dropped += 1
            continue
        key = sid
        if key in seen:
            continue
        seen.add(key)
        samples.append({"sample_id": sid, "class_name": cname, "ir_image": image})
    return samples, dropped


def load_classes(paths, samples):
    names = {s["class_name"] for s in samples}
    for path in paths:
        for row in load_jsonl(path):
            cname = class_name(row)
            if cname:
                names.add(cname)
    return sorted(names)


def load_selection(path):
    if not path:
        return {}
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    selected = doc.get("selected") or doc.get("selected_checkpoints") or {}
    out = {}
    for model, item in selected.items():
        if isinstance(item, dict):
            out[model] = item.get("checkpoint_name")
        else:
            out[model] = item
    return {k: v for k, v in out.items() if v}


def pct(x):
    return round(100.0 * x, 4)


def model_dirs(root, selected, checkpoint_name):
    dirs = []
    for path in sorted(Path(root).iterdir()):
        if not path.is_dir():
            continue
        ckpt = selected.get(path.name, checkpoint_name)
        if ckpt and (path / ckpt).exists():
            dirs.append((path, ckpt))
    return dirs


def encode_class_texts(runner, classes, batch_size):
    templates = [
        "a thermal infrared image of a {} scene",
        "an infrared image of {}",
        "a low light thermal view of {}",
    ]
    prompts = []
    for name in classes:
        text = label_text(name)
        prompts.extend([tmpl.format(text) for tmpl in templates])
    feats = runner.encode_texts(prompts, batch_size)
    feats = feats.view(len(classes), len(templates), -1).mean(dim=1)
    return feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def evaluate(model_dir, checkpoint_name, samples, classes, args, device):
    runner = load_runner(str(model_dir / checkpoint_name), device, args.precision)
    images = [s["ir_image"] for s in samples]
    labels = torch.tensor([classes.index(s["class_name"]) for s in samples], dtype=torch.long)
    image_feat = runner.encode_images(images, args.image_batch_size)
    text_feat = encode_class_texts(runner, classes, args.text_batch_size)
    del runner
    if device.type == "cuda":
        torch.cuda.empty_cache()

    hits1 = 0
    hits5 = 0
    confusion = torch.zeros((len(classes), len(classes)), dtype=torch.long)
    for start in range(0, image_feat.shape[0], args.sim_chunk_size):
        end = min(start + args.sim_chunk_size, image_feat.shape[0])
        sims = image_feat[start:end].float() @ text_feat.float().t()
        k = min(5, len(classes))
        topk = sims.topk(k, dim=1).indices.cpu()
        target = labels[start:end].view(-1, 1)
        hits1 += topk[:, :1].eq(target).any(dim=1).sum().item()
        hits5 += topk[:, :k].eq(target).any(dim=1).sum().item()
        pred1 = topk[:, 0]
        for gold, pred in zip(labels[start:end], pred1):
            confusion[int(gold), int(pred)] += 1

    n = len(samples)
    per_class = []
    for idx, name in enumerate(classes):
        total = int(confusion[idx].sum().item())
        correct = int(confusion[idx, idx].item())
        per_class.append(
            {
                "class_name": name,
                "num_samples": total,
                "top1": (correct / total) if total else 0.0,
            }
        )
    return {
        "model": model_dir.name,
        "checkpoint_name": checkpoint_name,
        "checkpoint": str(model_dir / checkpoint_name),
        "num_samples": n,
        "num_classes": len(classes),
        "top1": hits1 / n,
        "top5": hits5 / n,
        "per_class": per_class,
        "confusion": {
            "classes": classes,
            "matrix": confusion.tolist(),
        },
    }


def write_outputs(doc, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "zero_shot_ir_classification.json").write_text(
        json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (out_dir / "zero_shot_ir_classification.tsv").open("w", encoding="utf-8") as f:
        f.write("model\tcheckpoint_name\ttop1\ttop5\tnum_samples\tnum_classes\n")
        for row in doc["results"]:
            f.write(
                f"{row['model']}\t{row['checkpoint_name']}\t{pct(row['top1']):.4f}\t"
                f"{pct(row['top5']):.4f}\t{row['num_samples']}\t{row['num_classes']}\n"
            )
    with (out_dir / "zero_shot_ir_classification_per_class.tsv").open("w", encoding="utf-8") as f:
        f.write("model\tcheckpoint_name\tclass_name\ttop1\tnum_samples\n")
        for row in doc["results"]:
            for cls in row.get("per_class", []):
                f.write(
                    f"{row['model']}\t{row['checkpoint_name']}\t{cls['class_name']}\t"
                    f"{pct(cls['top1']):.4f}\t{cls['num_samples']}\n"
                )
    with (out_dir / "zero_shot_ir_classification.md").open("w", encoding="utf-8") as f:
        f.write("# Zero-shot IR scene classification\n\n")
        f.write(f"- samples: {doc['num_samples']}\n")
        f.write(f"- classes: {doc['num_classes']}\n\n")
        f.write("| model | checkpoint | Top1 | Top5 |\n")
        f.write("|---|---:|---:|---:|\n")
        for row in doc["results"]:
            f.write(f"| {row['model']} | {row['checkpoint_name']} | {pct(row['top1']):.4f} | {pct(row['top5']):.4f} |\n")
        f.write("\n## Per-class Top1\n\n")
        f.write("| model | class | Top1 | samples |\n")
        f.write("|---|---:|---:|---:|\n")
        for row in doc["results"]:
            for cls in row.get("per_class", []):
                f.write(
                    f"| {row['model']} | {cls['class_name']} | {pct(cls['top1']):.4f} | {cls['num_samples']} |\n"
                )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples-jsonl", required=True)
    ap.add_argument("--class-jsonl", action="append", default=[])
    ap.add_argument("--models-root", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--selection-json", default="")
    ap.add_argument("--checkpoint-name", default="latest.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--precision", default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--image-batch-size", type=int, default=256)
    ap.add_argument("--text-batch-size", type=int, default=512)
    ap.add_argument("--sim-chunk-size", type=int, default=1024)
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True

    samples, dropped = load_samples(args.samples_jsonl)
    classes = load_classes(args.class_jsonl, samples)
    selected = load_selection(args.selection_json)
    dirs = model_dirs(args.models_root, selected, args.checkpoint_name)
    if not dirs:
        raise SystemExit("No checkpoints found.")

    doc = {
        "title": "Zero-shot IR scene classification",
        "samples_jsonl": args.samples_jsonl,
        "class_jsonl": args.class_jsonl,
        "models_root": args.models_root,
        "selection_json": args.selection_json,
        "num_samples": len(samples),
        "dropped_samples": dropped,
        "num_classes": len(classes),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": [],
    }
    for model_dir, checkpoint_name in dirs:
        row = evaluate(model_dir, checkpoint_name, samples, classes, args, device)
        doc["results"].append(row)
        write_outputs(doc, Path(args.output_dir))
        print(json.dumps(row, ensure_ascii=False, indent=2), flush=True)
    write_outputs(doc, Path(args.output_dir))
    print("DONE", args.output_dir, flush=True)


if __name__ == "__main__":
    main()
