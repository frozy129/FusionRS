#!/usr/bin/env python3
import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_clip_hit_uav_object_presence import HFBaseRunner, parse_hf_base_specs  # noqa: E402
from eval_clip_retrieval import load_runner, recall_at  # noqa: E402


def parse_split(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            image, mask = line.split(",", 1)
            rows.append({"image": image, "mask": mask})
    return rows


def resolve_dataset_root(path):
    root = Path(path)
    if (root / "color").is_dir() and (root / "thermal8").is_dir():
        return root
    candidates = [
        child.parent
        for child in root.rglob("color")
        if child.is_dir() and (child.parent / "thermal8").is_dir()
    ]
    if len(candidates) != 1:
        raise SystemExit(f"expected one RGB-T dataset root under {root}, found {candidates}")
    return candidates[0]


def load_pairs(dataset_root, splits_root, split):
    rgb_rows = parse_split(Path(splits_root) / f"rgb_{split}.txt")
    ir_rows = parse_split(Path(splits_root) / f"thermal8_{split}.txt")
    rgb_by_mask = {row["mask"]: row["image"] for row in rgb_rows}
    ir_by_mask = {row["mask"]: row["image"] for row in ir_rows}
    common = [row["mask"] for row in rgb_rows if row["mask"] in ir_by_mask]
    if len(common) != len(rgb_rows) or len(common) != len(ir_rows):
        raise SystemExit(
            f"RGB/thermal split mismatch: rgb={len(rgb_rows)} thermal={len(ir_rows)} common={len(common)}"
        )

    rows = []
    missing = []
    for mask in common:
        rgb = dataset_root / rgb_by_mask[mask]
        thermal = dataset_root / ir_by_mask[mask]
        mask_path = dataset_root / mask
        for path in (rgb, thermal, mask_path):
            if not path.is_file():
                missing.append(str(path))
        rows.append(
            {
                "sample_id": Path(mask).stem,
                "rgb": str(rgb),
                "thermal": str(thermal),
                "mask": str(mask_path),
            }
        )
    if missing:
        raise SystemExit(f"missing {len(missing)} official split files; first={missing[0]}")
    return rows


def mean_recall(metrics):
    values = []
    for direction in metrics.values():
        values.extend(direction.values())
    return sum(values) / max(1, len(values))


def evaluate_runner(model_name, checkpoint_name, checkpoint, runner, rows, args, device):
    print(f"MODEL {model_name}", flush=True)
    rgb = runner.encode_images([row["rgb"] for row in rows], args.image_batch_size)
    thermal = runner.encode_images([row["thermal"] for row in rows], args.image_batch_size)
    del runner
    if device.type == "cuda":
        torch.cuda.empty_cache()
    metrics = {
        "RGB->thermal": recall_at(rgb, thermal, args.sim_chunk_size),
        "thermal->RGB": recall_at(thermal, rgb, args.sim_chunk_size),
    }
    return {
        "model": model_name,
        "checkpoint_name": checkpoint_name,
        "checkpoint": str(checkpoint),
        "mean_recall": mean_recall(metrics),
        "metrics": metrics,
    }


def model_dirs(root, checkpoint_name, only_models):
    allowed = set(only_models)
    return [
        path
        for path in sorted(Path(root).iterdir())
        if path.is_dir()
        and (path / checkpoint_name).exists()
        and (not allowed or path.name in allowed)
    ]


def write_outputs(doc, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "caltech_rgbt_retrieval.json").write_text(
        json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output_dir / "caltech_rgbt_retrieval.tsv").open("w", encoding="utf-8") as f:
        f.write("model\tcheckpoint\tmean_recall\tdirection\tR@1\tR@5\tR@10\n")
        for row in doc["results"]:
            for direction, metrics in row["metrics"].items():
                f.write(
                    f"{row['model']}\t{row['checkpoint_name']}\t{100 * row['mean_recall']:.4f}\t"
                    f"{direction}\t{100 * metrics['R@1']:.4f}\t"
                    f"{100 * metrics['R@5']:.4f}\t{100 * metrics['R@10']:.4f}\n"
                )
    with (output_dir / "caltech_rgbt_retrieval.md").open("w", encoding="utf-8") as f:
        f.write("# Caltech Aerial RGB-T paired retrieval\n\n")
        f.write(
            "Exact-pair retrieval on the official synchronized and rectified RGB-thermal test split.\n\n"
        )
        f.write(f"- pairs: {doc['samples']}\n")
        f.write(f"- random R@1: {100 / doc['samples']:.4f}\n\n")
        f.write("| model | checkpoint | Mean R | RGB->T R@1 | T->RGB R@1 |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for row in doc["results"]:
            f.write(
                f"| {row['model']} | {row['checkpoint_name']} | "
                f"{100 * row['mean_recall']:.4f} | "
                f"{100 * row['metrics']['RGB->thermal']['R@1']:.4f} | "
                f"{100 * row['metrics']['thermal->RGB']['R@1']:.4f} |\n"
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--splits-root", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--models-root", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--checkpoint-name", default="latest.pt")
    ap.add_argument("--only-model", action="append", default=[])
    ap.add_argument("--hf-base-model", action="append", default=[])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--precision", default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--image-batch-size", type=int, default=128)
    ap.add_argument("--sim-chunk-size", type=int, default=2048)
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)
    dataset_root = resolve_dataset_root(args.dataset_root)
    rows = load_pairs(dataset_root, args.splits_root, args.split)
    doc = {
        "title": "Caltech Aerial RGB-T exact-pair cross-modal retrieval",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_root": str(dataset_root),
        "splits_root": args.splits_root,
        "split": args.split,
        "samples": len(rows),
        "results": [],
    }
    output = Path(args.output_dir)

    for name, model_dir in parse_hf_base_specs(args.hf_base_model):
        runner = HFBaseRunner(model_dir, device, args.precision)
        row = evaluate_runner(name, "pretrained-base", model_dir, runner, rows, args, device)
        doc["results"].append(row)
        write_outputs(doc, output)
        print(json.dumps(row, ensure_ascii=False, indent=2), flush=True)

    for model_dir in model_dirs(args.models_root, args.checkpoint_name, args.only_model):
        checkpoint = model_dir / args.checkpoint_name
        runner = load_runner(str(checkpoint), device, args.precision)
        row = evaluate_runner(
            model_dir.name, args.checkpoint_name, checkpoint, runner, rows, args, device
        )
        doc["results"].append(row)
        write_outputs(doc, output)
        print(json.dumps(row, ensure_ascii=False, indent=2), flush=True)
    write_outputs(doc, output)


if __name__ == "__main__":
    main()
