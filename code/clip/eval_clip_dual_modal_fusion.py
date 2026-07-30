#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch

from eval_clip_retrieval import build_samples, l2norm, load_runner, recall_at


def parse_alphas(value):
    alphas = sorted({float(x.strip()) for x in value.split(",") if x.strip()})
    if not alphas or any(x < 0.0 or x > 1.0 for x in alphas):
        raise ValueError("fusion alphas must be within [0, 1]")
    return alphas


def mean_metric(metrics):
    values = []
    for direction in metrics.values():
        values.extend(direction.values())
    return sum(values) / max(1, len(values))


def image_text_metrics(image_features, text_features, chunk_size):
    return {
        "image->text": recall_at(image_features, text_features, chunk_size),
        "text->image": recall_at(text_features, image_features, chunk_size),
    }


def legacy_retrieval_metrics(rgb_features, ir_features, text_features, chunk_size):
    ir_metrics = image_text_metrics(ir_features, text_features, chunk_size)
    return {
        "IR->original_caption": ir_metrics["image->text"],
        "original_caption->IR": ir_metrics["text->image"],
        "RGB->IR": recall_at(rgb_features, ir_features, chunk_size),
        "IR->RGB": recall_at(ir_features, rgb_features, chunk_size),
    }


def fuse(rgb_features, ir_features, rgb_weight):
    return l2norm(rgb_weight * rgb_features + (1.0 - rgb_weight) * ir_features)


def encode_split(runner, samples, args):
    rgb_paths = [row["rgb_image"] for row in samples]
    ir_paths = [row["ir_image"] for row in samples]
    captions = [row["caption"] for row in samples]
    return {
        "rgb": runner.encode_images(rgb_paths, args.image_batch_size),
        "ir": runner.encode_images(ir_paths, args.image_batch_size),
        "text": runner.encode_texts(captions, args.text_batch_size),
    }


def evaluate_model(model_dir, checkpoint_name, val_samples, test_samples, args, device, alphas):
    checkpoint = model_dir / checkpoint_name
    print(f"MODEL {model_dir.name} checkpoint={checkpoint_name}", flush=True)
    runner = load_runner(str(checkpoint), device, args.precision)
    print(" encode validation", flush=True)
    val = encode_split(runner, val_samples, args)
    print(" encode test", flush=True)
    test = encode_split(runner, test_samples, args)
    del runner
    if device.type == "cuda":
        torch.cuda.empty_cache()

    alpha_scores = []
    for alpha in alphas:
        metrics = image_text_metrics(fuse(val["rgb"], val["ir"], alpha), val["text"], args.sim_chunk_size)
        alpha_scores.append({"rgb_weight": alpha, "val_mean_recall": mean_metric(metrics)})
    selected = max(alpha_scores, key=lambda row: (row["val_mean_recall"], -abs(row["rgb_weight"] - 0.5)))
    alpha = selected["rgb_weight"]

    val_legacy_metrics = legacy_retrieval_metrics(
        val["rgb"], val["ir"], val["text"], args.sim_chunk_size
    )
    rgb_metrics = image_text_metrics(test["rgb"], test["text"], args.sim_chunk_size)
    ir_metrics = image_text_metrics(test["ir"], test["text"], args.sim_chunk_size)
    fixed_fusion_metrics = image_text_metrics(
        fuse(test["rgb"], test["ir"], 0.5), test["text"], args.sim_chunk_size
    )
    selected_fusion_metrics = image_text_metrics(
        fuse(test["rgb"], test["ir"], alpha), test["text"], args.sim_chunk_size
    )
    legacy_metrics = legacy_retrieval_metrics(
        test["rgb"], test["ir"], test["text"], args.sim_chunk_size
    )
    cross_modal = {
        "RGB->IR": legacy_metrics["RGB->IR"],
        "IR->RGB": legacy_metrics["IR->RGB"],
    }
    settings = {
        "RGB-only": rgb_metrics,
        "IR-only": ir_metrics,
        "RGB+IR-mean": fixed_fusion_metrics,
        "RGB+IR-val-selected": selected_fusion_metrics,
    }
    return {
        "model": model_dir.name,
        "checkpoint_name": checkpoint_name,
        "selected_rgb_weight": alpha,
        "alpha_selection": alpha_scores,
        "validation_legacy_retrieval": {
            "mean_recall": mean_metric(val_legacy_metrics),
            "metrics": val_legacy_metrics,
        },
        "settings": {
            name: {"mean_recall": mean_metric(metrics), "metrics": metrics}
            for name, metrics in settings.items()
        },
        "cross_modal": cross_modal,
        "legacy_retrieval": {
            "mean_recall": mean_metric(legacy_metrics),
            "metrics": legacy_metrics,
        },
    }


def pct(value):
    return 100.0 * value


def write_outputs(doc, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "dual_modal_fusion.json").write_text(
        json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output_dir / "dual_modal_fusion.tsv").open("w", encoding="utf-8") as f:
        f.write("model\tcheckpoint\tsetting\tselected_rgb_weight\tmean_recall\tdirection\tR@1\tR@5\tR@10\n")
        for row in doc["results"]:
            for setting, setting_doc in row["settings"].items():
                for direction, metrics in setting_doc["metrics"].items():
                    f.write(
                        f"{row['model']}\t{row['checkpoint_name']}\t{setting}\t"
                        f"{row['selected_rgb_weight']:.2f}\t{pct(setting_doc['mean_recall']):.4f}\t{direction}\t"
                        f"{pct(metrics['R@1']):.4f}\t{pct(metrics['R@5']):.4f}\t{pct(metrics['R@10']):.4f}\n"
                    )
    with (output_dir / "dual_modal_fusion.md").open("w", encoding="utf-8") as f:
        f.write("# Dual-modal fusion and missing-modality robustness\n\n")
        f.write(f"- validation samples: {doc['val_samples']}\n")
        f.write(f"- test samples: {doc['test_samples']}\n")
        f.write("- fusion: normalized weighted average of RGB and IR embeddings\n")
        f.write("- fusion weight: selected on validation only\n\n")
        f.write("| model | setting | RGB weight | Mean R |\n")
        f.write("|---|---:|---:|---:|\n")
        for row in doc["results"]:
            for setting, setting_doc in row["settings"].items():
                f.write(
                    f"| {row['model']} | {setting} | {row['selected_rgb_weight']:.2f} | "
                    f"{pct(setting_doc['mean_recall']):.4f} |\n"
                )
        f.write("\n## Retrieval metric compatible with the main FusionRS table\n\n")
        f.write("| model | Validation Mean R | Test Mean R |\n")
        f.write("|---|---:|---:|\n")
        for row in doc["results"]:
            f.write(
                f"| {row['model']} | "
                f"{pct(row['validation_legacy_retrieval']['mean_recall']):.4f} | "
                f"{pct(row['legacy_retrieval']['mean_recall']):.4f} |\n"
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-pair-jsonl", required=True)
    ap.add_argument("--test-pair-jsonl", required=True)
    ap.add_argument("--models-root", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--checkpoint-name", default="latest.pt")
    ap.add_argument("--only-model", action="append", default=[])
    ap.add_argument("--fusion-alphas", default="0,0.25,0.5,0.75,1")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--precision", default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--image-batch-size", type=int, default=256)
    ap.add_argument("--text-batch-size", type=int, default=512)
    ap.add_argument("--sim-chunk-size", type=int, default=2048)
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)
    alphas = parse_alphas(args.fusion_alphas)
    val_samples, val_dropped = build_samples(args.val_pair_jsonl)
    test_samples, test_dropped = build_samples(args.test_pair_jsonl)
    if not val_samples or not test_samples:
        raise SystemExit("No complete RGB-IR-text samples found.")

    model_dirs = [
        path
        for path in sorted(Path(args.models_root).iterdir())
        if path.is_dir() and (path / args.checkpoint_name).exists()
    ]
    if args.only_model:
        allowed = set(args.only_model)
        model_dirs = [path for path in model_dirs if path.name in allowed]
    if not model_dirs:
        raise SystemExit("No matching checkpoints found.")

    doc = {
        "title": "Dual-modal embedding fusion and missing-modality robustness",
        "val_pair_jsonl": args.val_pair_jsonl,
        "test_pair_jsonl": args.test_pair_jsonl,
        "val_samples": len(val_samples),
        "test_samples": len(test_samples),
        "val_dropped": val_dropped,
        "test_dropped": test_dropped,
        "fusion_alphas": alphas,
        "results": [],
    }
    for model_dir in model_dirs:
        row = evaluate_model(model_dir, args.checkpoint_name, val_samples, test_samples, args, device, alphas)
        doc["results"].append(row)
        write_outputs(doc, Path(args.output_dir))
        print(json.dumps(row, ensure_ascii=False, indent=2), flush=True)
    write_outputs(doc, Path(args.output_dir))
    print("DONE", args.output_dir, flush=True)


if __name__ == "__main__":
    main()
