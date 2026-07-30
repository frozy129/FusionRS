#!/usr/bin/env python3
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_clip_hit_uav_object_presence import (  # noqa: E402
    CLASS_NAMES,
    HFBaseRunner,
    build_samples,
    parse_hf_base_specs,
)
from eval_clip_retrieval import load_runner  # noqa: E402


def label_matrix(samples):
    y = np.zeros((len(samples), len(CLASS_NAMES)), dtype=np.int64)
    for i, sample in enumerate(samples):
        for label in sample["labels"]:
            if label in CLASS_NAMES:
                y[i, CLASS_NAMES.index(label)] = 1
    return y


def load_splits(root, args):
    splits = {}
    audits = {}
    for split in ("train", "val", "test"):
        samples, audit = build_samples(
            root, 0, args.seed, args.annotation_template.format(split=split)
        )
        if not samples:
            raise SystemExit(f"no samples for split {split}")
        splits[split] = samples
        audits[split] = audit
    return splits, audits


def encode_splits(runner, splits, batch_size):
    features = {}
    for split, samples in splits.items():
        print(f" encode {split} {len(samples)}", flush=True)
        features[split] = runner.encode_images([row["image"] for row in samples], batch_size).float().numpy()
    return features


def evaluate_features(model_name, checkpoint_name, checkpoint, features, labels, args):
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import average_precision_score
    except ImportError as exc:
        raise SystemExit("scikit-learn is required for linear probing") from exc

    x_train = features["train"]
    y_train = labels["train"]
    x_val = features["val"]
    y_val = labels["val"]
    x_test = features["test"]
    y_test = labels["test"]
    x_fit = np.concatenate([x_train, x_val], axis=0)
    y_fit = np.concatenate([y_train, y_val], axis=0)

    per_class = []
    aps = []
    for class_index, class_name in enumerate(CLASS_NAMES):
        best = None
        for c_value in args.c_values:
            classifier = LogisticRegression(
                C=c_value,
                class_weight="balanced",
                max_iter=args.max_iter,
                solver="liblinear",
                random_state=args.seed,
            )
            classifier.fit(x_train, y_train[:, class_index])
            val_score = classifier.predict_proba(x_val)[:, 1]
            val_ap = float(average_precision_score(y_val[:, class_index], val_score))
            candidate = (val_ap, -abs(np.log10(c_value)), c_value)
            if best is None or candidate > best:
                best = candidate
        selected_c = best[2]
        classifier = LogisticRegression(
            C=selected_c,
            class_weight="balanced",
            max_iter=args.max_iter,
            solver="liblinear",
            random_state=args.seed,
        )
        classifier.fit(x_fit, y_fit[:, class_index])
        test_score = classifier.predict_proba(x_test)[:, 1]
        test_ap = float(average_precision_score(y_test[:, class_index], test_score))
        aps.append(test_ap)
        per_class.append(
            {
                "class_name": class_name,
                "selected_C": selected_c,
                "val_ap": best[0],
                "test_ap": test_ap,
                "train_positives": int(y_train[:, class_index].sum()),
                "val_positives": int(y_val[:, class_index].sum()),
                "test_positives": int(y_test[:, class_index].sum()),
            }
        )
    return {
        "model": model_name,
        "checkpoint_name": checkpoint_name,
        "checkpoint": str(checkpoint),
        "mAP": float(np.mean(aps)),
        "per_class": per_class,
    }


def evaluate_runner(model_name, checkpoint_name, checkpoint, runner, splits, labels, args, device):
    features = encode_splits(runner, splits, args.image_batch_size)
    del runner
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return evaluate_features(model_name, checkpoint_name, checkpoint, features, labels, args)


def model_dirs(root, checkpoint_name):
    return [
        path
        for path in sorted(Path(root).iterdir())
        if path.is_dir() and (path / checkpoint_name).exists()
    ]


def write_outputs(doc, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "hit_uav_linear_probe.json").write_text(
        json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output_dir / "hit_uav_linear_probe.tsv").open("w", encoding="utf-8") as f:
        f.write("model\tcheckpoint_name\tmAP\n")
        for row in doc["results"]:
            f.write(f"{row['model']}\t{row['checkpoint_name']}\t{100 * row['mAP']:.4f}\n")
    with (output_dir / "hit_uav_linear_probe_per_class.tsv").open("w", encoding="utf-8") as f:
        f.write(
            "model\tcheckpoint_name\tclass_name\tselected_C\tval_ap\ttest_ap\t"
            "train_positives\tval_positives\ttest_positives\n"
        )
        for row in doc["results"]:
            for cls in row["per_class"]:
                f.write(
                    f"{row['model']}\t{row['checkpoint_name']}\t{cls['class_name']}\t"
                    f"{cls['selected_C']}\t{100 * cls['val_ap']:.4f}\t{100 * cls['test_ap']:.4f}\t"
                    f"{cls['train_positives']}\t{cls['val_positives']}\t{cls['test_positives']}\n"
                )
    with (output_dir / "hit_uav_linear_probe.md").open("w", encoding="utf-8") as f:
        f.write("# HIT-UAV frozen-encoder linear probe\n\n")
        f.write("Image-level multi-label classification; this is not object detection.\n\n")
        f.write("| model | checkpoint | test mAP |\n")
        f.write("|---|---:|---:|\n")
        for row in doc["results"]:
            f.write(
                f"| {row['model']} | {row['checkpoint_name']} | {100 * row['mAP']:.4f} |\n"
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hit-uav-root", required=True)
    ap.add_argument("--models-root", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--checkpoint-name", default="latest.pt")
    ap.add_argument("--hf-base-model", action="append", default=[])
    ap.add_argument("--annotation-template", default="annotations/{split}.json")
    ap.add_argument("--c-values", type=float, nargs="+", default=[0.01, 0.1, 1.0, 10.0])
    ap.add_argument("--max-iter", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--precision", default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--image-batch-size", type=int, default=128)
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)
    splits, audits = load_splits(args.hit_uav_root, args)
    labels = {name: label_matrix(samples) for name, samples in splits.items()}
    doc = {
        "title": "HIT-UAV frozen-encoder image-level multi-label linear probe",
        "hit_uav_root": args.hit_uav_root,
        "annotation_template": args.annotation_template,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "audits": audits,
        "results": [],
    }
    output = Path(args.output_dir)

    for name, model_dir in parse_hf_base_specs(args.hf_base_model):
        runner = HFBaseRunner(model_dir, device, args.precision)
        row = evaluate_runner(name, "pretrained-base", model_dir, runner, splits, labels, args, device)
        doc["results"].append(row)
        write_outputs(doc, output)
        print(json.dumps(row, ensure_ascii=False, indent=2), flush=True)

    for model_dir in model_dirs(args.models_root, args.checkpoint_name):
        checkpoint = model_dir / args.checkpoint_name
        runner = load_runner(str(checkpoint), device, args.precision)
        row = evaluate_runner(
            model_dir.name, args.checkpoint_name, checkpoint, runner, splits, labels, args, device
        )
        doc["results"].append(row)
        write_outputs(doc, output)
        print(json.dumps(row, ensure_ascii=False, indent=2), flush=True)
    write_outputs(doc, output)
    print("DONE", args.output_dir, flush=True)


if __name__ == "__main__":
    main()
