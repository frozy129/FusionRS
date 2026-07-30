#!/usr/bin/env python3
import argparse
import json
import random
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_clip_retrieval import l2norm, load_runner, pil_open, to_feature_tensor  # noqa: E402


LABEL_MAP = {
    "person": "person",
    "people": "person",
    "pedestrian": "person",
    "bicycle": "bicycle",
    "bike": "bicycle",
    "car": "car",
    "vehicle": "other vehicle",
    "othervehicle": "other vehicle",
    "other_vehicle": "other vehicle",
    "other vehicle": "other vehicle",
}

CLASS_NAMES = ["person", "bicycle", "car", "other vehicle"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def norm_label(value):
    value = str(value or "").strip().lower().replace("-", " ").replace("_", " ")
    value = " ".join(value.split())
    return LABEL_MAP.get(value, value)


class ImageResolver:
    def __init__(self, root):
        self.root = Path(root)
        self._index = None

    def _build_index(self):
        index = defaultdict(list)
        for path in self.root.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                index[path.name.lower()].append(path)
        self._index = index

    def find(self, name, anchor=None):
        name = str(name or "").strip()
        if not name:
            return None
        name_path = Path(name)
        candidates = []
        if name_path.is_absolute():
            candidates.append(name_path)
        if anchor is not None:
            anchor = Path(anchor)
            candidates.extend(
                [
                    anchor / name,
                    anchor.parent / name,
                    anchor.parent / "images" / name,
                    anchor.parent / "JPEGImages" / name,
                    anchor.parent / "img" / name,
                    anchor.parent.parent / "images" / name,
                    anchor.parent.parent / "JPEGImages" / name,
                    anchor.parent.parent / "img" / name,
                ]
            )
        candidates.extend([self.root / name, self.root / "images" / name, self.root / "img" / name])
        for candidate in candidates:
            if candidate.exists() and candidate.is_file():
                return candidate
        if self._index is None:
            self._build_index()
        matches = self._index.get(Path(name).name.lower(), [])
        return matches[0] if matches else None

    def find_by_stem(self, stem):
        if self._index is None:
            self._build_index()
        stem = str(stem).lower()
        for name, paths in self._index.items():
            if Path(name).stem.lower() == stem:
                return paths[0]
        return None


def labels_from_freeform_json(path):
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    labels = []

    def walk(x):
        if isinstance(x, dict):
            for key in ("class", "name", "category", "label", "type"):
                if key in x:
                    lab = norm_label(x[key])
                    if lab in CLASS_NAMES:
                        labels.append(lab)
            for value in x.values():
                walk(value)
        elif isinstance(x, list):
            for value in x:
                walk(value)

    walk(doc)
    return labels


def labels_from_supervisely(doc):
    labels = []
    for obj in doc.get("objects", []) if isinstance(doc, dict) else []:
        if not isinstance(obj, dict):
            continue
        lab = norm_label(obj.get("classTitle"))
        if lab in CLASS_NAMES:
            labels.append(lab)
    return labels


def coco_rows(path, doc, resolver):
    images = doc.get("images")
    anns = doc.get("annotations", doc.get("annotation", []))
    cats = doc.get("categories", [])
    if not isinstance(images, list) or not isinstance(anns, list):
        return None, {"skipped": 0, "missing_images": 0}

    cat_id_to_name = {}
    for cat in cats if isinstance(cats, list) else []:
        if not isinstance(cat, dict):
            continue
        cat_id = cat.get("id", cat.get("category_id"))
        cat_id_to_name[cat_id] = norm_label(cat.get("name", cat.get("category", cat.get("class"))))

    image_to_labels = defaultdict(list)
    for ann in anns:
        if not isinstance(ann, dict):
            continue
        image_id = ann.get("image_id", ann.get("imageId"))
        lab = None
        if "category_id" in ann:
            lab = cat_id_to_name.get(ann.get("category_id"))
        for key in ("category", "class", "name", "label", "type", "classTitle"):
            if lab is None and key in ann:
                lab = norm_label(ann[key])
        if lab in CLASS_NAMES:
            image_to_labels[image_id].append(lab)

    rows = []
    audit = {"skipped": 0, "missing_images": 0}
    for image in images:
        if not isinstance(image, dict):
            audit["skipped"] += 1
            continue
        image_id = image.get("id", image.get("image_id", image.get("imageId")))
        image_name = image.get("file_name", image.get("filename", image.get("name", image.get("path"))))
        image_path = resolver.find(image_name, path.parent)
        if image_path is None:
            audit["missing_images"] += 1
            continue
        rows.append(
            {
                "sample_id": str(image_id if image_id is not None else Path(image_name).stem),
                "image": str(image_path),
                "labels": sorted(set(image_to_labels.get(image_id, []))),
                "annotation": str(path),
                "format": "coco",
            }
        )
    return rows, audit


def supervisely_row(path, doc, resolver):
    image_name = doc.get("name") if isinstance(doc, dict) else None
    if not image_name and path.name.endswith(".json"):
        image_name = path.name[:-5]
    image = resolver.find(image_name, path.parent)
    labels = labels_from_supervisely(doc)
    if image is None:
        return None, {"missing_images": 1}
    return {
        "sample_id": image.stem,
        "image": str(image),
        "labels": sorted(set(labels)),
        "annotation": str(path),
        "format": "supervisely",
    }, {"missing_images": 0}


def fallback_json_row(path, doc, resolver):
    labels = labels_from_freeform_json(path)
    image = resolver.find(path.name[:-5], path.parent) if path.name.endswith(".json") else None
    if image is None:
        image = resolver.find_by_stem(path.stem)
    if image is None:
        return None, {"missing_images": 1}
    return {
        "sample_id": image.stem,
        "image": str(image),
        "labels": sorted(set(labels)),
        "annotation": str(path),
        "format": "json_fallback",
    }, {"missing_images": 0}


def rows_from_json(path, resolver):
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return [], {"format": "json_parse_error", "skipped": 1, "missing_images": 0}
    rows, audit = coco_rows(path, doc, resolver)
    if rows is not None:
        audit["format"] = "coco"
        return rows, audit
    if isinstance(doc, dict) and "objects" in doc:
        row, audit = supervisely_row(path, doc, resolver)
        audit["format"] = "supervisely"
        return ([row] if row else []), audit
    row, audit = fallback_json_row(path, doc, resolver)
    audit["format"] = "json_fallback"
    return ([row] if row else []), audit


def labels_from_xml(path):
    try:
        root = ET.parse(path).getroot()
    except Exception:
        return [], None
    labels = []
    filename = None
    for elem in root.iter():
        if elem.tag.lower() == "filename" and elem.text:
            filename = elem.text.strip()
        if elem.tag.lower() in {"name", "class", "category", "label"} and elem.text:
            lab = norm_label(elem.text)
            if lab in CLASS_NAMES:
                labels.append(lab)
    return labels, filename


def row_from_xml(path, resolver):
    labels, filename = labels_from_xml(path)
    image = resolver.find(filename, path.parent) if filename else None
    if image is None:
        image = resolver.find_by_stem(path.stem)
    if image is None:
        return None, {"format": "xml", "missing_images": 1, "skipped": 0}
    return {
        "sample_id": image.stem,
        "image": str(image),
        "labels": sorted(set(labels)),
        "annotation": str(path),
        "format": "xml",
    }, {"format": "xml", "missing_images": 0, "skipped": 0}


def build_samples(root, limit, seed, annotation_glob=""):
    root = Path(root)
    resolver = ImageResolver(root)
    if annotation_glob:
        label_files = list(root.rglob(annotation_glob))
    else:
        label_files = list(root.rglob("*.json")) + list(root.rglob("*.xml"))
    rows = []
    seen = set()
    audit = {
        "label_files": len(label_files),
        "missing_images": 0,
        "skipped_annotations": 0,
        "format_counts": Counter(),
        "duplicate_images": 0,
    }
    for lab_path in sorted(label_files):
        if lab_path.suffix.lower() == ".json":
            parsed_rows, info = rows_from_json(lab_path, resolver)
        else:
            row, info = row_from_xml(lab_path, resolver)
            parsed_rows = [row] if row else []
        audit["format_counts"][info.get("format", "unknown")] += 1
        audit["missing_images"] += int(info.get("missing_images", 0))
        audit["skipped_annotations"] += int(info.get("skipped", 0))
        for row in parsed_rows:
            key = str(Path(row["image"]).resolve())
            if key in seen:
                audit["duplicate_images"] += 1
                continue
            seen.add(key)
            rows.append(row)

    rows = sorted(rows, key=lambda row: row["image"])
    if limit and len(rows) > limit:
        rng = random.Random(seed)
        rows = sorted(rng.sample(rows, limit), key=lambda row: row["image"])

    positives = Counter()
    for row in rows:
        positives.update(row["labels"])
    audit.update(
        {
            "num_samples": len(rows),
            "num_positive_samples": sum(1 for row in rows if row["labels"]),
            "num_all_negative_samples": sum(1 for row in rows if not row["labels"]),
            "positives_per_class": {name: positives.get(name, 0) for name in CLASS_NAMES},
            "format_counts": dict(audit["format_counts"]),
        }
    )
    return rows, audit


def encode_label_texts(runner, batch_size):
    templates = [
        "a thermal infrared UAV image containing {}",
        "an aerial thermal image of {}",
        "an infrared drone view with {}",
    ]
    prompts = []
    for name in CLASS_NAMES:
        prompts.extend([tmpl.format(name) for tmpl in templates])
    feats = runner.encode_texts(prompts, batch_size)
    feats = feats.view(len(CLASS_NAMES), len(templates), -1).mean(dim=1)
    return feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-6)


class HFBaseRunner:
    def __init__(self, model_dir, device, precision):
        from transformers import CLIPModel, CLIPProcessor

        self.processor = CLIPProcessor.from_pretrained(model_dir, local_files_only=True)
        self.model = CLIPModel.from_pretrained(model_dir, local_files_only=True).to(device).eval()
        self.device = device
        self.precision = precision

    def amp(self):
        if self.device.type != "cuda" or self.precision == "fp32":
            return torch.autocast("cpu", enabled=False)
        dtype = torch.bfloat16 if self.precision == "bf16" else torch.float16
        return torch.autocast("cuda", dtype=dtype)

    @torch.inference_mode()
    def encode_images(self, paths, batch_size):
        features = []
        for start in range(0, len(paths), batch_size):
            images = [pil_open(path) for path in paths[start : start + batch_size]]
            inputs = self.processor(images=images, return_tensors="pt")
            inputs = {key: value.to(self.device, non_blocking=True) for key, value in inputs.items()}
            with self.amp():
                output = to_feature_tensor(self.model.get_image_features(**inputs))
            features.append(l2norm(output).cpu())
            print(f"  images {min(start + batch_size, len(paths))}/{len(paths)}", flush=True)
        return torch.cat(features, dim=0)

    @torch.inference_mode()
    def encode_texts(self, texts, batch_size):
        features = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            inputs = self.processor(text=batch, return_tensors="pt", padding=True, truncation=True)
            inputs = {key: value.to(self.device, non_blocking=True) for key, value in inputs.items()}
            with self.amp():
                output = to_feature_tensor(self.model.get_text_features(**inputs))
            features.append(l2norm(output).cpu())
            print(f"  texts {min(start + batch_size, len(texts))}/{len(texts)}", flush=True)
        return torch.cat(features, dim=0)


def average_precision(scores, labels):
    order = torch.argsort(scores, descending=True)
    labels = labels[order].float()
    total_pos = labels.sum().item()
    if total_pos <= 0:
        return None
    cumsum = labels.cumsum(0)
    precision = cumsum / torch.arange(1, labels.numel() + 1, dtype=torch.float32)
    return float((precision * labels).sum().item() / total_pos)


def evaluate_runner(model_name, checkpoint_name, checkpoint, runner, samples, args, device):
    image_feats = runner.encode_images([s["image"] for s in samples], args.image_batch_size)
    text_feats = encode_label_texts(runner, args.text_batch_size)
    del runner
    if device.type == "cuda":
        torch.cuda.empty_cache()

    sims = image_feats.float() @ text_feats.float().t()
    y = torch.zeros((len(samples), len(CLASS_NAMES)), dtype=torch.long)
    for i, sample in enumerate(samples):
        for label in sample["labels"]:
            if label in CLASS_NAMES:
                y[i, CLASS_NAMES.index(label)] = 1

    aps = []
    per_class = []
    for j, name in enumerate(CLASS_NAMES):
        ap = average_precision(sims[:, j], y[:, j])
        positives = int(y[:, j].sum().item())
        if ap is not None:
            aps.append(ap)
        per_class.append({"class_name": name, "ap": ap, "positives": positives})

    top1 = sims.argmax(dim=1)
    top1_hit = 0
    for i, pred in enumerate(top1.tolist()):
        top1_hit += int(y[i, pred].item() > 0)
    return {
        "model": model_name,
        "checkpoint_name": checkpoint_name,
        "checkpoint": str(checkpoint),
        "num_samples": len(samples),
        "classes": CLASS_NAMES,
        "mAP": sum(aps) / len(aps) if aps else 0.0,
        "top1_presence_hit": top1_hit / max(1, len(samples)),
        "per_class": per_class,
    }


def evaluate_model(model_dir, checkpoint_name, samples, args, device):
    checkpoint = model_dir / checkpoint_name
    runner = load_runner(str(checkpoint), device, args.precision)
    return evaluate_runner(model_dir.name, checkpoint_name, checkpoint, runner, samples, args, device)


def parse_hf_base_specs(values):
    specs = []
    for value in values:
        if "=" not in value:
            raise ValueError("--hf-base-model must be NAME=MODEL_DIR")
        name, model_dir = value.split("=", 1)
        if not name or not model_dir:
            raise ValueError("--hf-base-model must be NAME=MODEL_DIR")
        specs.append((name, model_dir))
    return specs


def model_dirs(root, checkpoint_name):
    return [p for p in sorted(Path(root).iterdir()) if p.is_dir() and (p / checkpoint_name).exists()]


def write_outputs(doc, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "hit_uav_object_presence.json").write_text(
        json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (out_dir / "hit_uav_object_presence.tsv").open("w", encoding="utf-8") as f:
        f.write("model\tcheckpoint_name\tmAP\ttop1_presence_hit\tnum_samples\n")
        for row in doc["results"]:
            f.write(
                f"{row['model']}\t{row['checkpoint_name']}\t{100 * row['mAP']:.4f}\t"
                f"{100 * row['top1_presence_hit']:.4f}\t{row['num_samples']}\n"
            )
    with (out_dir / "hit_uav_object_presence_per_class.tsv").open("w", encoding="utf-8") as f:
        f.write("model\tcheckpoint_name\tclass_name\tap\tpositives\n")
        for row in doc["results"]:
            for cls in row["per_class"]:
                ap = "" if cls["ap"] is None else f"{100 * cls['ap']:.4f}"
                f.write(f"{row['model']}\t{row['checkpoint_name']}\t{cls['class_name']}\t{ap}\t{cls['positives']}\n")
    with (out_dir / "hit_uav_object_presence.md").open("w", encoding="utf-8") as f:
        f.write("# HIT-UAV Object Presence\n\n")
        f.write(f"- samples: {doc['num_samples']}\n")
        f.write(f"- positive samples: {doc['audit']['num_positive_samples']}\n")
        f.write(f"- all-negative samples: {doc['audit']['num_all_negative_samples']}\n")
        f.write(f"- classes: {', '.join(CLASS_NAMES)}\n\n")
        f.write("| model | checkpoint | mAP | Top1 presence hit |\n")
        f.write("|---|---:|---:|---:|\n")
        for row in doc["results"]:
            f.write(
                f"| {row['model']} | {row['checkpoint_name']} | {100 * row['mAP']:.4f} | "
                f"{100 * row['top1_presence_hit']:.4f} |\n"
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hit-uav-root", required=True)
    ap.add_argument("--models-root", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--checkpoint-name", default="latest.pt")
    ap.add_argument(
        "--hf-base-model",
        action="append",
        default=[],
        help="Optional pretrained baseline in NAME=LOCAL_MODEL_DIR form; may be repeated.",
    )
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--annotation-glob",
        default="",
        help="Optional annotation file glob relative to the dataset root, e.g. annotations/test.json.",
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--precision", default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--image-batch-size", type=int, default=128)
    ap.add_argument("--text-batch-size", type=int, default=128)
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)
    samples, audit = build_samples(args.hit_uav_root, args.limit, args.seed, args.annotation_glob)
    if not samples:
        raise SystemExit("No HIT-UAV samples with labels and images found.")

    doc = {
        "title": "HIT-UAV real thermal object-presence zero-shot evaluation",
        "hit_uav_root": args.hit_uav_root,
        "models_root": args.models_root,
        "checkpoint_name": args.checkpoint_name,
        "annotation_glob": args.annotation_glob,
        "num_samples": len(samples),
        "audit": audit,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": [],
    }
    out = Path(args.output_dir)
    (out / "samples.jsonl").parent.mkdir(parents=True, exist_ok=True)
    with (out / "samples.jsonl").open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    for name, model_dir in parse_hf_base_specs(args.hf_base_model):
        runner = HFBaseRunner(model_dir, device, args.precision)
        row = evaluate_runner(name, "pretrained-base", model_dir, runner, samples, args, device)
        doc["results"].append(row)
        write_outputs(doc, out)
        print(json.dumps(row, ensure_ascii=False, indent=2), flush=True)

    for model_dir in model_dirs(args.models_root, args.checkpoint_name):
        row = evaluate_model(model_dir, args.checkpoint_name, samples, args, device)
        doc["results"].append(row)
        write_outputs(doc, out)
        print(json.dumps(row, ensure_ascii=False, indent=2), flush=True)
    write_outputs(doc, out)
    print("DONE", args.output_dir, flush=True)


if __name__ == "__main__":
    main()
