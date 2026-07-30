#!/usr/bin/env python3
"""Build an annotation-derived real-thermal VQA test set from HIT-UAV."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def stable_key(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def region_name(x: float, y: float, width: float, height: float) -> str:
    horizontal = "left" if x < width / 3 else "right" if x > 2 * width / 3 else "center"
    vertical = "upper" if y < height / 3 else "lower" if y > 2 * height / 3 else "middle"
    if horizontal == "center" and vertical == "middle":
        return "center"
    if vertical == "middle":
        return horizontal
    if horizontal == "center":
        return vertical
    return f"{vertical}-{horizontal}"


def unique_dominant_class(present_counts: dict[str, int]) -> str | None:
    if not present_counts:
        return "none"
    maximum = max(present_counts.values())
    winners = [
        name for name, count in present_counts.items() if count == maximum
    ]
    return winners[0] if len(winners) == 1 else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260726)
    args = parser.parse_args()

    coco = json.loads(args.annotations.read_text(encoding="utf-8"))
    annotation_rows = coco.get("annotations") or coco.get("annotation") or []
    categories = {
        int(row["id"]): str(row["name"]).replace("OtherVehicle", "other vehicle").lower()
        for row in coco["categories"]
        if str(row["name"]).lower() != "dontcare"
    }
    images = {
        int(row["id"]): {
            "image_id": int(row["id"]),
            "filename": row.get("file_name") or row.get("filename"),
            "width": int(row["width"]),
            "height": int(row["height"]),
        }
        for row in coco["images"]
    }
    boxes: dict[int, dict[str, list[list[float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in annotation_rows:
        category = categories.get(int(row["category_id"]))
        if category is not None:
            boxes[int(row["image_id"])][category].append(row["bbox"])

    missing = [
        row["filename"]
        for row in images.values()
        if not (args.image_root / row["filename"]).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"{len(missing)} HIT-UAV images missing")

    class_names = tuple(sorted(categories.values()))
    records = []
    question_index = 0

    # Balance yes/no presence questions independently for every class.
    for class_name in class_names:
        positives = [
            image_id
            for image_id in images
            if boxes[image_id].get(class_name)
        ]
        negatives = [
            image_id
            for image_id in images
            if not boxes[image_id].get(class_name)
        ]
        negatives.sort(
            key=lambda image_id: stable_key(
                args.seed, f"presence:{class_name}:{image_id}"
            )
        )
        positives.sort(
            key=lambda image_id: stable_key(
                args.seed, f"presence-positive:{class_name}:{image_id}"
            )
        )
        balanced_count = min(len(positives), len(negatives))
        selected = [(image_id, "yes") for image_id in positives[:balanced_count]] + [
            (image_id, "no") for image_id in negatives[:balanced_count]
        ]
        for image_id, answer in selected:
            question_index += 1
            image = images[image_id]
            records.append(
                {
                    "question_id": f"HITVQA-{question_index:05d}",
                    "image_id": image_id,
                    "image_locator": f"test/{image['filename']}",
                    "question_type": "object_presence",
                    "question": f"Is a {class_name} visible in this thermal image?",
                    "answer": answer,
                    "answer_type": "yes_no",
                    "object_class": class_name,
                    "label_source": "official_bbox_annotation",
                }
            )

    # Count and group-location questions are defined only for positive classes.
    for image_id in sorted(images):
        image = images[image_id]
        present_counts = {}
        for class_name in class_names:
            class_boxes = boxes[image_id].get(class_name, [])
            if not class_boxes:
                continue
            present_counts[class_name] = len(class_boxes)
            question_index += 1
            records.append(
                {
                    "question_id": f"HITVQA-{question_index:05d}",
                    "image_id": image_id,
                    "image_locator": f"test/{image['filename']}",
                    "question_type": "object_count",
                    "question": f"How many {class_name} objects are visible?",
                    "answer": str(len(class_boxes)),
                    "answer_type": "integer",
                    "object_class": class_name,
                    "label_source": "official_bbox_annotation",
                }
            )
            centers = [
                (bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2)
                for bbox in class_boxes
            ]
            mean_x = sum(x for x, _ in centers) / len(centers)
            mean_y = sum(y for _, y in centers) / len(centers)
            question_index += 1
            records.append(
                {
                    "question_id": f"HITVQA-{question_index:05d}",
                    "image_id": image_id,
                    "image_locator": f"test/{image['filename']}",
                    "question_type": "group_location",
                    "question": (
                        f"In which image region is the center of the "
                        f"{class_name} group located?"
                    ),
                    "answer": region_name(
                        mean_x, mean_y, image["width"], image["height"]
                    ),
                    "answer_type": "region",
                    "object_class": class_name,
                    "label_source": "official_bbox_annotation",
                }
            )
        answer = unique_dominant_class(present_counts)
        if answer is None:
            continue
        question_index += 1
        records.append(
            {
                "question_id": f"HITVQA-{question_index:05d}",
                "image_id": image_id,
                "image_locator": f"test/{image['filename']}",
                "question_type": "dominant_object_class",
                "question": "Which annotated object class appears most frequently?",
                "answer": answer,
                "answer_type": "class_name",
                "object_class": None,
                "label_source": "official_bbox_annotation",
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.output, "wt", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    type_counts = Counter(row["question_type"] for row in records)
    yes_no = Counter(
        row["answer"]
        for row in records
        if row["question_type"] == "object_presence"
    )
    summary = {
        "images": len(images),
        "questions": len(records),
        "question_type_counts": dict(sorted(type_counts.items())),
        "presence_answer_counts": dict(sorted(yes_no.items())),
        "classes": list(class_names),
        "split": "official_test",
        "image_bytes_included": False,
        "label_source": "official_bbox_annotation",
    }
    summary_path = args.output.with_suffix(args.output.suffix + ".summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
