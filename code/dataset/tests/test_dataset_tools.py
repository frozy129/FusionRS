from __future__ import annotations

import importlib.util
import csv
import gzip
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = load_script("build_benchmark_candidates")
release_builder = load_script("build_release_manifest")
verifier = load_script("verify_dataset_release")
reconstructor = load_script("reconstruct")
vqa_audit = load_script("summarize_vqa_human_audit")
hit_vqa_builder = load_script("build_hit_uav_vqa")


class DatasetToolTests(unittest.TestCase):
    def test_hit_uav_dominant_class_skips_ties(self):
        self.assertEqual(
            hit_vqa_builder.unique_dominant_class({"car": 3, "person": 1}),
            "car",
        )
        self.assertIsNone(
            hit_vqa_builder.unique_dominant_class({"car": 3, "person": 3})
        )
        self.assertEqual(
            hit_vqa_builder.unique_dominant_class({}),
            "none",
        )

    def test_source_balanced_selection(self):
        grouped = {}
        for source in builder.SOURCES:
            for index in range(3):
                sample_id = f"{source}_{index:03d}"
                grouped[sample_id] = {
                    "sample_id": sample_id,
                    "dataset": source,
                    "class_name": f"class_{index % 2}",
                    "joint_group_id": f"joint_{source}_{index:03d}",
                }
        selected = builder.select_source_balanced(grouped, per_source=2, seed=7)
        self.assertEqual(len(selected), 10)
        self.assertEqual(len({row["sample_id"] for row in selected}), 10)
        self.assertEqual(len({row["joint_group_id"] for row in selected}), 10)
        for source in builder.SOURCES:
            self.assertEqual(
                sum(row["dataset"] == source for row in selected), 2
            )

    def test_release_manifest_removes_absolute_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            manifest = tmp_path / "manifest.jsonl"
            clusters = tmp_path / "clusters.jsonl"
            output = tmp_path / "release.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "sample_id": "RSICD_00001",
                        "dataset": "RSICD",
                        "rgb_path": (
                            "/private/server/datasets/RSICD_unpacked/"
                            "images/rsicd_images/airport_1.jpg"
                        ),
                        "benchmark_split": "train",
                        "class_name": "airport",
                        "caption": "An airport.",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            clusters.write_text("", encoding="utf-8")
            components, canonical = release_builder.load_components(clusters)
            self.assertEqual(components, {})
            self.assertEqual(canonical, {})
            self.assertEqual(
                release_builder.source_locator(
                    "RSICD",
                    "/private/server/datasets/RSICD_unpacked/"
                    "images/rsicd_images/airport_1.jpg",
                ),
                "images/rsicd_images/airport_1.jpg",
            )

    def test_release_manifest_merges_rgb_and_ir_components(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rgb = root / "rgb.jsonl"
            ir = root / "ir.jsonl"
            rgb.write_text(
                json.dumps(
                    {
                        "members": [
                            {"sample_id": "S1"},
                            {"sample_id": "S2"},
                        ]
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            ir.write_text(
                json.dumps(
                    {
                        "members": [
                            {"sample_id": "S2"},
                            {"sample_id": "S3"},
                        ]
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            components, canonical = release_builder.load_components([rgb, ir])
            self.assertEqual(len(set(components.values())), 1)
            self.assertEqual(canonical, {"S1": "S1", "S2": "S1", "S3": "S1"})

    def test_unique_representatives_are_deduplicated(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pairs.jsonl"
            rows = [
                {"sample_id": "RSICD_1", "pair_type": "ir_caption"},
                {"sample_id": "RSICD_1", "pair_type": "rgb_caption"},
                {"sample_id": "RSICD_2", "pair_type": "ir_caption"},
            ]
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            self.assertEqual(
                release_builder.load_unique_representatives([path]),
                {"RSICD_1", "RSICD_2"},
            )

    def test_index_verifier_rejects_machine_path(self):
        version = {
            "expected_counts": {
                "manifest": 1,
                "train": 1,
                "validation": 0,
                "test": 0,
                "validation_unique_representatives": 0,
                "test_unique_representatives": 0,
            }
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "index.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "sample_id": "RSICD_1",
                        "split": "train",
                        "source_dataset": "RSICD",
                        "source_locator": "/root/private.jpg",
                        "phash4_component_id": "singleton_RSICD_1",
                        "is_eval_representative": False,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            errors = verifier.verify_index_manifest(path, version)
            self.assertTrue(any("machine-specific path" in error for error in errors))

    def test_index_verifier_rejects_caption_cross_split(self):
        caption_a = "An Airport!"
        caption_b = "an airport"

        def row(sample_id, split, caption):
            return {
                "schema_version": "fusionrs-index-v1",
                "sample_id": sample_id,
                "source_dataset": "RSICD",
                "source_record_id": sample_id,
                "source_locator": f"images/{sample_id}.jpg",
                "source_rights_note": "index-only",
                "split": split,
                "class_name": None,
                "caption": caption,
                "caption_usable": True,
                "normalized_caption_key": verifier.normalized_caption_key(caption),
                "joint_group_id": (
                    "joint-group:"
                    + hashlib.sha256(sample_id.encode()).hexdigest()[:24]
                ),
                "phash4_component_id": f"singleton_{sample_id}",
                "canonical_component_representative": sample_id,
                "is_canonical_component_representative": True,
                "is_eval_representative": split != "train",
                "vlm_train_used": False,
                "prior_reused_protected": False,
                "translation": {
                    "model": "DiffV2IR",
                    "output_key": f"RSICD/{sample_id}.png",
                    "redistribution": "conditional_on_upstream_rights",
                },
            }

        version = {
            "heldout_minimum_per_source": 0,
            "expected_counts": {
                "manifest": 2,
                "train": 1,
                "validation": 0,
                "test": 1,
                "validation_unique_representatives": 0,
                "test_unique_representatives": 1,
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "index.jsonl"
            path.write_text(
                json.dumps(row("RSICD_1", "train", caption_a)) + "\n"
                + json.dumps(row("RSICD_2", "test", caption_b)) + "\n",
                encoding="utf-8",
            )
            errors = verifier.verify_index_manifest(path, version)
        self.assertTrue(any("caption key crosses" in error for error in errors))

    def test_index_verifier_rejects_joint_group_cross_split(self):
        def row(sample_id, split):
            return {
                "schema_version": "fusionrs-index-v1",
                "sample_id": sample_id,
                "source_dataset": "RSICD",
                "source_record_id": sample_id,
                "source_locator": f"images/{sample_id}.jpg",
                "source_rights_note": "index-only",
                "split": split,
                "class_name": None,
                "caption": None,
                "caption_usable": False,
                "normalized_caption_key": None,
                "joint_group_id": "joint-group:" + "a" * 24,
                "phash4_component_id": f"singleton_{sample_id}",
                "canonical_component_representative": sample_id,
                "is_canonical_component_representative": True,
                "is_eval_representative": split != "train",
                "vlm_train_used": False,
                "prior_reused_protected": False,
                "translation": {
                    "model": "DiffV2IR",
                    "output_key": f"RSICD/{sample_id}.png",
                    "redistribution": "conditional_on_upstream_rights",
                },
            }

        version = {
            "heldout_minimum_per_source": 0,
            "expected_counts": {
                "manifest": 2,
                "train": 1,
                "validation": 0,
                "test": 1,
                "validation_unique_representatives": 0,
                "test_unique_representatives": 1,
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "index.jsonl"
            path.write_text(
                json.dumps(row("RSICD_1", "train")) + "\n"
                + json.dumps(row("RSICD_2", "test")) + "\n",
                encoding="utf-8",
            )
            errors = verifier.verify_index_manifest(path, version)
        self.assertTrue(any("joint group" in error for error in errors))

    def test_reconstruction_materializes_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.jpg"
            target = root / "workspace" / "rgb" / "RSICD" / "RSICD_1.jpg"
            source.write_bytes(b"fixture")
            reconstructor.materialize(source, target, "copy")
            self.assertEqual(target.read_bytes(), b"fixture")

    def test_vqa_audit_requires_majority_full_acceptance(self):
        fields = [
            "question_id",
            "image",
            "question_type",
            "question",
            "proposed_answer",
            *vqa_audit.RATING_FIELDS,
            "comments",
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ratings = (("1", "1", "1"), ("1", "1", "1"), ("1", "0", "1"))
            for annotator, values in enumerate(ratings, 1):
                with (root / f"annotator_{annotator}.csv").open(
                    "w", encoding="utf-8", newline=""
                ) as handle:
                    writer = csv.DictWriter(handle, fieldnames=fields)
                    writer.writeheader()
                    writer.writerow(
                        {
                            "question_id": "HITVQA-00001",
                            "image": "images/HIT-1.jpg",
                            "question_type": "object_count",
                            "question": "How many cars are visible?",
                            "proposed_answer": "2",
                            **dict(zip(vqa_audit.RATING_FIELDS, values)),
                            "comments": "",
                        }
                    )
            forms = [
                vqa_audit.load_form(root / f"annotator_{index}.csv")
                for index in range(1, 4)
            ]
            summary = vqa_audit.summarize(forms)
            self.assertEqual(summary["complete_questions"], 1)
            self.assertEqual(summary["accepted_questions"], 1)
            self.assertEqual(summary["unanimous_decisions"], 0)

    def test_real_thermal_vqa_verifier_detects_unbalanced_presence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = (
                root / "manifests" / "hit_uav_real_thermal_vqa_test.jsonl.gz"
            )
            manifest.parent.mkdir()
            row = {
                "question_id": "HITVQA-00001",
                "image_id": 1,
                "image_locator": "test/1.jpg",
                "question_type": "object_presence",
                "question": "Is a car visible?",
                "answer": "yes",
                "label_source": "official_bbox_annotation",
            }
            with gzip.open(manifest, "wt", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
            manifest.with_suffix(manifest.suffix + ".summary.json").write_text(
                json.dumps(
                    {
                        "questions": 1,
                        "images": 1,
                        "question_type_counts": {"object_presence": 1},
                        "presence_answer_counts": {"yes": 1},
                    }
                ),
                encoding="utf-8",
            )
            errors = verifier.verify_real_thermal_vqa(root)
            self.assertTrue(
                any("not balanced" in error for error in errors)
            )


if __name__ == "__main__":
    unittest.main()
