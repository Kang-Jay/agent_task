from __future__ import annotations

import copy
import unittest
from pathlib import Path

from src.evaluation.manifest import (
    ManifestValidationError,
    load_manifest,
    parse_manifest,
)


def valid_manifest() -> dict:
    pose = {
        "position": {"x": 0.0, "y": 0.9, "z": 1.5},
        "rotation": {"x": 0.0, "y": 90.0, "z": 0.0},
        "horizon": 0.0,
        "standing": True,
    }
    task = {
        "instruction": "Find the television",
        "task_type": "visual_search",
        "target": {"object_type": "Television"},
        "required_actions": ["STOP"],
        "allows_approximate_success": False,
    }
    reference = {
        "optimal_path_length_meters": 0.5,
        "source": "AI2-THOR GetShortestPath",
        "allowed_error_meters": 0.05,
    }
    episodes = []
    for pair_id, scene in (("pair-a", "FloorPlan211"), ("pair-b", "FloorPlan212")):
        for group in ("oracle", "non_oracle"):
            episodes.append(
                {
                    "episode_id": f"{pair_id}-{group}",
                    "pair_id": pair_id,
                    "group": group,
                    "split": "validation",
                    "scene": scene,
                    "seed": 123,
                    "initial_pose": copy.deepcopy(pose),
                    "task": copy.deepcopy(task),
                    "reference": copy.deepcopy(reference),
                    "result_file": f"validation/{group}/{pair_id}.json",
                }
            )
    return {
        "schema_version": "1.0",
        "benchmark_id": "unit-test",
        "dataset_version": "unit-test",
        "inference_only": True,
        "description": "Unit-test manifest for evaluation schema validation.",
        "protocol": {
            "required_groups": ["oracle", "non_oracle"],
            "minimum_scene_count": 2,
        },
        "episodes": episodes,
    }


class EvaluationManifestTests(unittest.TestCase):
    def test_valid_paired_manifest_has_deterministic_order(self) -> None:
        manifest = parse_manifest(valid_manifest())
        ordered = [episode.episode_id for episode in manifest.ordered_episodes()]
        self.assertEqual(
            ordered,
            [
                "pair-a-oracle",
                "pair-a-non_oracle",
                "pair-b-oracle",
                "pair-b-non_oracle",
            ],
        )

    def test_duplicate_episode_id_is_rejected(self) -> None:
        raw = valid_manifest()
        raw["episodes"][1]["episode_id"] = raw["episodes"][0]["episode_id"]
        with self.assertRaisesRegex(ManifestValidationError, "episode_id"):
            parse_manifest(raw)

    def test_pair_mismatch_is_rejected(self) -> None:
        raw = valid_manifest()
        raw["episodes"][1]["seed"] = 999
        with self.assertRaisesRegex(ManifestValidationError, "mismatched"):
            parse_manifest(raw)

    def test_missing_group_is_rejected(self) -> None:
        raw = valid_manifest()
        raw["episodes"] = [
            episode
            for episode in raw["episodes"]
            if episode["episode_id"] != "pair-b-non_oracle"
        ]
        with self.assertRaisesRegex(ManifestValidationError, "exactly one episode"):
            parse_manifest(raw)

    def test_path_traversal_is_rejected(self) -> None:
        raw = valid_manifest()
        raw["episodes"][0]["result_file"] = "../outside.json"
        with self.assertRaisesRegex(ManifestValidationError, "relative JSON path"):
            parse_manifest(raw)

    def test_windows_drive_result_file_is_rejected(self) -> None:
        raw = valid_manifest()
        raw["episodes"][0]["result_file"] = "C:/escape.json"
        with self.assertRaisesRegex(ManifestValidationError, "safe relative JSON path"):
            parse_manifest(raw)

    def test_result_file_must_match_split_and_group(self) -> None:
        raw = valid_manifest()
        raw["episodes"][0]["result_file"] = "validation/non_oracle/wrong.json"
        with self.assertRaisesRegex(ManifestValidationError, "<split>/<group>"):
            parse_manifest(raw)

    def test_duplicate_result_file_is_rejected(self) -> None:
        raw = valid_manifest()
        raw["episodes"][1]["result_file"] = raw["episodes"][0]["result_file"]
        with self.assertRaisesRegex(ManifestValidationError, "result_file"):
            parse_manifest(raw)

    def test_non_finite_pose_is_rejected(self) -> None:
        raw = valid_manifest()
        raw["episodes"][0]["initial_pose"]["position"]["x"] = float("nan")
        with self.assertRaisesRegex(ManifestValidationError, "finite"):
            parse_manifest(raw)

    def test_insufficient_scene_count_is_rejected(self) -> None:
        raw = valid_manifest()
        for episode in raw["episodes"]:
            episode["scene"] = "FloorPlan211"
        with self.assertRaisesRegex(ManifestValidationError, "at least 2 scenes"):
            parse_manifest(raw)

    def test_inference_only_must_be_explicit_true(self) -> None:
        raw = valid_manifest()
        raw["inference_only"] = False
        with self.assertRaisesRegex(ManifestValidationError, "inference_only"):
            parse_manifest(raw)

        raw = valid_manifest()
        raw.pop("inference_only")
        with self.assertRaisesRegex(ManifestValidationError, "inference_only"):
            parse_manifest(raw)

        raw = valid_manifest()
        raw["inference_only"] = "true"
        with self.assertRaisesRegex(ManifestValidationError, "inference_only"):
            parse_manifest(raw)

    def test_unknown_keys_are_rejected(self) -> None:
        raw = valid_manifest()
        raw["training_hint"] = "never allow hidden eval knobs"
        with self.assertRaisesRegex(ManifestValidationError, "unknown keys"):
            parse_manifest(raw)

    def test_interaction_task_requires_source_destination_and_actions(self) -> None:
        raw = valid_manifest()
        task = raw["episodes"][0]["task"]
        task["task_type"] = "interaction"
        task["target"] = {
            "object_type": "Vase",
            "source_object_type": "Vase",
        }
        task["required_actions"] = ["PickupObject"]
        with self.assertRaisesRegex(ManifestValidationError, "destination_object_type"):
            parse_manifest(raw)

        task["target"]["destination_object_type"] = "Box"
        with self.assertRaisesRegex(ManifestValidationError, "PickupObject and PutObject"):
            parse_manifest(raw)

    def test_unknown_required_action_is_rejected(self) -> None:
        raw = valid_manifest()
        raw["episodes"][0]["task"]["required_actions"] = ["STOP", "Dance"]
        with self.assertRaisesRegex(ManifestValidationError, "unsupported actions"):
            parse_manifest(raw)

    def test_instruction_mojibake_is_rejected(self) -> None:
        raw = valid_manifest()
        raw["episodes"][0]["task"]["instruction"] = "鎵惧埌鐢佃"
        with self.assertRaisesRegex(ManifestValidationError, "mojibake"):
            parse_manifest(raw)

    def test_plan2_manifest_file_is_inference_only_and_paired(self) -> None:
        manifest = load_manifest("configs/evaluation/plan2_multiscene_v1.json")
        coverage = manifest.coverage()

        self.assertTrue(manifest.inference_only)
        self.assertEqual(manifest.required_groups, ("oracle", "non_oracle"))
        self.assertGreaterEqual(coverage["episode_count"], 6)
        self.assertGreaterEqual(len(coverage["scenes"]), 3)
        self.assertIn("interaction", coverage["task_types"])
        self.assertIn("visual_search", coverage["task_types"])
        for episode in manifest.episodes:
            self.assertEqual(
                episode.result_file.split("/")[:2],
                [episode.split, episode.group],
            )
            self.assertNotRegex(episode.task.instruction, r"[鎵鎶鐢搴绠噷]")

    def test_requirements_are_pinned_to_remote_runtime_versions(self) -> None:
        requirements = Path("requirements.txt").read_text(encoding="utf-8")
        expected = {
            "fastapi==0.139.0",
            "uvicorn==0.50.2",
            "pillow==12.3.0",
            "numpy==2.2.6",
            "pyyaml==6.0.3",
            "opencv-python==5.0.0.93",
            "openai==2.44.0",
            "ai2thor==5.0.0",
        }

        self.assertEqual(
            set(line.strip() for line in requirements.splitlines() if line.strip()),
            expected,
        )


if __name__ == "__main__":
    unittest.main()
