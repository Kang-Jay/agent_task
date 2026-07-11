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
        "required_actions": [],
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
                    "result_file": f"{pair_id}-{group}/ai2thor_demo_summary.json",
                }
            )
    return {
        "schema_version": "1.0",
        "benchmark_id": "unit-test",
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

    def test_plan2_manifest_file_is_inference_only_and_paired(self) -> None:
        manifest = load_manifest("configs/evaluation/plan2_multiscene_v1.json")
        coverage = manifest.coverage()

        self.assertTrue(manifest.inference_only)
        self.assertEqual(manifest.required_groups, ("oracle", "non_oracle"))
        self.assertGreaterEqual(coverage["episode_count"], 6)
        self.assertGreaterEqual(len(coverage["scenes"]), 3)
        self.assertIn("interaction", coverage["task_types"])
        self.assertIn("visual_search", coverage["task_types"])

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
