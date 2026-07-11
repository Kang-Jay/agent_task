from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.evaluation.evaluator import (
    evaluate_manifest_results,
    load_result_trajectory,
    write_manifest_summary,
)
from src.evaluation.manifest import load_manifest


def _pose() -> dict:
    return {
        "position": {"x": 0.0, "y": 0.9, "z": 0.0},
        "rotation": {"x": 0.0, "y": 90.0, "z": 0.0},
        "horizon": 0.0,
        "standing": True,
    }


def _episode(group: str) -> dict:
    return {
        "episode_id": f"case-television-{group}",
        "pair_id": "case-television",
        "group": group,
        "split": "development",
        "scene": "FloorPlan211",
        "seed": 211001,
        "initial_pose": _pose(),
        "task": {
            "instruction": "Find the television",
            "task_type": "visual_search",
            "target": {"object_type": "Television"},
            "required_actions": ["STOP"],
            "allows_approximate_success": False,
        },
        "reference": {
            "optimal_path_length_meters": None,
            "source": "unit-test",
            "allowed_error_meters": 0.05,
        },
        "result_file": f"development/{group}/case-television-{group}.json",
    }


def _write_manifest(path: Path) -> None:
    manifest = {
        "schema_version": "1.0",
        "benchmark_id": "unit-evaluator",
        "dataset_version": "unit-test",
        "inference_only": True,
        "description": "Inference-only evaluator aggregation fixture.",
        "protocol": {
            "required_groups": ["oracle", "non_oracle"],
            "minimum_scene_count": 1,
        },
        "episodes": [_episode("oracle"), _episode("non_oracle")],
    }
    path.write_text(json.dumps(manifest), encoding="utf-8")


def _write_result(path: Path, *, confidence: float = 0.95) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "steps": [
            {
                "action": {"type": "STOP"},
                "done": True,
                "confidence": confidence,
                "planner_source": "model_planner",
                "observation": {"best_candidate": {"bbox": [0, 0, 10, 10]}},
            }
        ],
        "path_length_meters": 0.0,
        "execution_time": 0.25,
    }
    path.write_text(json.dumps(result), encoding="utf-8")


class EvaluatorManifestAggregationTests(unittest.TestCase):
    def test_evaluate_manifest_results_aggregates_existing_results(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "manifest.json"
            output_dir = root / "results"
            _write_manifest(manifest_path)
            manifest = load_manifest(manifest_path)
            for episode in manifest.episodes:
                _write_result(output_dir / episode.result_file)

            summary = evaluate_manifest_results(manifest_path, output_dir)

            self.assertEqual(summary["benchmark_id"], "unit-evaluator")
            self.assertEqual(summary["episode_count"], 2)
            self.assertEqual(summary["pair_count"], 1)
            self.assertEqual(summary["evaluated_episodes"], 2)
            self.assertEqual(summary["manifest_sha256"], manifest.sha256())
            self.assertEqual(summary["metrics"]["total_episodes"], 2)
            self.assertEqual(summary["metrics"]["success_rate"], 1.0)
            self.assertEqual(summary["metrics"]["strict_success_rate"], 1.0)
            self.assertEqual(summary["metrics"]["model_planner_usage_rate"], 1.0)
            self.assertEqual(
                set(summary["metrics"]["by_group"]),
                {"oracle", "non_oracle"},
            )
            self.assertEqual(
                summary["metrics"]["by_task_type"]["visual_search"]["episodes"],
                2,
            )

            summary_path = write_manifest_summary(summary, output_dir)
            saved = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["manifest_sha256"], manifest.sha256())
            self.assertEqual(saved["metrics"]["total_episodes"], 2)

    def test_evaluate_manifest_results_filters_group(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "manifest.json"
            output_dir = root / "results"
            _write_manifest(manifest_path)
            manifest = load_manifest(manifest_path)
            for episode in manifest.episodes:
                _write_result(output_dir / episode.result_file)

            summary = evaluate_manifest_results(
                manifest_path,
                output_dir,
                group="oracle",
            )

            self.assertEqual(summary["evaluated_episodes"], 1)
            self.assertEqual(summary["episodes"][0]["group"], "oracle")
            self.assertEqual(summary["metrics"]["total_episodes"], 1)
            self.assertEqual(set(summary["metrics"]["by_group"]), {"oracle"})

    def test_dry_run_validates_manifest_without_reading_results(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "manifest.json"
            _write_manifest(manifest_path)

            summary = evaluate_manifest_results(
                manifest_path,
                root / "missing-results",
                dry_run=True,
            )

            self.assertTrue(summary["dry_run"])
            self.assertEqual(summary["episode_count"], 2)
            self.assertEqual(summary["evaluated_episodes"], 0)
            self.assertEqual(summary["metrics"]["total_episodes"], 0)

    def test_missing_result_file_is_reported_with_path(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "manifest.json"
            _write_manifest(manifest_path)

            with self.assertRaisesRegex(FileNotFoundError, "missing result file"):
                evaluate_manifest_results(manifest_path, root / "results")

    def test_non_object_result_step_is_rejected(self) -> None:
        with TemporaryDirectory() as tmp:
            result_path = Path(tmp) / "result.json"
            result_path.write_text(
                json.dumps({"steps": [{"action": {"type": "STOP"}}, "bad-step"]}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "step 1 must be a JSON object"):
                load_result_trajectory(result_path)


if __name__ == "__main__":
    unittest.main()
