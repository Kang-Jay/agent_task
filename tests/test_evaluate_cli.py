from __future__ import annotations

import io
import json
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from tools.evaluate import main

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


class EvaluateCliTests(unittest.TestCase):
    def test_cli_writes_manifest_summary(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "manifest.json"
            output_dir = root / "results"
            _write_manifest(manifest_path)
            manifest = load_manifest(manifest_path)
            for episode in manifest.episodes:
                _write_result(output_dir / episode.result_file)

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "--manifest",
                        str(manifest_path),
                        "--output-dir",
                        str(output_dir),
                    ]
                )

            self.assertEqual(exit_code, 0)
            summary_path = output_dir / "summary.json"
            self.assertTrue(summary_path.exists())
            printed = json.loads(stdout.getvalue())
            saved = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(printed["manifest_sha256"], manifest.sha256())
            self.assertEqual(saved["metrics"]["total_episodes"], 2)

    def test_cli_no_write_leaves_output_directory_clean(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "manifest.json"
            output_dir = root / "results"
            _write_manifest(manifest_path)
            manifest = load_manifest(manifest_path)
            for episode in manifest.episodes:
                _write_result(output_dir / episode.result_file)

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "--manifest",
                        str(manifest_path),
                        "--output-dir",
                        str(output_dir),
                        "--no-write",
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertFalse((output_dir / "summary.json").exists())
            printed = json.loads(stdout.getvalue())
            self.assertEqual(printed["evaluated_episodes"], 2)

    def test_cli_returns_2_on_missing_result(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "manifest.json"
            output_dir = root / "results"
            _write_manifest(manifest_path)

            stderr = io.StringIO()
            with redirect_stderr(stderr):
                exit_code = main(
                    [
                        "--manifest",
                        str(manifest_path),
                        "--output-dir",
                        str(output_dir),
                    ]
                )

            self.assertEqual(exit_code, 2)
            self.assertIn("evaluation failed: missing result file", stderr.getvalue())
            self.assertFalse((output_dir / "summary.json").exists())


if __name__ == "__main__":
    unittest.main()
