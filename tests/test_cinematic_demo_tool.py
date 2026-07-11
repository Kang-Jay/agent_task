from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from PIL import Image

import tools.make_cinematic_demo as cinematic


class CinematicDemoToolTests(unittest.TestCase):
    def _write_summary(
        self,
        root: Path,
        *,
        post_action: bool,
    ) -> Path:
        frame_dir = root / "frames"
        frame_dir.mkdir(parents=True, exist_ok=True)
        observation_name = (
            "ai2thor_obs_after_00.png"
            if post_action
            else "ai2thor_obs_00.png"
        )
        observation_path = frame_dir / observation_name
        topdown_path = frame_dir / "ai2thor_topdown_00.png"
        Image.new("RGB", (960, 540), (120, 80, 40)).save(
            observation_path
        )
        Image.new("RGB", (500, 500), (230, 235, 240)).save(
            topdown_path
        )
        summary = {
            "instruction": "找到右边的门，然后走出去",
            "scene": "FloorPlan211",
            "steps": [
                {
                    "action": "MOVE_FORWARD",
                    "backend": "ai2thor",
                    "confidence": 0.81,
                    "done": False,
                    "thought": "Move through the verified doorway.",
                    "observation_path": str(
                        observation_path.relative_to(root)
                    ),
                    "topdown_path": str(topdown_path.relative_to(root)),
                    "robot": {"x": 1.0, "y": 2.0, "heading": 90.0},
                    "best_candidate": None,
                    "visible_objects": ["Door"],
                }
            ],
        }
        summary_path = root / "summary.json"
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False),
            encoding="utf-8",
        )
        return summary_path

    def test_module_import_has_no_file_side_effects(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(cinematic.PROJECT_ROOT)
            completed = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-c",
                    "import tools.make_cinematic_demo",
                ],
                cwd=root,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(list(root.iterdir()), [])

    def test_legacy_pre_action_summary_is_rejected_by_default(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            summary_path = self._write_summary(
                root,
                post_action=False,
            )

            with self.assertRaisesRegex(
                ValueError,
                "legacy pre-action observation",
            ):
                cinematic.load_demo_summary(
                    summary_path,
                    project_root=root,
                    settings=cinematic.CinematicSettings(),
                )

    def test_post_action_labels_describe_executed_action(self) -> None:
        step = {
            "action": "TURN_RIGHT",
            "observation_path": "ai2thor_obs_after_02.png",
        }

        self.assertEqual(
            cinematic.infer_observation_phase(step),
            "after_action",
        )
        self.assertIn("AFTER EXECUTED ACTION", cinematic.observation_badge(step))
        self.assertIn("EXECUTED ACTION", cinematic.action_badge(step))
        self.assertNotIn("NEXT ACTION", cinematic.action_badge(step))

    def test_generation_cleans_default_temporary_frames(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            summary_path = self._write_summary(
                root,
                post_action=True,
            )
            output_path = root / "output.mp4"
            verification_path = root / "verification.json"
            captured_frame_dirs: list[Path] = []
            original_renderer = cinematic.render_source_frames

            def capture_renderer(**kwargs):
                captured_frame_dirs.append(kwargs["frame_dir"])
                return original_renderer(**kwargs)

            def fake_encoder(frames, path, *, fps):
                path.write_bytes(b"fake-h264-video")
                return {
                    "codec": "h264",
                    "pixel_format": "yuv420p",
                    "frame_count": len(frames),
                    "fps": fps,
                }

            with patch.object(
                cinematic,
                "render_source_frames",
                side_effect=capture_renderer,
            ), patch.object(
                cinematic,
                "probe_video",
                return_value={
                    "frame_count": 1,
                    "fps": 24.0,
                    "width": cinematic.FRAME_WIDTH,
                    "height": cinematic.FRAME_HEIGHT,
                },
            ):
                verification = cinematic.generate_cinematic_demo(
                    summary_path=summary_path,
                    output_video=output_path,
                    verification_path=verification_path,
                    project_root=root,
                    settings=cinematic.CinematicSettings(
                        fps=24,
                        hold_frames=1,
                        intro_frames=0,
                        outro_frames=0,
                    ),
                    encoder=fake_encoder,
                )

            self.assertTrue(output_path.is_file())
            self.assertTrue(verification_path.is_file())
            self.assertTrue(verification["post_action_semantics"])
            self.assertEqual(
                verification["observation_phases"],
                ["after_action"],
            )
            self.assertEqual(verification["actions"], ["MOVE_FORWARD"])
            self.assertEqual(len(captured_frame_dirs), 1)
            self.assertFalse(captured_frame_dirs[0].exists())

    def test_real_encoder_produces_browser_compatible_video(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            summary_path = self._write_summary(
                root,
                post_action=True,
            )
            output_path = root / "real-output.mp4"
            verification_path = root / "real-verification.json"

            verification = cinematic.generate_cinematic_demo(
                summary_path=summary_path,
                output_video=output_path,
                verification_path=verification_path,
                project_root=root,
                settings=cinematic.CinematicSettings(
                    fps=2,
                    hold_frames=1,
                    intro_frames=0,
                    outro_frames=0,
                ),
            )

            self.assertTrue(output_path.is_file())
            self.assertGreater(output_path.stat().st_size, 0)
            self.assertEqual(verification["codec"], "h264")
            self.assertEqual(verification["pixel_format"], "yuv420p")
            self.assertEqual(verification["frame_count"], 1)
            self.assertEqual(verification["width"], cinematic.FRAME_WIDTH)
            self.assertEqual(verification["height"], cinematic.FRAME_HEIGHT)
            self.assertTrue(verification["post_action_semantics"])

    def test_cli_generates_verification_without_kept_frames(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            summary_path = self._write_summary(root, post_action=True)
            output_path = root / "cli-output.mp4"
            verification_path = root / "cli-verification.json"

            completed = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(cinematic.PROJECT_ROOT / "tools" / "make_cinematic_demo.py"),
                    "--project-root",
                    str(root),
                    "--summary",
                    str(summary_path),
                    "--output",
                    str(output_path),
                    "--verification",
                    str(verification_path),
                    "--fps",
                    "2",
                    "--hold-frames",
                    "1",
                    "--intro-frames",
                    "0",
                    "--outro-frames",
                    "0",
                ],
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue(output_path.is_file())
            self.assertTrue(verification_path.is_file())
            verification = json.loads(
                verification_path.read_text(encoding="utf-8")
            )
            self.assertTrue(verification["post_action_semantics"])
            self.assertIsNone(verification["kept_frames_dir"])


if __name__ == "__main__":
    unittest.main()
