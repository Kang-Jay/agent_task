from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from PIL import Image

from src.agent.controller import EmbodiedSearchAgent
from src.agent.model_adapter import ModelAdapter
from src.data.generate_demo_dataset import build_dataset
from src.evaluation.evaluator import evaluate, validate_dataset
from src.evaluation.metrics import compute_iou
from src.simulation.ai2thor_adapter import AI2ThorVisualSearchDemo
from src.task.config import AgentConfig, load_config
from src.types.schema import AgentRequest
from src.vision.image_io import image_to_data_url


TEST_SESSION_IDS = ["unit-basic", "unit-clicked", "unit-enhanced", "unit-terminal"]


class AgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temporary_directory = TemporaryDirectory()
        temporary_root = Path(cls._temporary_directory.name)
        default_config = load_config()
        raw = deepcopy(default_config.raw)
        raw["data"] = {
            "dataset_root": str(temporary_root),
            "annotation_file": str(temporary_root / "annotations" / "episodes.jsonl"),
            "trajectory_dir": str(temporary_root / "trajectories"),
            "image_dir": str(temporary_root / "images"),
        }
        cls.config = AgentConfig(raw=raw, path=default_config.path)
        build_dataset(cls.config)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary_directory.cleanup()

    def tearDown(self) -> None:
        for session_id in TEST_SESSION_IDS:
            trace_path = self.config.trajectory_dir / f"{session_id}.json"
            if trace_path.exists():
                trace_path.unlink()

    def make_agent(self) -> EmbodiedSearchAgent:
        return EmbodiedSearchAgent(
            self.config,
            model_adapter=ModelAdapter(credentials=[]),
        )

    def test_config_is_consistent(self) -> None:
        audit = self.make_agent().audit()
        self.assertEqual(audit["status"], "ok")
        self.assertIn("STOP", audit["allowed_actions"])

    def test_dataset_validates(self) -> None:
        validate_dataset(self.config)

    def test_demo_targets_are_localized_with_sufficient_iou(self) -> None:
        agent = self.make_agent()
        episodes = build_dataset(self.config)
        for episode in episodes:
            response = agent.step(
                AgentRequest(
                    session_id=f"localize-{episode['episode_id'].replace('_', '-')}",
                    instruction=episode["instruction"],
                    observation_image=str(
                        self.config.data_root / str(episode["image"])
                    ),
                    step_id=0,
                )
            )
            self.assertIsNotNone(response.observation.best_candidate)
            iou = compute_iou(
                response.observation.best_candidate.bbox,
                episode["target"]["bbox"],
            )
            self.assertGreaterEqual(iou, 0.5, episode["episode_id"])

    def test_basic_language_step_returns_thought_and_valid_action(self) -> None:
        agent = self.make_agent()
        image_path = self.config.image_dir / "ep_red_cup_visible_000.png"
        response = agent.step(
            AgentRequest(
                session_id="unit-basic",
                instruction="Find the red cup on the table",
                observation_image=str(image_path),
                step_id=0,
            )
        )
        self.assertTrue(response.thought)
        self.assertIn(response.action.type, self.config.allowed_actions)
        self.assertEqual(response.action.type, "STOP")
        self.assertTrue(response.done)

    def test_clicked_target_crop_is_supported(self) -> None:
        agent = self.make_agent()
        image_path = self.config.image_dir / "ep_blue_book_visible_000.png"
        response = agent.step(
            AgentRequest(
                session_id="unit-clicked",
                instruction="找到这个目标物体",
                observation_image=str(image_path),
                step_id=0,
                clicked_point=[336, 210],
            )
        )
        self.assertIn(response.action.type, self.config.allowed_actions)
        self.assertGreaterEqual(response.confidence, self.config.target_visible_threshold)
        self.assertTrue(response.observation.candidates)
        self.assertEqual(response.target_binding["mode"], "multimodal")
        self.assertTrue(response.target_binding["target_crop"])

    def test_enhanced_response_contains_replay_map_and_confidence_trace(self) -> None:
        agent = self.make_agent()
        image_path = self.config.image_dir / "ep_green_plant_visible_000.png"
        response = agent.step(
            AgentRequest(
                session_id="unit-enhanced",
                instruction="Find the green plant near the window",
                observation_image=str(image_path),
                step_id=0,
            )
        )
        self.assertIn("visited_counts", response.search_map)
        self.assertIn("confidence_by_region", response.search_map)
        self.assertEqual(response.confidence_trace, [response.confidence])
        exported = agent.export_trace("unit-enhanced")
        self.assertEqual(exported["session_id"], "unit-enhanced")
        self.assertEqual(len(exported["steps"]), 1)

    def test_configured_terminal_action_marks_response_done(self) -> None:
        agent = self.make_agent()
        state = agent.memory.get_or_create("unit-terminal", "Find the purple spaceship")
        for step_id in range(self.config.max_steps - 1):
            state.steps.append({"step_id": step_id, "action": {"type": "TURN_RIGHT"}, "confidence": 0.0, "done": False})
        blank_observation = Image.new("RGB", self.config.image_size, (24, 24, 24))
        response = agent.step(
            AgentRequest(
                session_id="unit-terminal",
                instruction="Find the purple spaceship",
                observation_image=image_to_data_url(blank_observation),
                step_id=self.config.max_steps - 1,
            )
        )
        self.assertEqual(response.action.type, "ASK_CLARIFY")
        self.assertTrue(response.done)

    def test_ai2thor_segmentation_grounding_matches_target_mask(self) -> None:
        demo = AI2ThorVisualSearchDemo()
        mask = np.zeros((120, 160), dtype=bool)
        mask[30:70, 80:130] = True
        event = type(
            "FakeEvent",
            (),
            {
                "instance_masks": {"Television|+00.00|+00.00|+00.00": mask},
                "metadata": {
                    "objects": [
                        {
                            "objectId": "Television|+00.00|+00.00|+00.00",
                            "objectType": "Television",
                        }
                    ]
                },
            },
        )()
        candidate = demo._ground_target_from_segmentation(event, "Find the television in the room")
        self.assertIsNotNone(candidate)
        self.assertEqual(candidate["object_type"], "Television")
        self.assertEqual(candidate["bbox"], [80, 30, 130, 70])

    def test_evaluator_passes_demo_dataset(self) -> None:
        result = evaluate(self.config, agent=self.make_agent())
        self.assertEqual(result.illegal_actions, 0)
        self.assertEqual(result.successes, result.episodes)
        self.assertGreaterEqual(result.average_iou, 0.5)
        self.assertIsNone(result.spl)
        self.assertEqual(result.spl_coverage, 0.0)


if __name__ == "__main__":
    unittest.main()
