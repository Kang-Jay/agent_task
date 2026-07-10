"""Test Phase 3 click multimodal integration.

According to Plan_1_agent_demo_repair.md Phase 3 requirements.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from src.agent.controller import EmbodiedSearchAgent
from src.agent.model_adapter import ModelAdapter
from src.data.generate_demo_dataset import build_dataset
from src.types.schema import AgentRequest
from src.task.config import AgentConfig, load_config


class ClickMultimodalTests(unittest.TestCase):
    """Test click integration from frontend to backend to agent."""

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

    def make_agent(self) -> EmbodiedSearchAgent:
        return EmbodiedSearchAgent(
            self.config,
            model_adapter=ModelAdapter(credentials=[]),
        )

    def tearDown(self) -> None:
        for session_id in ("test-click", "test-language", "test-serialize", "recorded-demo"):
            trace_path = self.config.trajectory_dir / f"{session_id}.json"
            if trace_path.exists():
                trace_path.unlink()

    def test_agent_accepts_clicked_point(self) -> None:
        """Test Agent.step() accepts clicked_point parameter."""
        agent = self.make_agent()
        image_path = self.config.image_dir / "ep_red_cup_visible_000.png"

        response = agent.step(
            AgentRequest(
                session_id="test-click",
                instruction="找到这个物体",
                observation_image=str(image_path),
                step_id=0,
                clicked_point=[224, 224]  # Center of 448x448 image
            )
        )

        # Should have multimodal mode
        self.assertEqual(response.target_binding["mode"], "multimodal")
        self.assertTrue(response.target_binding["target_crop"])
        self.assertEqual(response.target_binding["clicked_point"], [224, 224])

    def test_no_clicked_point_is_language_only(self) -> None:
        """Test without clicked_point defaults to language_only mode."""
        agent = self.make_agent()
        image_path = self.config.image_dir / "ep_red_cup_visible_000.png"

        response = agent.step(
            AgentRequest(
                session_id="test-language",
                instruction="找到红色杯子",
                observation_image=str(image_path),
                step_id=0,
                clicked_point=None
            )
        )

        self.assertEqual(response.target_binding["mode"], "language_only")
        self.assertFalse(response.target_binding["target_crop"])
        self.assertIsNone(response.target_binding["clicked_point"])

    def test_room_simulator_accepts_clicked_point(self) -> None:
        """Test RoomSimulator.run_demo() accepts clicked_point."""
        from src.simulation.room_simulator import RoomSimulator

        with TemporaryDirectory() as output_dir:
            simulator = RoomSimulator(agent=self.make_agent(), output_dir=output_dir)

            result = simulator.run_demo(
                instruction="找到目标物体",
                max_steps=3,
                clicked_point=[200, 200]
            )

            self.assertGreater(len(result.steps), 0)
            if result.steps[0].target_binding:
                self.assertEqual(result.steps[0].target_binding.get("mode"), "multimodal")

    def test_ai2thor_adapter_accepts_clicked_point(self) -> None:
        """Test AI2ThorVisualSearchDemo.run_demo() signature accepts clicked_point."""
        from src.simulation.ai2thor_adapter import AI2ThorVisualSearchDemo
        import inspect

        sig = inspect.signature(AI2ThorVisualSearchDemo.run_demo)
        params = list(sig.parameters.keys())

        self.assertIn("clicked_point", params)

    def test_api_payload_includes_clicked_point(self) -> None:
        """Test API endpoints accept clicked_point in payload."""
        from src.ui.app import StepPayload

        # Test StepPayload model
        payload = StepPayload(
            session_id="test",
            instruction="test",
            observation_image="test.png",
            step_id=0,
            clicked_point=[100, 100]
        )

        self.assertEqual(payload.clicked_point, [100, 100])

    def test_demo_run_endpoint_passes_clicked_point(self) -> None:
        """Test /api/demo/run endpoint passes clicked_point to simulator."""
        from src.ui.app import run_demo

        # Mock RoomSimulator
        with patch('src.ui.app.RoomSimulator') as MockSimulator:
            mock_instance = Mock()
            mock_result = Mock()
            mock_result.to_dict.return_value = {"steps": [], "video_path": "", "summary_path": ""}
            mock_instance.run_demo.return_value = mock_result
            MockSimulator.return_value = mock_instance

            # Call endpoint with clicked_point
            result = run_demo({
                "session_id": "api-session",
                "instruction": "test",
                "max_steps": 5,
                "clicked_point": [150, 150]
            })

            # Verify RoomSimulator.run_demo was called with clicked_point
            mock_instance.run_demo.assert_called_once()
            MockSimulator.assert_called_once()
            self.assertIsNotNone(MockSimulator.call_args.kwargs.get("agent"))
            call_kwargs = mock_instance.run_demo.call_args.kwargs
            self.assertEqual(call_kwargs["clicked_point"], [150, 150])
            self.assertEqual(call_kwargs["session_id"], "api-session")

    def test_audit_does_not_call_paid_model_smoke_test(self) -> None:
        """Static audit must not make a live model API request."""
        from src.ui.app import audit

        with patch("src.ui.app.smoke_test") as mocked_smoke_test:
            result = audit()

        mocked_smoke_test.assert_not_called()
        self.assertIn("model_adapter", result)
        self.assertIn("available", result["model_adapter"])

    def test_target_binding_serialization(self) -> None:
        """Test target_binding with clicked_point serializes correctly."""
        agent = self.make_agent()
        image_path = self.config.image_dir / "ep_red_cup_visible_000.png"

        response = agent.step(
            AgentRequest(
                session_id="test-serialize",
                instruction="找到目标",
                observation_image=str(image_path),
                step_id=0,
                clicked_point=[300, 200]
            )
        )

        response_dict = response.to_dict()

        self.assertIn("target_binding", response_dict)
        self.assertEqual(response_dict["target_binding"]["mode"], "multimodal")
        self.assertEqual(response_dict["target_binding"]["clicked_point"], [300, 200])
        self.assertTrue(response_dict["target_binding"]["target_crop"])


if __name__ == "__main__":
    unittest.main()
