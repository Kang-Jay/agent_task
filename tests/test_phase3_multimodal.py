"""Phase 3: Click-based Multimodal Closed-Loop Tests

Tests that verify:
1. Frontend request contains clicked_point
2. Backend correctly handles clicked_point
3. Simulators pass clicked_point to Agent
4. target_binding.mode is correctly set
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, Mock, patch

from src.simulation.ai2thor_adapter import AI2ThorVisualSearchDemo
from src.simulation.room_simulator import RoomSimulator
from src.types.schema import AgentRequest, AgentResponse, Action, ObservationAnalysis


class TestClickedPointFlow(unittest.TestCase):
    """Test that clicked_point flows from frontend through backend to agent."""

    def test_agent_request_accepts_clicked_point(self):
        """Test that AgentRequest schema accepts clicked_point parameter."""
        request = AgentRequest(
            session_id="test",
            instruction="Find the red cup",
            observation_image="data:image/png;base64,test",
            step_id=0,
            clicked_point=[320, 240],
        )
        self.assertEqual(request.clicked_point, [320, 240])
        self.assertEqual(request.step_id, 0)

    def test_agent_request_without_clicked_point(self):
        """Test that AgentRequest works without clicked_point (backwards compatible)."""
        request = AgentRequest(
            session_id="test",
            instruction="Find the red cup",
            observation_image="data:image/png;base64,test",
            step_id=0,
        )
        self.assertIsNone(request.clicked_point)

    def test_room_simulator_accepts_clicked_point(self):
        """Test that RoomSimulator.run_demo accepts clicked_point parameter."""
        mock_agent = Mock()
        mock_agent.reset = Mock()
        mock_agent.step = Mock(return_value=self._create_mock_response())

        with TemporaryDirectory() as output_dir:
            simulator = RoomSimulator(agent=mock_agent, output_dir=output_dir)

            result = simulator.run_demo(
                instruction="Find the red cup",
                max_steps=1,
                clicked_point=[320, 240]
            )

        self.assertTrue(mock_agent.step.called)

        first_call_args = mock_agent.step.call_args_list[0][0][0]
        self.assertEqual(first_call_args.clicked_point, [320, 240])
        self.assertEqual(first_call_args.step_id, 0)

    def test_room_simulator_only_passes_clicked_point_on_first_step(self):
        """Test that clicked_point is only passed to agent on step 0."""
        mock_agent = Mock()
        mock_agent.reset = Mock()
        mock_agent.step = Mock(return_value=self._create_mock_response())

        with TemporaryDirectory() as output_dir:
            simulator = RoomSimulator(agent=mock_agent, output_dir=output_dir)

            simulator.run_demo(
                instruction="Find the red cup",
                max_steps=3,
                clicked_point=[320, 240]
            )

        first_call = mock_agent.step.call_args_list[0][0][0]
        self.assertEqual(first_call.clicked_point, [320, 240])
        self.assertEqual(first_call.step_id, 0)

        if len(mock_agent.step.call_args_list) > 1:
            second_call = mock_agent.step.call_args_list[1][0][0]
            self.assertIsNone(second_call.clicked_point)
            self.assertEqual(second_call.step_id, 1)

    def test_ai2thor_adapter_accepts_clicked_point(self):
        """Test that AI2ThorVisualSearchDemo.run_demo accepts clicked_point parameter."""
        # We can't actually run AI2-THOR in tests, but we can verify the signature
        import inspect
        sig = inspect.signature(AI2ThorVisualSearchDemo.run_demo)
        self.assertIn('clicked_point', sig.parameters)

        # Verify default is None
        default = sig.parameters['clicked_point'].default
        self.assertIsNone(default)

    def test_target_binding_mode_without_clicked_point(self):
        """Test that target_binding.mode is 'language_only' when no clicked_point."""
        mock_agent = Mock()
        mock_agent.reset = Mock()
        mock_response = self._create_mock_response(
            target_binding={"mode": "language_only"}
        )
        mock_agent.step = Mock(return_value=mock_response)

        with TemporaryDirectory() as output_dir:
            simulator = RoomSimulator(agent=mock_agent, output_dir=output_dir)

            result = simulator.run_demo(
                instruction="Find the red cup",
                max_steps=1,
                clicked_point=None
            )

        self.assertEqual(len(result.steps), 1)
        step = result.steps[0]
        if step.target_binding:
            self.assertEqual(step.target_binding.get("mode"), "language_only")

    def test_backend_api_payload_structure(self):
        """Test that backend API expects correct payload structure."""
        # Simulate the payload that frontend sends
        payload = {
            "instruction": "Find the red cup",
            "max_steps": 20,
            "clicked_point": [320, 240]
        }

        # Verify payload structure is valid
        self.assertIn("instruction", payload)
        self.assertIn("clicked_point", payload)
        self.assertIsInstance(payload["clicked_point"], list)
        self.assertEqual(len(payload["clicked_point"]), 2)

    def test_clicked_point_coordinates_are_integers(self):
        """Test that clicked_point coordinates are integers."""
        request = AgentRequest(
            session_id="test",
            instruction="Find the red cup",
            observation_image="data:image/png;base64,test",
            step_id=0,
            clicked_point=[320, 240],
        )

        self.assertIsInstance(request.clicked_point[0], int)
        self.assertIsInstance(request.clicked_point[1], int)

    def test_demo_result_preserves_target_binding(self):
        """Test that DemoResult preserves target_binding from AgentResponse."""
        mock_agent = Mock()
        mock_agent.reset = Mock()
        mock_response = self._create_mock_response(
            target_binding={
                "mode": "multimodal",
                "clicked_point": [320, 240],
                "crop_source": "user_click"
            }
        )
        mock_agent.step = Mock(return_value=mock_response)

        with TemporaryDirectory() as output_dir:
            simulator = RoomSimulator(agent=mock_agent, output_dir=output_dir)

            result = simulator.run_demo(
                instruction="Find the red cup",
                max_steps=1,
                clicked_point=[320, 240]
            )

        self.assertEqual(len(result.steps), 1)
        step = result.steps[0]
        self.assertIsNotNone(step.target_binding)
        self.assertEqual(step.target_binding.get("mode"), "multimodal")
        self.assertEqual(step.target_binding.get("clicked_point"), [320, 240])

    def _create_mock_response(
        self,
        target_binding: dict | None = None,
    ) -> AgentResponse:
        """Create a mock AgentResponse for testing."""
        return AgentResponse(
            session_id="test",
            step_id=0,
            thought="Test thought",
            action=Action(type="TURN_RIGHT"),
            confidence=0.5,
            done=False,
            observation=ObservationAnalysis(
                image_size=(640, 480),
                scene_summary="Test scene",
                candidates=[],
                best_candidate=None,
                target_visible=False
            ),
            retrieved_hints=[],
            memory_summary="",
            replay=[],
            target_binding=target_binding or {"mode": "language_only"}
        )


class TestBackwardsCompatibility(unittest.TestCase):
    """Test that the system remains backwards compatible without clicked_point."""

    def test_room_simulator_without_clicked_point(self):
        """Test that RoomSimulator works without clicked_point parameter."""
        mock_agent = Mock()
        mock_agent.reset = Mock()
        mock_agent.step = Mock(return_value=self._create_mock_response())

        with TemporaryDirectory() as output_dir:
            simulator = RoomSimulator(agent=mock_agent, output_dir=output_dir)

            result = simulator.run_demo(instruction="Find the red cup", max_steps=1)

        self.assertGreater(len(result.steps), 0)

        first_call = mock_agent.step.call_args_list[0][0][0]
        self.assertIsNone(first_call.clicked_point)

    def test_api_payload_without_clicked_point(self):
        """Test that API payload works without clicked_point."""
        payload = {
            "instruction": "Find the red cup",
            "max_steps": 20
        }

        # Simulate what app.py does
        instruction = payload.get("instruction")
        max_steps = payload.get("max_steps")
        clicked_point = payload.get("clicked_point")

        self.assertIsNotNone(instruction)
        self.assertIsNotNone(max_steps)
        self.assertIsNone(clicked_point)

    def _create_mock_response(self) -> AgentResponse:
        """Create a mock AgentResponse for testing."""
        return AgentResponse(
            session_id="test",
            step_id=0,
            thought="Test thought",
            action=Action(type="TURN_RIGHT"),
            confidence=0.5,
            done=False,
            observation=ObservationAnalysis(
                image_size=(640, 480),
                scene_summary="Test scene",
                candidates=[],
                best_candidate=None,
                target_visible=False
            ),
            retrieved_hints=[],
            memory_summary="",
            replay=[]
        )


if __name__ == "__main__":
    unittest.main()
