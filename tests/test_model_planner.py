"""Tests for model planner integration.

According to Plan_1_agent_demo_repair.md Phase 2 requirements.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from src.agent.controller import EmbodiedSearchAgent
from src.agent.model_adapter import ApiCredential, ModelAdapter
from src.types.schema import AgentRequest, Action, SkillCall
from src.task.config import AgentConfig, load_config


class ModelPlannerTests(unittest.TestCase):
    """Test model planner integration into agent main loop."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._temporary_directory = TemporaryDirectory()
        default_config = load_config()
        raw = deepcopy(default_config.raw)
        raw["data"]["trajectory_dir"] = str(
            Path(cls._temporary_directory.name) / "trajectories"
        )
        cls.config = AgentConfig(raw=raw, path=default_config.path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary_directory.cleanup()

    def test_model_adapter_plan_action_requires_payload(self) -> None:
        """Test ModelAdapter.plan_action exists and accepts payload."""
        adapter = ModelAdapter(credentials=[])
        result = adapter.plan_action({
            "instruction": "Find red cup",
            "observation_summary": "Room with table",
            "confidence": 0.5,
            "allowed_actions": ["STOP", "TURN_RIGHT"],
            "terminal_actions": ["STOP"],
            "current_step": 0,
            "max_steps": 20
        })
        # Should return error when no credentials
        self.assertIn("error", result)
        self.assertIn("no_credentials", result.get("error", ""))

    def test_model_adapter_sends_robot_rgb_and_target_crop_to_vision_model(self) -> None:
        adapter = ModelAdapter(
            credentials=[
                ApiCredential(
                    provider="openai_compatible",
                    api_key="sk-test",
                    model="gpt-4o-mini",
                )
            ]
        )
        client = Mock()
        completion = Mock()
        completion.choices = [
            Mock(
                message=Mock(
                    content=(
                        '{"thought_summary":"inspect","action":{"type":"INSPECT",'
                        '"args":{}},"confidence":0.6}'
                    )
                )
            )
        ]
        client.chat.completions.create.return_value = completion

        with patch("src.agent.model_adapter.OpenAI", return_value=client):
            result = adapter.plan_action(
                {
                    "instruction": "Find this object",
                    "observation_summary": "Room view",
                    "confidence": 0.2,
                    "allowed_actions": ["INSPECT", "TURN_RIGHT"],
                    "terminal_actions": [],
                    "current_step": 0,
                    "max_steps": 20,
                    "observation_image": "data:image/png;base64,robot-rgb",
                    "target_crop": "data:image/png;base64,target-crop",
                }
            )

        messages = client.chat.completions.create.call_args.kwargs["messages"]
        user_content = messages[1]["content"]
        image_items = [item for item in user_content if item["type"] == "image_url"]
        self.assertEqual(len(image_items), 2)
        self.assertEqual(
            image_items[0]["image_url"]["url"],
            "data:image/png;base64,robot-rgb",
        )
        self.assertEqual(
            image_items[1]["image_url"]["url"],
            "data:image/png;base64,target-crop",
        )
        self.assertTrue(result["vision_input_used"])
        self.assertEqual(result["provider_used"], "openai_compatible")

    def test_deepseek_text_planner_does_not_claim_vision_input(self) -> None:
        adapter = ModelAdapter(
            credentials=[
                ApiCredential(
                    provider="deepseek",
                    api_key="sk-test",
                    base_url="https://api.deepseek.com/v1",
                    model="deepseek-chat",
                )
            ]
        )
        client = Mock()
        completion = Mock()
        completion.choices = [
            Mock(
                message=Mock(
                    content=(
                        '{"thought_summary":"turn","action":{"type":"TURN_RIGHT",'
                        '"args":{}},"confidence":0.2}'
                    )
                )
            )
        ]
        client.chat.completions.create.return_value = completion

        with patch("src.agent.model_adapter.OpenAI", return_value=client):
            result = adapter.plan_action(
                {
                    "instruction": "Find the television",
                    "observation_summary": "No target visible",
                    "confidence": 0.1,
                    "allowed_actions": ["TURN_RIGHT"],
                    "terminal_actions": [],
                    "current_step": 0,
                    "max_steps": 20,
                    "observation_image": "data:image/png;base64,robot-rgb",
                }
            )

        messages = client.chat.completions.create.call_args.kwargs["messages"]
        self.assertIsInstance(messages[1]["content"], str)
        self.assertFalse(result["vision_input_used"])

    def test_visual_request_does_not_fallback_to_text_only_provider(self) -> None:
        adapter = ModelAdapter(
            credentials=[
                ApiCredential(
                    provider="deepseek",
                    api_key="sk-test",
                    base_url="https://api.deepseek.com/v1",
                    model="deepseek-chat",
                )
            ]
        )
        result = adapter.plan_action(
            {
                "instruction": "Find the television",
                "observation_summary": "An image is attached",
                "confidence": 0.1,
                "allowed_actions": ["TURN_RIGHT"],
                "terminal_actions": [],
                "current_step": 0,
                "max_steps": 20,
                "observation_image": "data:image/png;base64,robot-rgb",
                "require_vision": True,
            }
        )

        self.assertEqual(result["error"], "all_model_calls_failed")
        self.assertIn("visual input is required", result["errors"][0])

    def test_agent_uses_model_planner_when_available(self) -> None:
        """Test agent calls model planner when credentials available."""
        agent = EmbodiedSearchAgent(self.config, model_adapter=ModelAdapter(credentials=[]))

        # Mock the model adapter to return valid response
        mock_result = {
            "thought_summary": "Turn right to explore",
            "action": {"type": "TURN_RIGHT", "args": {"angle": 30}},
            "skill_call": {
                "name": "TURN_RIGHT",
                "args": {"angle": 30},
                "preconditions": [],
                "expected_observation": "camera heading changes"
            },
            "confidence": 0.42
        }

        with patch.object(agent.model_adapter, 'plan_action', return_value=mock_result):
            with patch.object(agent.model_adapter, 'available', return_value=True):
                image_path = self.config.image_dir / "ep_red_cup_visible_000.png"
                response = agent.step(
                    AgentRequest(
                        session_id="test-model-planner",
                        instruction="Find the red cup",
                        observation_image=str(image_path),
                        step_id=0,
                    )
                )

                # Should use model_planner
                self.assertEqual(response.planner_source, "model_planner")
                self.assertIsNotNone(response.skill_call)
                self.assertEqual(response.skill_call.name, "TURN_RIGHT")
                self.assertEqual(response.thought, "Turn right to explore")

    def test_agent_accepts_task_relevant_native_ai2thor_action(self) -> None:
        agent = EmbodiedSearchAgent(self.config, model_adapter=ModelAdapter(credentials=[]))
        mock_result = {
            "thought_summary": "The cabinet is visible, so opening it is the next required task action.",
            "action": {"type": "OpenObject", "args": {"objectId": "Cabinet|1"}},
            "confidence": 0.74,
        }
        with patch.object(agent.model_adapter, "plan_action", return_value=mock_result):
            with patch.object(agent.model_adapter, "available", return_value=True):
                image_path = self.config.image_dir / "ep_red_cup_visible_000.png"
                response = agent.step(
                    AgentRequest(
                        session_id="test-native-open",
                        instruction="打开柜子",
                        observation_image=str(image_path),
                        step_id=0,
                    )
                )
        self.assertEqual(response.planner_source, "model_planner")
        self.assertEqual(response.action.type, "OpenObject")
        self.assertEqual(response.action.args["objectId"], "Cabinet|1")

    def test_agent_rejects_unverifiable_sit_task(self) -> None:
        agent = EmbodiedSearchAgent(self.config, model_adapter=ModelAdapter(credentials=[]))
        mock_result = {
            "thought_summary": "The sofa is visible.",
            "action": {"type": "INSPECT", "args": {}},
            "confidence": 0.9,
        }
        with patch.object(agent.model_adapter, "plan_action", return_value=mock_result):
            with patch.object(agent.model_adapter, "available", return_value=True):
                image_path = self.config.image_dir / "ep_red_cup_visible_000.png"
                response = agent.step(
                    AgentRequest(
                        session_id="test-unsupported-sit",
                        instruction="走到沙发上并坐下",
                        observation_image=str(image_path),
                        step_id=0,
                    )
                )
        self.assertEqual(response.action.type, "ASK_CLARIFY")
        self.assertFalse(response.task_plan["supported"])
        self.assertFalse(response.task_plan["is_visual_search"])
        self.assertFalse(response.completion_status["complete"])
        self.assertEqual(response.fallback_reason, "unsupported_task_capability")
        self.assertIn("sit-on-furniture", response.action.args["reason"])
        self.assertIn("任务尚未完成", response.thought)
        self.assertIn("任务未完成", response.structured_thought["reasoning"])

    def test_agent_fallback_when_model_returns_illegal_action(self) -> None:
        """Test agent falls back to rules when model returns illegal action."""
        agent = EmbodiedSearchAgent(self.config, model_adapter=ModelAdapter(credentials=[]))

        # Mock model to return illegal action
        mock_result = {
            "thought_summary": "Jump to target",
            "action": {"type": "JUMP", "args": {}},  # JUMP not in allowed_actions
            "confidence": 0.5
        }

        with patch.object(agent.model_adapter, 'plan_action', return_value=mock_result):
            with patch.object(agent.model_adapter, 'available', return_value=True):
                image_path = self.config.image_dir / "ep_red_cup_visible_000.png"
                response = agent.step(
                    AgentRequest(
                        session_id="test-illegal-action",
                        instruction="Find the red cup",
                        observation_image=str(image_path),
                        step_id=0,
                    )
                )

                # Should fallback
                self.assertEqual(response.planner_source, "rule_fallback")
                self.assertEqual(response.action.type, "ASK_CLARIFY")

    def test_agent_rejects_stop_with_low_confidence(self) -> None:
        """Test agent rejects STOP when confidence below threshold."""
        agent = EmbodiedSearchAgent(self.config, model_adapter=ModelAdapter(credentials=[]))

        # Mock model to return STOP with low confidence
        mock_result = {
            "thought_summary": "Found target",
            "action": {"type": "STOP", "args": {}},
            "confidence": 0.3  # Below stop_confidence_threshold (0.78)
        }

        with patch.object(agent.model_adapter, 'plan_action', return_value=mock_result):
            with patch.object(agent.model_adapter, 'available', return_value=True):
                image_path = self.config.image_dir / "ep_red_cup_visible_000.png"
                response = agent.step(
                    AgentRequest(
                        session_id="test-low-confidence-stop",
                        instruction="Find the red cup",
                        observation_image=str(image_path),
                        step_id=0,
                    )
                )

                # Should NOT stop, should fallback to exploration
                self.assertNotEqual(response.action.type, "STOP")
                self.assertEqual(response.planner_source, "rule_fallback")

    def test_agent_fallback_when_no_credentials(self) -> None:
        """Test agent uses rule fallback when no API credentials."""
        agent = EmbodiedSearchAgent(self.config, model_adapter=ModelAdapter(credentials=[]))

        with patch.object(agent.model_adapter, 'available', return_value=False):
            image_path = self.config.image_dir / "ep_red_cup_visible_000.png"
            response = agent.step(
                AgentRequest(
                    session_id="test-no-credentials",
                    instruction="Find the red cup",
                    observation_image=str(image_path),
                    step_id=0,
                )
            )

            # Should use rule_fallback
            self.assertEqual(response.planner_source, "rule_fallback")
            self.assertIn(response.action.type, self.config.allowed_actions)

    def test_planner_source_in_response(self) -> None:
        """Test planner_source field is present in AgentResponse."""
        agent = EmbodiedSearchAgent(self.config, model_adapter=ModelAdapter(credentials=[]))
        image_path = self.config.image_dir / "ep_red_cup_visible_000.png"
        response = agent.step(
            AgentRequest(
                session_id="test-planner-source",
                instruction="Find the red cup",
                observation_image=str(image_path),
                step_id=0,
            )
        )

        # planner_source must be one of the valid enum values
        valid_sources = ["model_planner", "rule_fallback", "simulator_oracle", "human_manual"]
        self.assertIn(response.planner_source, valid_sources)

        # Should be serializable
        response_dict = response.to_dict()
        self.assertIn("planner_source", response_dict)
        self.assertIn(response_dict["planner_source"], valid_sources)

    def test_skill_call_in_response(self) -> None:
        """Test skill_call field is present in AgentResponse."""
        agent = EmbodiedSearchAgent(self.config, model_adapter=ModelAdapter(credentials=[]))
        image_path = self.config.image_dir / "ep_red_cup_visible_000.png"
        response = agent.step(
            AgentRequest(
                session_id="test-skill-call",
                instruction="Find the red cup",
                observation_image=str(image_path),
                step_id=0,
            )
        )

        # skill_call should be present
        self.assertIsNotNone(response.skill_call)
        self.assertIsInstance(response.skill_call, SkillCall)
        self.assertIn(response.skill_call.name, self.config.allowed_actions)

        # Should be serializable
        response_dict = response.to_dict()
        self.assertIn("skill_call", response_dict)
        if response_dict["skill_call"]:
            self.assertIn("name", response_dict["skill_call"])


if __name__ == "__main__":
    unittest.main()
