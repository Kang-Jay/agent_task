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

    def test_kimi_k2_uses_recorded_request_settings(self) -> None:
        adapter = ModelAdapter(
            credentials=[
                ApiCredential(
                    provider="kimi",
                    api_key="sk-test",
                    base_url="https://api.moonshot.cn/v1",
                    model="kimi-k2.6",
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
        openai_constructor = Mock(return_value=client)

        with patch("src.agent.model_adapter.OpenAI", openai_constructor):
            result = adapter.plan_action(
                {
                    "instruction": "Find the television",
                    "observation_summary": "Room view",
                    "confidence": 0.2,
                    "allowed_actions": ["INSPECT", "TURN_RIGHT"],
                    "terminal_actions": [],
                    "current_step": 0,
                    "max_steps": 20,
                    "observation_image": "data:image/png;base64,robot-rgb",
                }
            )

        self.assertEqual(
            openai_constructor.call_args.kwargs["timeout"],
            90.0,
        )
        request = client.chat.completions.create.call_args.kwargs
        self.assertEqual(request["model"], "kimi-k2.6")
        self.assertEqual(request["temperature"], 1.0)
        self.assertEqual(request["max_tokens"], 2048)
        self.assertEqual(result["provider_used"], "kimi")
        self.assertTrue(result["vision_input_used"])

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
                self.assertNotEqual(response.thought, "Turn right to explore")
                self.assertNotEqual(
                    response.structured_thought["reasoning"],
                    "Turn right to explore",
                )
                self.assertIn(
                    "model_summary_present=True",
                    response.structured_thought["decision_trace"],
                )

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

    def test_agent_replans_premature_done_for_vase_box_task(self) -> None:
        agent = EmbodiedSearchAgent(self.config, model_adapter=ModelAdapter(credentials=[]))
        mock_result = {
            "thought_summary": "The vase and box are visible, so the task is complete.",
            "action": {"type": "Done", "args": {}},
            "confidence": 0.9,
        }
        with patch.object(agent.model_adapter, "plan_action", return_value=mock_result):
            with patch.object(agent.model_adapter, "available", return_value=True):
                image_path = self.config.image_dir / "ep_red_cup_visible_000.png"
                response = agent.step(
                    AgentRequest(
                        session_id="test-vase-box-premature-done",
                        instruction="把花瓶放到纸箱里",
                        observation_image=str(image_path),
                        step_id=0,
                        environment_context={
                            "agent": {"isStanding": True},
                            "objects": [
                                {
                                    "objectId": "Vase|1",
                                    "objectType": "Vase",
                                    "visible": True,
                                    "distance": 1.0,
                                    "pickupable": True,
                                },
                                {
                                    "objectId": "Box|1",
                                    "objectType": "Box",
                                    "visible": True,
                                    "distance": 1.5,
                                    "receptacle": True,
                                },
                            ],
                        },
                    )
                )

        self.assertEqual(response.planner_source, "rule_fallback")
        self.assertEqual(response.fallback_reason, "premature_done_replanned")
        self.assertEqual(response.action.type, "PickupObject")
        self.assertFalse(response.done)
        self.assertFalse(response.completion_status["complete"])

    def test_agent_replans_premature_done_for_right_door_exit(self) -> None:
        agent = EmbodiedSearchAgent(self.config, model_adapter=ModelAdapter(credentials=[]))
        mock_result = {
            "thought_summary": "The right door is visible, so the task is complete.",
            "action": {"type": "Done", "args": {}},
            "confidence": 0.9,
        }
        with patch.object(agent.model_adapter, "plan_action", return_value=mock_result):
            with patch.object(agent.model_adapter, "available", return_value=True):
                image_path = self.config.image_dir / "ep_red_cup_visible_000.png"
                response = agent.step(
                    AgentRequest(
                        session_id="test-right-door-premature-done",
                        instruction="找到右边的门，然后走出去",
                        observation_image=str(image_path),
                        step_id=0,
                        environment_context={
                            "agent": {"isStanding": True},
                            "objects": [
                                {
                                    "objectId": "Door|runtime-visible|+02.00|+00.00|+03.50",
                                    "objectType": "Door",
                                    "visible": True,
                                    "distance": 1.0,
                                    "openable": True,
                                }
                            ],
                        },
                    )
                )

        self.assertEqual(response.planner_source, "rule_fallback")
        self.assertEqual(response.fallback_reason, "premature_done_replanned")
        self.assertEqual(response.action.type, "OpenObject")
        self.assertEqual(response.action.args["objectType"], "Door")
        self.assertFalse(response.done)
        self.assertFalse(response.completion_status["complete"])

    def test_agent_replans_premature_done_for_sit_approximation(self) -> None:
        agent = EmbodiedSearchAgent(self.config, model_adapter=ModelAdapter(credentials=[]))
        mock_result = {
            "thought_summary": "The sofa is visible, so the task may be done.",
            "action": {"type": "Done", "args": {}},
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
                        environment_context={
                            "agent": {"isStanding": True},
                            "objects": [
                                {
                                    "objectId": "Sofa|1",
                                    "objectType": "Sofa",
                                    "visible": True,
                                    "distance": 1.2,
                                }
                            ],
                        },
                    )
                )
        self.assertEqual(response.action.type, "MOVE_FORWARD")
        self.assertTrue(response.task_plan["supported"])
        self.assertFalse(response.task_plan["is_visual_search"])
        self.assertFalse(response.completion_status["complete"])
        self.assertEqual(response.completion_status["outcome"], "in_progress")
        self.assertEqual(response.fallback_reason, "premature_done_replanned")
        self.assertFalse(response.done)
        self.assertIn("approach target", response.action.args["reason"])
        self.assertIn("任务尚未完成", response.thought)
        self.assertIn("任务未完成", response.structured_thought["reasoning"])

    def test_agent_replans_premature_crouch_before_approach(self) -> None:
        agent = EmbodiedSearchAgent(
            self.config,
            model_adapter=ModelAdapter(credentials=[]),
        )
        mock_result = {
            "thought_summary": "The sofa is visible, so crouch now.",
            "action": {"type": "Crouch", "args": {}},
            "confidence": 0.9,
        }
        with patch.object(
            agent.model_adapter,
            "plan_action",
            return_value=mock_result,
        ):
            with patch.object(
                agent.model_adapter,
                "available",
                return_value=True,
            ):
                image_path = (
                    self.config.image_dir
                    / "ep_red_cup_visible_000.png"
                )
                response = agent.step(
                    AgentRequest(
                        session_id="test-premature-crouch",
                        instruction="走到沙发上并坐下",
                        observation_image=str(image_path),
                        step_id=0,
                        environment_context={
                            "agent": {"isStanding": True},
                            "objects": [
                                {
                                    "objectId": "Sofa|1",
                                    "objectType": "Sofa",
                                    "visible": True,
                                    "distance": 1.2,
                                }
                            ],
                        },
                    )
                )

        self.assertEqual(response.action.type, "MOVE_FORWARD")
        self.assertEqual(
            response.fallback_reason,
            "premature_crouch_replanned",
        )
        self.assertFalse(response.done)

    def test_first_step_approach_guidance_does_not_bypass_vlm(self) -> None:
        agent = EmbodiedSearchAgent(
            self.config,
            model_adapter=ModelAdapter(credentials=[]),
        )
        mock_result = {
            "thought_summary": "Use the visual observation first, then approach the sofa.",
            "action": {"type": "TURN_LEFT", "args": {"angle": 30}},
            "skill_call": {
                "name": "TURN_LEFT",
                "args": {"angle": 30},
                "preconditions": ["current robot RGB has been inspected"],
                "expected_observation": "camera heading changes",
            },
            "confidence": 0.6,
            "provider_used": "openai",
            "model_used": "gpt-4o",
            "vision_input_used": True,
        }
        image_path = (
            self.config.image_dir / "ep_red_cup_visible_000.png"
        )

        with patch.object(agent.model_adapter, "plan_action", return_value=mock_result) as plan_action:
            with patch.object(agent.model_adapter, "available", return_value=True):
                response = agent.step(
                    AgentRequest(
                        session_id="test-first-step-approach-needs-vlm",
                        instruction="找到沙发并坐下",
                        observation_image=str(image_path),
                        step_id=0,
                        environment_context={
                            "agent": {"isStanding": True},
                            "objects": [
                                {
                                    "objectId": "Sofa|1",
                                    "objectType": "Sofa",
                                    "visible": True,
                                }
                            ],
                            "approach": {
                                "verified": False,
                                "objectId": "Sofa|1",
                                "source": "ai2thor_interactable_pose",
                                "path_status": "PathComplete",
                                "recommended_action": {
                                    "type": "TURN_RIGHT",
                                    "args": {"angle": 90.0},
                                },
                            },
                        },
                    )
                )

        plan_action.assert_called_once()
        payload = plan_action.call_args.args[0]
        self.assertTrue(payload["require_vision"])
        self.assertTrue(payload["observation_image"].startswith("data:image/"))
        self.assertEqual(response.action.type, "TURN_LEFT")
        self.assertEqual(response.planner_source, "model_planner")
        self.assertTrue(response.model_info["vision_input_used"])
        self.assertNotEqual(response.skill_call.name, "APPROACH_TARGET")

    def test_agent_uses_verified_approach_navigation_after_real_vlm_step(self) -> None:
        agent = EmbodiedSearchAgent(
            self.config,
            model_adapter=ModelAdapter(credentials=[]),
        )
        state = agent.memory.get_or_create(
            "test-approach-skill-after-vlm",
            "找到沙发并坐下",
        )
        state.steps.append(
            {
                "model_info": {
                    "status": "ok",
                    "provider": "openai",
                    "model": "gpt-4o",
                    "vision_input_used": True,
                },
                "action_success": True,
                "executed_action": {"type": "TURN_LEFT", "args": {"angle": 30}},
            }
        )
        image_path = (
            self.config.image_dir / "ep_red_cup_visible_000.png"
        )
        response = agent.step(
            AgentRequest(
                session_id="test-approach-skill-after-vlm",
                instruction="找到沙发并坐下",
                observation_image=str(image_path),
                step_id=1,
                environment_context={
                    "agent": {"isStanding": True},
                    "objects": [
                        {
                            "objectId": "Sofa|1",
                            "objectType": "Sofa",
                            "visible": True,
                        }
                    ],
                    "approach": {
                        "verified": False,
                        "objectId": "Sofa|1",
                        "source": "ai2thor_interactable_pose",
                        "path_status": "PathComplete",
                        "recommended_action": {
                            "type": "TURN_RIGHT",
                            "args": {"angle": 90.0},
                        },
                    },
                },
            )
        )

        self.assertEqual(response.action.type, "TURN_RIGHT")
        self.assertEqual(response.action.args, {"angle": 90.0})
        self.assertEqual(response.planner_source, "simulator_oracle")
        self.assertEqual(
            response.fallback_reason,
            "verified_approach_navigation",
        )
        self.assertEqual(response.skill_call.name, "APPROACH_TARGET")

    def test_interaction_approach_guidance_yields_after_repeated_successes(self) -> None:
        agent = EmbodiedSearchAgent(
            self.config,
            model_adapter=ModelAdapter(credentials=[]),
        )
        task_plan = agent.task_semantics.analyze(
            "把花瓶放到纸箱里",
            mode="default",
            legacy_actions=self.config.allowed_actions,
        )
        context = {
            "objects": [
                {
                    "objectId": "Vase|1",
                    "objectType": "Vase",
                    "visible": True,
                    "pickupable": True,
                }
            ],
            "approach": {
                "verified": False,
                "objectId": "Vase|1",
                "source": "ai2thor_interactable_pose",
                "path_status": "PathComplete",
                "recommended_action": {
                    "type": "TURN_RIGHT",
                    "args": {"angle": 4.0},
                },
            },
        }
        state = Mock(
            steps=[
                {
                    "planner_source": "simulator_oracle",
                    "fallback_reason": "verified_approach_navigation",
                    "action_success": True,
                    "executed_action": {"type": "TURN_RIGHT", "args": {"angle": 4.0}},
                },
                {
                    "planner_source": "simulator_oracle",
                    "fallback_reason": "verified_approach_navigation",
                    "action_success": True,
                    "executed_action": {"type": "TURN_LEFT", "args": {"angle": 4.0}},
                },
                {
                    "planner_source": "simulator_oracle",
                    "fallback_reason": "verified_approach_navigation",
                    "action_success": True,
                    "executed_action": {"type": "TURN_RIGHT", "args": {"angle": 4.0}},
                },
            ]
        )

        action = agent._verified_approach_action(
            task_plan=task_plan,
            completion_status={
                "approach_verified": False,
                "missing_actions": ["PickupObject", "PutObject"],
            },
            environment_context=context,
            state=state,
        )

        self.assertIsNone(action)

    def test_agent_rejects_partial_approach_path(self) -> None:
        agent = EmbodiedSearchAgent(
            self.config,
            model_adapter=ModelAdapter(credentials=[]),
        )
        image_path = (
            self.config.image_dir / "ep_red_cup_visible_000.png"
        )
        response = agent.step(
            AgentRequest(
                session_id="test-partial-approach-path",
                instruction="找到沙发并坐下",
                observation_image=str(image_path),
                step_id=0,
                environment_context={
                    "agent": {"isStanding": True},
                    "objects": [
                        {
                            "objectId": "Sofa|1",
                            "objectType": "Sofa",
                            "visible": True,
                        }
                    ],
                    "approach": {
                        "verified": False,
                        "objectId": "Sofa|1",
                        "source": "ai2thor_interactable_pose",
                        "path_status": "PathPartial",
                        "recommended_action": {
                            "type": "MOVE_FORWARD",
                            "args": {"distance": 0.25},
                        },
                    },
                },
            )
        )

        self.assertNotEqual(
            response.planner_source,
            "simulator_oracle",
        )

    def test_agent_rejects_approach_for_wrong_target(self) -> None:
        agent = EmbodiedSearchAgent(
            self.config,
            model_adapter=ModelAdapter(credentials=[]),
        )
        image_path = (
            self.config.image_dir / "ep_red_cup_visible_000.png"
        )
        response = agent.step(
            AgentRequest(
                session_id="test-wrong-approach-target",
                instruction="找到沙发并坐下",
                observation_image=str(image_path),
                step_id=0,
                environment_context={
                    "agent": {"isStanding": True},
                    "objects": [
                        {
                            "objectId": "Sofa|1",
                            "objectType": "Sofa",
                            "visible": True,
                        }
                    ],
                    "approach": {
                        "verified": False,
                        "objectId": "Chair|1",
                        "source": "ai2thor_interactable_pose",
                        "path_status": "PathComplete",
                        "recommended_action": {
                            "type": "MOVE_FORWARD",
                            "args": {"distance": 0.25},
                        },
                    },
                },
            )
        )

        self.assertNotEqual(
            response.planner_source,
            "simulator_oracle",
        )

    def test_agent_rejects_nonfinite_approach_argument(self) -> None:
        agent = EmbodiedSearchAgent(
            self.config,
            model_adapter=ModelAdapter(credentials=[]),
        )
        image_path = (
            self.config.image_dir / "ep_red_cup_visible_000.png"
        )
        response = agent.step(
            AgentRequest(
                session_id="test-nonfinite-approach-argument",
                instruction="找到沙发并坐下",
                observation_image=str(image_path),
                step_id=0,
                environment_context={
                    "agent": {"isStanding": True},
                    "objects": [
                        {
                            "objectId": "Sofa|1",
                            "objectType": "Sofa",
                            "visible": True,
                        }
                    ],
                    "approach": {
                        "verified": False,
                        "objectId": "Sofa|1",
                        "source": "ai2thor_interactable_pose",
                        "path_status": "PathComplete",
                        "recommended_action": {
                            "type": "MOVE_FORWARD",
                            "args": {"distance": float("nan")},
                        },
                    },
                },
            )
        )

        self.assertNotEqual(
            response.planner_source,
            "simulator_oracle",
        )

    def test_failed_approach_action_is_not_repeated_as_oracle(self) -> None:
        agent = EmbodiedSearchAgent(
            self.config,
            model_adapter=ModelAdapter(credentials=[]),
        )
        task_plan = agent.task_semantics.analyze(
            "找到沙发并坐下",
            mode="default",
            legacy_actions=self.config.allowed_actions,
        )
        context = {
            "objects": [
                {
                    "objectId": "Sofa|1",
                    "objectType": "Sofa",
                    "visible": True,
                }
            ],
            "approach": {
                "verified": False,
                "objectId": "Sofa|1",
                "source": "ai2thor_interactable_pose",
                "path_status": "PathComplete",
                "recommended_action": {
                    "type": "MOVE_FORWARD",
                    "args": {"distance": 0.25},
                },
            },
        }
        state = Mock(
            steps=[
                {
                    "action_success": False,
                    "executed_action": {
                        "type": "MOVE_FORWARD",
                        "args": {"distance": 0.25},
                    },
                },
                {
                    "action_success": True,
                    "executed_action": {
                        "type": "INSPECT",
                        "args": {},
                    },
                },
            ]
        )

        action = agent._verified_approach_action(
            task_plan=task_plan,
            completion_status={"approach_verified": False},
            environment_context=context,
            state=state,
        )

        self.assertIsNone(action)

    def test_interaction_task_uses_verified_approach_to_pickup_target(self) -> None:
        agent = EmbodiedSearchAgent(
            self.config,
            model_adapter=ModelAdapter(credentials=[]),
        )
        task_plan = agent.task_semantics.analyze(
            "把花瓶放到纸箱里",
            mode="default",
            legacy_actions=self.config.allowed_actions,
        )
        context = {
            "objects": [
                {
                    "objectId": "Vase|1",
                    "objectType": "Vase",
                    "visible": False,
                    "pickupable": True,
                },
                {
                    "objectId": "Box|1",
                    "objectType": "Box",
                    "visible": True,
                    "receptacle": True,
                },
            ],
            "approach": {
                "verified": False,
                "objectId": "Vase|1",
                "source": "ai2thor_interactable_pose",
                "path_status": "PathComplete",
                "recommended_action": {
                    "type": "MOVE_FORWARD",
                    "args": {"distance": 0.25},
                },
            },
        }
        state = Mock(steps=[])

        action = agent._verified_approach_action(
            task_plan=task_plan,
            completion_status={
                "approach_verified": False,
                "missing_actions": ["PickupObject", "PutObject"],
            },
            environment_context=context,
            state=state,
        )

        self.assertIsNotNone(action)
        self.assertEqual(action.type, "MOVE_FORWARD")
        self.assertEqual(action.args, {"distance": 0.25})

    def test_interaction_task_does_not_approach_receptacle_before_pickup(self) -> None:
        agent = EmbodiedSearchAgent(
            self.config,
            model_adapter=ModelAdapter(credentials=[]),
        )
        task_plan = agent.task_semantics.analyze(
            "把花瓶放到纸箱里",
            mode="default",
            legacy_actions=self.config.allowed_actions,
        )
        context = {
            "objects": [
                {
                    "objectId": "Vase|1",
                    "objectType": "Vase",
                    "visible": False,
                    "pickupable": True,
                },
                {
                    "objectId": "Box|1",
                    "objectType": "Box",
                    "visible": True,
                    "receptacle": True,
                },
            ],
            "approach": {
                "verified": False,
                "objectId": "Box|1",
                "source": "ai2thor_interactable_pose",
                "path_status": "PathComplete",
                "recommended_action": {
                    "type": "MOVE_FORWARD",
                    "args": {"distance": 0.25},
                },
            },
        }

        action = agent._verified_approach_action(
            task_plan=task_plan,
            completion_status={
                "approach_verified": False,
                "missing_actions": ["PickupObject", "PutObject"],
            },
            environment_context=context,
            state=Mock(steps=[]),
        )

        self.assertIsNone(action)

    def test_interaction_task_approaches_receptacle_after_pickup(self) -> None:
        agent = EmbodiedSearchAgent(
            self.config,
            model_adapter=ModelAdapter(credentials=[]),
        )
        task_plan = agent.task_semantics.analyze(
            "把花瓶放到纸箱里",
            mode="default",
            legacy_actions=self.config.allowed_actions,
        )
        context = {
            "objects": [
                {
                    "objectId": "Vase|1",
                    "objectType": "Vase",
                    "visible": True,
                    "pickupable": True,
                },
                {
                    "objectId": "Box|1",
                    "objectType": "Box",
                    "visible": True,
                    "receptacle": True,
                },
            ],
            "approach": {
                "verified": False,
                "objectId": "Box|1",
                "source": "ai2thor_interactable_pose",
                "path_status": "PathComplete",
                "recommended_action": {
                    "type": "TURN_RIGHT",
                    "args": {"angle": 30.0},
                },
            },
        }

        action = agent._verified_approach_action(
            task_plan=task_plan,
            completion_status={
                "approach_verified": False,
                "missing_actions": ["PutObject"],
            },
            environment_context=context,
            state=Mock(steps=[]),
        )

        self.assertIsNotNone(action)
        self.assertEqual(action.type, "TURN_RIGHT")

    def test_model_context_removes_oracle_navigation_payload(self) -> None:
        context = {
            "objects": [{"objectId": "Sofa|1"}],
            "approach": {
                "verified": False,
                "objectId": "Sofa|1",
                "source": "ai2thor_interactable_pose",
                "path_status": "PathComplete",
                "target_pose": {"x": 1.0, "z": 2.0},
                "matched_pose": {"x": 1.0, "z": 2.0},
                "recommended_action": {
                    "type": "MOVE_FORWARD",
                    "args": {"distance": 0.25},
                },
            },
        }

        sanitized = EmbodiedSearchAgent._model_environment_context(
            context
        )

        self.assertEqual(sanitized["objects"], context["objects"])
        self.assertEqual(
            sanitized["approach"]["path_status"],
            "PathComplete",
        )
        self.assertNotIn(
            "target_pose",
            sanitized["approach"],
        )
        self.assertNotIn(
            "matched_pose",
            sanitized["approach"],
        )
        self.assertNotIn(
            "recommended_action",
            sanitized["approach"],
        )
        self.assertIn(
            "recommended_action",
            context["approach"],
        )

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
