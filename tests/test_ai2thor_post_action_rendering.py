from __future__ import annotations

import base64
import io
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image, ImageFont

import src.simulation.ai2thor_adapter as adapter_module
from src.simulation.ai2thor_adapter import AI2ThorVisualSearchDemo


def _agent_metadata(z: float) -> dict[str, object]:
    return {
        "position": {"x": 0.0, "y": 0.9, "z": z},
        "rotation": {"x": 0.0, "y": 0.0, "z": 0.0},
        "cameraHorizon": 0.0,
        "isStanding": True,
    }


class _FakeEvent:
    def __init__(
        self,
        color: tuple[int, int, int],
        *,
        z: float,
        action_return: object = None,
    ) -> None:
        self.frame = np.full((120, 160, 3), color, dtype=np.uint8)
        self.instance_masks: dict[str, np.ndarray] = {}
        self.third_party_camera_frames: list[np.ndarray] = []
        self.metadata = {
            "lastAction": "Initialize",
            "lastActionSuccess": True,
            "errorMessage": "",
            "actionReturn": action_return,
            "agent": _agent_metadata(z),
            "inventoryObjects": [],
            "objects": [],
        }


class _FakeExecution:
    def __init__(
        self,
        event: _FakeEvent,
        action: str,
        args: dict[str, object] | None = None,
    ) -> None:
        self.event = event
        self.action = action
        self.args = args or {}
        self.success = True

    def to_dict(self) -> dict[str, object]:
        return {
            "action": self.action,
            "args": dict(self.args),
            "success": self.success,
            "error_message": "",
        }


class _FakeActionExecutor:
    def __init__(
        self,
        reachable_event: _FakeEvent,
        post_action_event: _FakeEvent,
    ) -> None:
        self.reachable_event = reachable_event
        self.post_action_event = post_action_event

    def execute(
        self,
        controller: object,
        *,
        mode: str,
        action: str,
        args: dict[str, object] | None = None,
        actor: str,
    ) -> _FakeExecution:
        del controller, mode, actor
        if action == "GetReachablePositions":
            return _FakeExecution(
                self.reachable_event,
                action,
                args,
            )
        return _FakeExecution(self.post_action_event, action, args)


class _FakeTaskPlan:
    task_types: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "supported": True,
            "is_visual_search": False,
            "task_types": [],
        }


class _FakeAgentResponse:
    def to_dict(self) -> dict[str, object]:
        return {
            "thought": "Move toward the target.",
            "structured_thought": None,
            "action": {
                "type": "MOVE_FORWARD",
                "args": {"moveMagnitude": 0.25},
            },
            "confidence": 0.91,
            "done": False,
            "observation": {"best_candidate": None},
            "planner_source": "model_planner",
            "model_info": {"model": "fake-vlm"},
            "task_plan": {
                "supported": True,
                "is_visual_search": False,
            },
            "completion_status": {"complete": False},
            "skill_call": None,
            "memory_summary": "",
            "recalled_memories": [],
            "search_map": None,
            "fallback_reason": None,
            "target_binding": None,
        }


class _FakeAgent:
    def __init__(self) -> None:
        self.task_semantics = SimpleNamespace(
            analyze=lambda *args, **kwargs: _FakeTaskPlan()
        )
        self.model_adapter = SimpleNamespace(
            audit=lambda: {"model": "fake-vlm"}
        )
        self.requests: list[object] = []

    def reset(self, session_id: str) -> None:
        self.session_id = session_id

    def step(self, request: object) -> _FakeAgentResponse:
        self.requests.append(request)
        return _FakeAgentResponse()

    def commit_execution(
        self,
        session_id: str,
        response: dict[str, object],
        **kwargs: object,
    ) -> dict[str, object]:
        del session_id, response, kwargs
        return {
            "done": True,
            "completion_status": {"complete": True},
        }


class _FakeController:
    def __init__(self, initial_event: _FakeEvent) -> None:
        self.last_event = initial_event
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True


class AI2ThorPostActionRenderingTests(unittest.TestCase):
    def test_compose_frame_names_decision_and_observation_phases(
        self,
    ) -> None:
        demo = AI2ThorVisualSearchDemo.__new__(
            AI2ThorVisualSearchDemo
        )
        demo.scene = "FloorPlan211"
        response = {
            "action": {"type": "MOVE_FORWARD"},
            "confidence": 0.9,
            "thought": "Move.",
            "execution": {
                "success": True,
                "postcondition": {"passed": True},
            },
        }
        draw = MagicMock()
        draw.textbbox.side_effect = (
            lambda _position, text, font=None: (
                0,
                0,
                len(str(text)) * 8,
                18,
            )
        )
        font_loader = MagicMock(return_value=ImageFont.load_default())
        with (
            patch.object(adapter_module.ImageDraw, "Draw", return_value=draw),
            patch.object(
                adapter_module,
                "load_render_font",
                font_loader,
            ),
        ):
            demo._compose_frame(
                Image.new("RGB", (160, 120), "blue"),
                Image.new("RGB", (160, 120), "white"),
                response,
                "Find the sofa",
                0,
                [],
            )

        rendered_text = [
            str(call.args[1])
            for call in draw.text.call_args_list
            if len(call.args) > 1
        ]
        self.assertIn(
            "Observation after action: MOVE_FORWARD",
            rendered_text,
        )
        self.assertIn(
            "Decision before action | Step 0",
            rendered_text,
        )
        self.assertIn(
            "Selected action: MOVE_FORWARD",
            rendered_text,
        )
        self.assertEqual(
            [call.args[0] for call in font_loader.call_args_list],
            [24, 18, 16],
        )

    def test_run_demo_keeps_pre_action_audit_and_displays_post_action_state(
        self,
    ) -> None:
        initial_event = _FakeEvent((255, 0, 0), z=0.0)
        reachable_event = _FakeEvent(
            (255, 0, 0),
            z=0.0,
            action_return=[],
        )
        post_action_event = _FakeEvent((0, 0, 255), z=0.25)
        controller = _FakeController(initial_event)
        agent = _FakeAgent()
        demo = AI2ThorVisualSearchDemo(
            scene="FloorPlan211",
            agent=agent,
        )
        demo.action_executor = _FakeActionExecutor(
            reachable_event,
            post_action_event,
        )
        demo.interaction_resolver = SimpleNamespace(
            build_context=lambda metadata: {}
        )
        demo.postconditions = SimpleNamespace(
            verify=lambda **kwargs: SimpleNamespace(
                passed=True,
                to_dict=lambda: {"passed": True},
            )
        )
        demo.action_catalog.verify_installed_runtime = lambda: None
        emitted: list[dict[str, object]] = []
        rendered_paths: list[list[dict[str, float]]] = []

        def render_map(
            *args: object,
            **kwargs: object,
        ) -> Image.Image:
            del args
            rendered_paths.append(
                list(kwargs.get("agent_path") or [])
            )
            return Image.new("RGB", (420, 420), "white")

        fake_ai2thor = types.ModuleType("ai2thor")
        fake_controller_module = types.ModuleType("ai2thor.controller")
        fake_controller_module.Controller = object
        fake_platform_module = types.ModuleType("ai2thor.platform")
        fake_platform_module.CloudRendering = object()
        fake_platform_module.Linux64 = object()
        fake_ai2thor.controller = fake_controller_module
        fake_ai2thor.platform = fake_platform_module

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_dir = root / "docs" / "ai2thor_outputs"
            module_patch = {
                "ai2thor": fake_ai2thor,
                "ai2thor.controller": fake_controller_module,
                "ai2thor.platform": fake_platform_module,
            }
            with (
                patch.dict(sys.modules, module_patch),
                patch.object(adapter_module, "ROOT", root),
                patch.object(
                    adapter_module,
                    "AI2THOR_OUTPUT_DIR",
                    output_dir,
                ),
                patch.object(
                    adapter_module,
                    "create_controller_safely",
                    return_value=controller,
                ) as create_controller,
                patch.dict(os.environ, {"AI2THOR_PLATFORM": "Linux64"}),
                patch.object(
                    demo,
                    "_initialize_map_camera",
                    return_value=(initial_event, None, None),
                ),
                patch.object(
                    demo,
                    "_ground_target_from_segmentation",
                    return_value=None,
                ),
                patch.object(
                    demo,
                    "_render_unity_map_view",
                    side_effect=render_map,
                ),
                patch.object(demo, "_write_video"),
            ):
                result = demo.run_demo(
                    "Move toward the sofa",
                    max_steps=1,
                    session_id="post-action-test",
                    episode_id="episode",
                    emit=emitted.append,
                )

            self.assertEqual(len(result.steps), 1)
            self.assertIs(
                create_controller.call_args.kwargs["platform"],
                fake_platform_module.Linux64,
            )
            step = result.steps[0]
            self.assertTrue(
                step.observation_path.endswith(
                    "ai2thor_obs_after_00.png"
                )
            )
            self.assertEqual(step.robot["y"], 0.25)
            self.assertEqual(rendered_paths[-1][-1]["y"], 0.25)

            run_dir = output_dir / "post-action-test" / "episode"
            pre_path = (
                run_dir
                / "frames"
                / "ai2thor_obs_00.png"
            )
            post_path = root / step.observation_path
            frame_path = root / step.frame_path
            self.assertTrue(pre_path.exists())
            self.assertTrue(post_path.exists())
            self.assertEqual(
                Image.open(pre_path).getpixel((0, 0)),
                (255, 0, 0),
            )
            self.assertEqual(
                Image.open(post_path).getpixel((0, 0)),
                (0, 0, 255),
            )
            self.assertEqual(
                Image.open(frame_path).getpixel((100, 200)),
                (0, 0, 255),
            )

            request_data_url = agent.requests[0].observation_image
            encoded = request_data_url.split(",", 1)[1]
            model_input = Image.open(
                io.BytesIO(base64.b64decode(encoded))
            ).convert("RGB")
            self.assertEqual(
                model_input.getpixel((0, 0)),
                (255, 0, 0),
            )

            observation_event = next(
                event
                for event in emitted
                if event["event"] == "observation_ready"
            )
            self.assertEqual(
                observation_event["payload"]["observation_phase"],
                "before_action",
            )
            self.assertEqual(
                observation_event["payload"]["purpose"],
                "model_input_audit",
            )
            feedback_event = next(
                event
                for event in emitted
                if event["event"] == "environment_feedback"
            )
            self.assertEqual(
                feedback_event["payload"]["observation_phase"],
                "after_action",
            )
            self.assertTrue(
                str(
                    feedback_event["payload"]["observation_path"]
                ).endswith(
                    "ai2thor_obs_after_00.png"
                )
            )
            self.assertTrue(controller.stopped)

    def test_terminal_decision_label_does_not_claim_transition(
        self,
    ) -> None:
        label = AI2ThorVisualSearchDemo._post_action_observation_label(
            {
                "action": {"type": "STOP"},
                "execution": None,
            }
        )
        self.assertIn("Observation after action decision", label)
        self.assertIn("no simulator transition", label)

    def test_failed_execution_label_does_not_claim_success(
        self,
    ) -> None:
        label = AI2ThorVisualSearchDemo._post_action_observation_label(
            {
                "action": {"type": "MOVE_FORWARD"},
                "execution": {
                    "success": True,
                    "postcondition": {"passed": False},
                },
            }
        )
        self.assertEqual(
            label,
            "Observation after action attempt: MOVE_FORWARD (failed)",
        )

    def test_fallback_map_labels_post_action_phase(self) -> None:
        demo = AI2ThorVisualSearchDemo.__new__(
            AI2ThorVisualSearchDemo
        )
        demo.scene = "FloorPlan211"
        draw = MagicMock()
        with patch.object(
            adapter_module.ImageDraw,
            "Draw",
            return_value=draw,
        ):
            demo._render_topdown(
                {
                    "agent": {
                        "position": {"x": 0.0, "z": 0.25},
                        "rotation": {"y": 30.0},
                    },
                    "objects": [],
                },
                None,
                instruction="找到沙发",
                reachable_positions=[],
                agent_path=[
                    {"x": 0.0, "y": 0.0},
                    {"x": 0.0, "y": 0.25},
                ],
                planned_action="TURN_RIGHT",
            )

        rendered_text = [
            str(call.args[1])
            for call in draw.text.call_args_list
            if len(call.args) > 1
        ]
        self.assertTrue(
            any(
                "after TURN_RIGHT" in text
                for text in rendered_text
            )
        )
        self.assertFalse(
            any(
                "before TURN_RIGHT" in text
                for text in rendered_text
            )
        )


if __name__ == "__main__":
    unittest.main()
