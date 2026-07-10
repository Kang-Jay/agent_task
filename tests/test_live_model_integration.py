"""Opt-in live model integration test.

Run explicitly with:
    $env:RUN_LIVE_MODEL_TESTS = "1"
    python -B -m unittest discover -s tests -p test_live_model_integration.py -v
"""
from __future__ import annotations

import os
import unittest

from src.agent.model_adapter import ModelAdapter
from src.task.config import load_config
from src.vision.image_io import image_to_data_url, load_image_from_any


@unittest.skipUnless(
    os.environ.get("RUN_LIVE_MODEL_TESTS") == "1",
    "set RUN_LIVE_MODEL_TESTS=1 to call the configured model API",
)
class LiveModelIntegrationTests(unittest.TestCase):
    def test_live_global_task_plan_uses_visual_input(self) -> None:
        config = load_config()
        adapter = ModelAdapter()
        self.assertTrue(adapter.available(), "No model credentials are configured")
        observation_image = image_to_data_url(
            load_image_from_any(
                str(config.image_dir / "ep_red_cup_visible_000.png")
            )
        )
        semantic_subgoals = [
            {
                "id": "locate_target",
                "description": "Locate and ground the requested sofa",
                "success_evidence": "visual or simulator target observation",
            },
            {
                "id": "approach_target",
                "description": "Approach the sofa",
                "success_evidence": "finite AI2-THOR target distance",
            },
            {
                "id": "execute_crouch",
                "description": "Execute Crouch",
                "success_evidence": "successful Crouch execution",
            },
            {
                "id": "verify_posture",
                "description": "Verify the crouched posture",
                "success_evidence": "agent.isStanding is false",
            },
        ]
        result = adapter.plan_task(
            {
                "instruction": "找到房间里的沙发并坐下",
                "observation_summary": "The current image is an initial observation.",
                "task_contract": {
                    "completion_mode": "approximate_sit",
                    "limitations": [
                        "native_sit_on_furniture_state_unavailable"
                    ],
                    "subgoals": semantic_subgoals,
                },
                "environment_context": {},
                "observation_image": observation_image,
                "target_crop": None,
                "require_vision": True,
            }
        )
        self.assertNotIn("error", result, result)
        self.assertEqual(
            set(result.get("ordered_subgoal_ids", [])),
            {item["id"] for item in semantic_subgoals},
        )
        self.assertEqual(len(result["ordered_subgoal_ids"]), len(semantic_subgoals))
        self.assertTrue(result.get("vision_input_used"), result)

    def test_live_planner_returns_allowed_action(self) -> None:
        config = load_config()
        adapter = ModelAdapter()
        self.assertTrue(adapter.available(), "No model credentials are configured")

        result = adapter.plan_action(
            {
                "instruction": "Find the television in the room",
                "observation_summary": (
                    "No television is currently visible. The right side of the "
                    "room has not been explored."
                ),
                "candidates": [],
                "confidence": 0.1,
                "memory_summary": "No prior steps in this session.",
                "negative_memory": [],
                "explored_regions": {},
                "retrieved_hints": [],
                "allowed_actions": config.allowed_actions,
                "terminal_actions": sorted(config.terminal_actions),
                "current_step": 0,
                "max_steps": config.max_steps,
                "observation_image": image_to_data_url(
                    load_image_from_any(
                        str(config.image_dir / "ep_red_cup_visible_000.png")
                    )
                ),
            }
        )

        self.assertNotIn("error", result, result)
        action_type = result.get("action", {}).get("type")
        self.assertIn(action_type, config.allowed_actions)
        confidence = float(result.get("confidence", 0.0))
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
        if result.get("provider_used") != "deepseek":
            self.assertTrue(result.get("vision_input_used"), result)


if __name__ == "__main__":
    unittest.main()
