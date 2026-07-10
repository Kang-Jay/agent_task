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
