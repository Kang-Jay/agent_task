"""Tests for Phase 4 AI2-THOR structured_thought sync.

According to Plan_1_agent_demo_repair.md Phase 4 requirements.
"""
from __future__ import annotations

import unittest
from unittest.mock import Mock, patch, MagicMock

from src.simulation.ai2thor_adapter import AI2ThorVisualSearchDemo
from src.task.config import load_config


class AI2ThorStructuredThoughtTests(unittest.TestCase):
    def test_non_search_clarification_is_not_overridden_by_search_fallback(self):
        self.assertFalse(
            AI2ThorVisualSearchDemo._should_force_search(
                visual_search_task=False,
                action_type="ASK_CLARIFY",
            )
        )
        self.assertTrue(
            AI2ThorVisualSearchDemo._should_force_search(
                visual_search_task=True,
                action_type="ASK_CLARIFY",
            )
        )

    """Test AI2-THOR adapter syncs structured_thought when overriding actions."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config()

    def test_apply_grounded_target_updates_structured_thought(self) -> None:
        """Test _apply_grounded_target() syncs structured_thought."""
        demo = AI2ThorVisualSearchDemo(scene="FloorPlan1")

        response = {
            "action": {"type": "TURN_RIGHT", "args": {}},
            "confidence": 0.5,
            "done": False,
            "thought": "Original thought",
            "structured_thought": {"observation": "Original", "reasoning": "Original", "action": "向右转", "confidence": "0.500"},
            "observation": {"image_size": [448, 448], "scene_summary": ""}
        }

        target = {
            "object_type": "Television",
            "region": "middle center",
            "confidence": 0.95,
            "label": "television",
            "bbox": [100, 100, 200, 200],
            "color_name": "black",
            "reason": "segmentation",
            "image_size": [448, 448]
        }

        demo._apply_grounded_target(response, target, "STOP", 2)

        # Verify action was overridden
        self.assertEqual(response["action"]["type"], "STOP")
        self.assertTrue(response["done"])
        self.assertEqual(response["planner_source"], "simulator_oracle")
        self.assertEqual(response["skill_call"]["name"], "STOP")

        # Verify structured_thought was synced
        self.assertIn("structured_thought", response)
        st = response["structured_thought"]
        self.assertIn("Television", st["observation"])
        self.assertIn("0.95", st["observation"])
        self.assertIn("停止", st["action"])
        self.assertIn("0.950", st["confidence"])

    def test_apply_search_response_updates_structured_thought(self) -> None:
        """Test _apply_search_response() syncs structured_thought."""
        demo = AI2ThorVisualSearchDemo(scene="FloorPlan1")

        response = {
            "action": {"type": "STOP", "args": {}},
            "confidence": 0.8,
            "done": True,
            "thought": "Original thought",
            "structured_thought": {"observation": "Original", "reasoning": "Original", "action": "停止", "confidence": "0.800"},
            "observation": {"target_visible": True, "scene_summary": "Target visible", "best_candidate": {}, "candidates": [{}]}
        }

        demo._apply_search_response(response, "TURN_LEFT")

        # Verify action was overridden
        self.assertEqual(response["action"]["type"], "TURN_LEFT")
        self.assertFalse(response["done"])
        self.assertEqual(response["planner_source"], "simulator_oracle")
        self.assertEqual(response["skill_call"]["name"], "TURN_LEFT")

        # Verify structured_thought was synced
        self.assertIn("structured_thought", response)
        st = response["structured_thought"]
        self.assertIn("实例分割", st["observation"])
        self.assertIn("向左转", st["action"])
        self.assertIn(str(response["confidence"])[:5], st["confidence"])

        # Verify observation was reset
        self.assertFalse(response["observation"]["target_visible"])
        self.assertIsNone(response["observation"]["best_candidate"])
        self.assertEqual(response["observation"]["candidates"], [])

    def test_structured_thought_includes_chinese_action_name(self) -> None:
        """Test structured_thought uses Chinese action names."""
        demo = AI2ThorVisualSearchDemo(scene="FloorPlan1")

        response = {
            "action": {"type": "MOVE_FORWARD", "args": {}},
            "confidence": 0.3,
            "done": False,
            "thought": "Test",
            "structured_thought": {},
            "observation": {"target_visible": False, "scene_summary": "", "best_candidate": None, "candidates": []}
        }

        demo._apply_search_response(response, "MOVE_FORWARD")

        self.assertEqual(response["structured_thought"]["action"], "向前移动")

    def test_confidence_synced_between_response_and_structured_thought(self) -> None:
        """Test confidence value is consistent in response and structured_thought."""
        demo = AI2ThorVisualSearchDemo(scene="FloorPlan1")

        target = {
            "object_type": "Cup",
            "region": "middle center",
            "confidence": 0.87,
            "label": "cup",
            "bbox": [100, 100, 150, 150],
            "color_name": "red",
            "reason": "segmentation",
            "image_size": [448, 448]
        }

        response = {
            "action": {"type": "TURN_RIGHT", "args": {}},
            "confidence": 0.5,
            "done": False,
            "thought": "Test",
            "structured_thought": {},
            "observation": {"image_size": [448, 448], "scene_summary": ""}
        }

        demo._apply_grounded_target(response, target, "INSPECT", 1)

        # Confidence should be updated to target confidence
        self.assertEqual(response["confidence"], 0.87)
        # structured_thought should reflect the same
        self.assertIn("0.870", response["structured_thought"]["confidence"])

    def test_heading_triangle_uses_ai2thor_clockwise_yaw(self) -> None:
        """AI2-THOR yaw must look clockwise on the screen-space map."""
        center = (100, 100)

        nose_0, _, _ = AI2ThorVisualSearchDemo._heading_triangle(*center, 0)
        nose_90, _, _ = AI2ThorVisualSearchDemo._heading_triangle(*center, 90)
        nose_180, _, _ = AI2ThorVisualSearchDemo._heading_triangle(*center, 180)
        nose_270, _, _ = AI2ThorVisualSearchDemo._heading_triangle(*center, 270)

        self.assertLess(nose_0[1], center[1], "0 degrees should point up (+Z)")
        self.assertGreater(nose_90[0], center[0], "90 degrees should point right")
        self.assertGreater(nose_180[1], center[1], "180 degrees should point down")
        self.assertLess(nose_270[0], center[0], "270 degrees should point left")

    def test_unity_map_projection_uses_camera_center_and_positive_z_up(
        self,
    ) -> None:
        properties = {
            "position": {"x": 0.0, "y": 3.0, "z": 0.0},
            "orthographicSize": 2.5,
        }
        center = AI2ThorVisualSearchDemo._project_unity_map_point(
            x=0.0,
            z=0.0,
            camera_properties=properties,
            size=520,
        )
        right = AI2ThorVisualSearchDemo._project_unity_map_point(
            x=1.0,
            z=0.0,
            camera_properties=properties,
            size=520,
        )
        up = AI2ThorVisualSearchDemo._project_unity_map_point(
            x=0.0,
            z=1.0,
            camera_properties=properties,
            size=520,
        )

        self.assertEqual(center, (260, 260))
        self.assertGreater(right[0], center[0])
        self.assertLess(up[1], center[1])

    def test_topdown_map_renders_reachable_space_path_and_target(self) -> None:
        demo = AI2ThorVisualSearchDemo(scene="FloorPlan211")
        metadata = {
            "agent": {
                "position": {"x": 0.0, "y": 0.9, "z": 0.0},
                "rotation": {"x": 0.0, "y": 90.0, "z": 0.0},
            },
            "objects": [
                {
                    "objectType": "Floor",
                    "objectId": "Floor|0",
                    "position": {"x": 0.0, "y": 0.0, "z": 0.0},
                    "visible": True,
                },
                {
                    "objectType": "Television",
                    "objectId": "Television|1",
                    "position": {"x": 1.0, "y": 1.0, "z": 0.5},
                    "visible": True,
                },
            ],
        }
        image = demo._render_topdown(
            metadata,
            {
                "label": "Television",
                "object_id": "Television|1",
                "confidence": 0.943,
            },
            instruction="find the television",
            reachable_positions=[
                {"x": 0.0, "y": 0.9, "z": 0.0},
                {"x": 0.5, "y": 0.9, "z": 0.0},
            ],
            agent_path=[
                {"x": -0.5, "y": 0.0, "heading": 0.0},
                {"x": 0.0, "y": 0.0, "heading": 90.0},
            ],
            planned_action="TURN_RIGHT",
        )

        self.assertEqual(image.size, (520, 520))
        colors = {color for _, color in (image.getcolors(maxcolors=520 * 520) or [])}
        self.assertGreater(len(colors), 6)
        self.assertIn((226, 60, 46), colors)

    def test_topdown_map_does_not_leak_unseen_target_location(self) -> None:
        demo = AI2ThorVisualSearchDemo(scene="FloorPlan211")
        metadata = {
            "agent": {
                "position": {"x": 0.0, "y": 0.9, "z": 0.0},
                "rotation": {"x": 0.0, "y": 90.0, "z": 0.0},
            },
            "objects": [
                {
                    "objectType": "Television",
                    "objectId": "Television|hidden",
                    "position": {"x": 1.0, "y": 1.0, "z": 0.5},
                    "visible": False,
                },
            ],
        }

        image = demo._render_topdown(
            metadata,
            None,
            instruction="find the television",
            reachable_positions=[{"x": 0.0, "y": 0.9, "z": 0.0}],
            agent_path=[{"x": 0.0, "y": 0.0, "heading": 90.0}],
            planned_action="TURN_RIGHT",
        )
        colors = {color for _, color in (image.getcolors(maxcolors=520 * 520) or [])}

        self.assertNotIn((226, 60, 46), colors)
        self.assertNotIn((244, 142, 38), colors)


if __name__ == "__main__":
    unittest.main()
