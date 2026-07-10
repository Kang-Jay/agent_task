from __future__ import annotations

import unittest

from src.agent.task_semantics import TaskSemantics


class TaskSemanticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.semantics = TaskSemantics()

    def test_sit_on_sofa_is_not_falsely_supported(self):
        plan = self.semantics.analyze(
            "走到沙发上并坐下",
            mode="default",
            legacy_actions=["ASK_CLARIFY"],
        )
        self.assertFalse(plan.supported)
        self.assertFalse(plan.is_visual_search)
        self.assertIn("human_or_robot_sitting_pose", plan.unsupported_capabilities)
        self.assertEqual(plan.action_candidates, ("ASK_CLARIFY",))

    def test_find_sofa_and_sit_is_not_reduced_to_visual_search(self):
        plan = self.semantics.analyze(
            "找到房间里的沙发并坐下",
            mode="default",
            legacy_actions=["STOP", "ASK_CLARIFY"],
        )
        self.assertFalse(plan.supported)
        self.assertFalse(plan.is_visual_search)
        self.assertEqual(plan.action_candidates, ("ASK_CLARIFY",))

    def test_pickup_task_exposes_navigation_and_pickup(self):
        plan = self.semantics.analyze(
            "走到桌边拿起杯子",
            mode="default",
            legacy_actions=["MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "ASK_CLARIFY"],
        )
        self.assertTrue(plan.supported)
        self.assertIn("navigate_to", plan.task_types)
        self.assertIn("PickupObject", plan.required_actions)
        self.assertIn("PickupObject", plan.action_candidates)
        self.assertIn("MOVE_FORWARD", plan.action_candidates)

    def test_drone_mode_keeps_continuous_flight_actions_out_of_planner(self):
        plan = self.semantics.analyze(
            "找到房间里的杯子",
            mode="drone",
            legacy_actions=["STOP", "ASK_CLARIFY"],
        )
        self.assertNotIn("FlyUp", plan.action_candidates)
        self.assertNotIn("FlyAhead", plan.action_candidates)
        manual_actions = {
            action["name"]
            for action in self.semantics.catalog.list_actions(
                mode="drone",
                actor="manual",
            )
        }
        self.assertIn("FlyUp", manual_actions)

    def test_successful_required_action_completes_interaction(self):
        plan = self.semantics.analyze(
            "拿起杯子",
            mode="default",
            legacy_actions=["ASK_CLARIFY"],
        )
        status = plan.completion_status(
            steps=[
                {
                    "executed_action": {"type": "PickupObject"},
                    "action_success": True,
                }
            ],
            target_visible=True,
            confidence=0.9,
            stop_confidence_threshold=0.78,
        )
        self.assertTrue(status["complete"])
        self.assertEqual(status["missing_actions"], [])


if __name__ == "__main__":
    unittest.main()
