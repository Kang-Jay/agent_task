from __future__ import annotations

import unittest

from src.agent.task_semantics import TaskSemantics


class TaskSemanticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.semantics = TaskSemantics()

    def test_sit_on_sofa_is_supported_only_as_approximation(self):
        plan = self.semantics.analyze(
            "走到沙发上并坐下",
            mode="default",
            legacy_actions=["ASK_CLARIFY"],
        )
        self.assertTrue(plan.supported)
        self.assertFalse(plan.is_visual_search)
        self.assertEqual(plan.completion_mode, "approximate_sit")
        self.assertIn("Crouch", plan.required_actions)
        self.assertIn("Crouch", plan.action_candidates)
        self.assertIn("native_sit_on_furniture_state_unavailable", plan.limitations)

    def test_find_sofa_and_sit_is_not_reduced_to_pure_visual_search(self):
        plan = self.semantics.analyze(
            "找到房间里的沙发并坐下",
            mode="default",
            legacy_actions=["STOP", "ASK_CLARIFY"],
        )
        self.assertTrue(plan.supported)
        self.assertFalse(plan.is_visual_search)
        self.assertIn("visual_search", plan.task_types)
        self.assertIn("navigate_to", plan.task_types)
        self.assertIn("sit_approximation", plan.task_types)
        self.assertIn("Crouch", plan.action_candidates)

    def test_visible_sofa_without_distance_is_not_approached(self):
        plan = self.semantics.analyze("找到沙发并坐下", mode="default")
        status = plan.completion_status(
            steps=[],
            target_visible=True,
            confidence=0.9,
            stop_confidence_threshold=0.78,
            environment_context={
                "agent": {"isStanding": True},
                "objects": [{"objectType": "Sofa", "visible": True}],
            },
        )
        self.assertFalse(status["complete"])
        self.assertFalse(status["approach_verified"])
        self.assertEqual(status["outcome"], "in_progress")

    def test_approached_sofa_without_crouch_is_incomplete(self):
        plan = self.semantics.analyze("找到沙发并坐下", mode="default")
        status = plan.completion_status(
            steps=[],
            target_visible=True,
            confidence=0.9,
            stop_confidence_threshold=0.78,
            environment_context={
                "agent": {"isStanding": True},
                "objects": [
                    {"objectType": "Sofa", "visible": True, "distance": 1.2}
                ],
            },
        )
        self.assertFalse(status["complete"])
        self.assertTrue(status["approach_verified"])
        self.assertIn("Crouch", status["missing_actions"])

    def test_crouched_near_sofa_is_approximate_success(self):
        plan = self.semantics.analyze("找到沙发并坐下", mode="default")
        status = plan.completion_status(
            steps=[
                {
                    "executed_action": {"type": "Crouch"},
                    "action_success": True,
                }
            ],
            target_visible=True,
            confidence=0.9,
            stop_confidence_threshold=0.78,
            environment_context={
                "agent": {"isStanding": False},
                "objects": [
                    {"objectType": "Sofa", "visible": True, "distance": 1.2}
                ],
            },
        )
        self.assertTrue(status["complete"])
        self.assertTrue(status["approximate"])
        self.assertEqual(status["outcome"], "approximate_success")

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
