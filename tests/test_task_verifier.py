from __future__ import annotations

import unittest

from src.agent.task_semantics import TaskPlan, TaskSemantics
from src.simulation.task_verifier import TaskVerifier


class TaskVerifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.verifier = TaskVerifier()
        cls.semantics = TaskSemantics()

    def test_exact_visual_search_success(self) -> None:
        plan = self.semantics.analyze("找到电视", mode="default")
        result = self.verifier.verify(
            plan,
            steps=[],
            target_visible=True,
            confidence=0.9,
            stop_confidence_threshold=0.78,
        )
        self.assertTrue(result.complete)
        self.assertEqual(result.outcome, "exact_success")
        self.assertTrue(result.evidence_ledger[0]["passed"])

    def test_sofa_completion_is_approximate_success(self) -> None:
        plan = self.semantics.analyze("找到沙发并坐下", mode="default")
        result = self.verifier.verify(
            plan,
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
        self.assertTrue(result.complete)
        self.assertEqual(result.outcome, "approximate_success")
        self.assertTrue(all(item["passed"] for item in result.evidence_ledger))

    def test_unsupported_task_has_distinct_outcome(self) -> None:
        plan = TaskPlan(
            instruction="fly through the wall",
            mode="default",
            task_types=("unsupported",),
            required_actions=(),
            unsupported_capabilities=("wall_phasing",),
            limitations=(),
            completion_mode="exact",
            subgoals=(),
            completion_rule="unsupported",
            clarification="wall phasing is unsupported",
            action_candidates=("ASK_CLARIFY",),
            action_specs=(),
        )
        result = self.verifier.verify(
            plan,
            steps=[],
            target_visible=False,
            confidence=0.0,
            stop_confidence_threshold=0.78,
        )
        self.assertFalse(result.complete)
        self.assertEqual(result.outcome, "unsupported")

    def test_failed_and_terminated_are_not_success(self) -> None:
        plan = self.semantics.analyze("找到电视", mode="default")
        failed = self.verifier.verify(
            plan,
            steps=[],
            target_visible=False,
            confidence=0.0,
            stop_confidence_threshold=0.78,
            failed=True,
            failure_reason="navigation action failed",
        )
        terminated = self.verifier.verify(
            plan,
            steps=[],
            target_visible=False,
            confidence=0.0,
            stop_confidence_threshold=0.78,
            terminated=True,
            termination_reason="max steps reached",
        )
        self.assertEqual(failed.outcome, "failed")
        self.assertEqual(terminated.outcome, "terminated")
        self.assertFalse(failed.complete)
        self.assertFalse(terminated.complete)


if __name__ == "__main__":
    unittest.main()
