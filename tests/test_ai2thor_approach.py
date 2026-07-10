from __future__ import annotations

import unittest
from types import SimpleNamespace

from src.simulation.ai2thor_approach import AI2ThorApproachVerifier


class FakeController:
    def __init__(
        self,
        poses: list[dict[str, object]],
        *,
        success: bool = True,
        path_results: list[dict[str, object]] | None = None,
    ) -> None:
        self.poses = poses
        self.success = success
        self.path_results = path_results
        self.path_call_count = 0
        self.calls: list[dict[str, object]] = []

    def step(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("action") == "GetShortestPathToPoint":
            if self.path_results is None:
                action_return = {
                    "corners": [
                        {"x": 1.0, "y": 0.0, "z": 2.0},
                        {"x": 2.0, "y": 0.0, "z": 2.0},
                    ],
                    "status": "PathComplete",
                }
            else:
                index = min(
                    self.path_call_count,
                    len(self.path_results) - 1,
                )
                action_return = self.path_results[index]
            self.path_call_count += 1
        else:
            action_return = self.poses
        return SimpleNamespace(
            metadata={
                "lastActionSuccess": self.success,
                "errorMessage": "" if self.success else "query failed",
                "actionReturn": action_return,
            }
        )


class AI2ThorApproachVerifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.verifier = AI2ThorApproachVerifier()
        self.agent = {
            "position": {"x": 1.0, "y": 0.9, "z": 2.0},
            "rotation": {"x": 0.0, "y": 90.0, "z": 0.0},
            "cameraHorizon": 0.0,
            "isStanding": True,
        }
        self.pose = {
            "x": 1.0,
            "y": 0.9,
            "z": 2.0,
            "rotation": 90.0,
            "horizon": 0.0,
            "standing": True,
        }

    def test_matching_interactable_pose_verifies_approach(self) -> None:
        controller = FakeController([self.pose])
        result = self.verifier.verify(
            controller,
            mode="default",
            metadata={"agent": self.agent},
            object_id="Sofa|1",
        )

        self.assertTrue(result.verified)
        self.assertEqual(result.objectId, "Sofa|1")
        self.assertEqual(result.candidate_count, 1)
        self.assertEqual(
            controller.calls,
            [
                {
                    "action": "GetInteractablePoses",
                    "objectId": "Sofa|1",
                    "standings": [True],
                    "maxPoses": 64,
                }
            ],
        )

    def test_nonmatching_pose_does_not_verify_approach(self) -> None:
        controller = FakeController(
            [{**self.pose, "z": 2.25}]
        )
        result = self.verifier.verify(
            controller,
            mode="default",
            metadata={"agent": self.agent},
            object_id="Sofa|1",
        )

        self.assertFalse(result.verified)
        self.assertEqual(result.candidate_count, 1)

    def test_nonmatching_pose_returns_shortest_path_action(self) -> None:
        controller = FakeController(
            [
                {
                    **self.pose,
                    "x": 2.0,
                    "z": 2.0,
                }
            ]
        )
        result = self.verifier.verify(
            controller,
            mode="default",
            metadata={"agent": self.agent},
            object_id="Sofa|1",
        )

        self.assertFalse(result.verified)
        self.assertEqual(result.path_status, "PathComplete")
        self.assertEqual(
            result.recommended_action,
            {"type": "MOVE_FORWARD", "args": {"distance": 0.25}},
        )
        self.assertEqual(
            controller.calls[1],
            {
                "action": "GetShortestPathToPoint",
                "target": {"x": 2.0, "y": 0.9, "z": 2.0},
            },
        )

    def test_partial_path_does_not_return_navigation_action(self) -> None:
        controller = FakeController(
            [{**self.pose, "x": 2.0}],
            path_results=[
                {
                    "corners": [
                        {"x": 1.0, "y": 0.0, "z": 2.0},
                        {"x": 1.5, "y": 0.0, "z": 2.0},
                    ],
                    "status": "PathPartial",
                }
            ],
        )
        result = self.verifier.verify(
            controller,
            mode="default",
            metadata={"agent": self.agent},
            object_id="Sofa|1",
        )

        self.assertFalse(result.verified)
        self.assertEqual(result.path_status, "PathPartial")
        self.assertIsNone(result.recommended_action)
        self.assertIsNone(result.target_pose)

    def test_malformed_path_corner_does_not_return_action(self) -> None:
        controller = FakeController(
            [{**self.pose, "x": 2.0}],
            path_results=[
                {
                    "corners": [
                        {"x": 1.0, "z": 2.0},
                        {"x": float("nan"), "z": 2.0},
                    ],
                    "status": "PathComplete",
                }
            ],
        )
        result = self.verifier.verify(
            controller,
            mode="default",
            metadata={"agent": self.agent},
            object_id="Sofa|1",
        )

        self.assertFalse(result.verified)
        self.assertEqual(result.path_status, "PathComplete")
        self.assertIsNone(result.recommended_action)

    def test_unreachable_nearest_pose_uses_next_complete_candidate(self) -> None:
        near_pose = {**self.pose, "x": 1.25}
        farther_pose = {**self.pose, "x": 2.0}
        controller = FakeController(
            [farther_pose, near_pose],
            path_results=[
                {
                    "corners": [{"x": 1.25, "z": 2.0}],
                    "status": "PathPartial",
                },
                {
                    "corners": [
                        {"x": 1.0, "z": 2.0},
                        {"x": 2.0, "z": 2.0},
                    ],
                    "status": "PathComplete",
                },
            ],
        )
        result = self.verifier.verify(
            controller,
            mode="default",
            metadata={"agent": self.agent},
            object_id="Sofa|1",
        )

        self.assertEqual(result.target_pose, farther_pose)
        self.assertEqual(result.path_status, "PathComplete")
        self.assertEqual(
            result.recommended_action,
            {"type": "MOVE_FORWARD", "args": {"distance": 0.25}},
        )
        self.assertEqual(
            [call["target"] for call in controller.calls[1:]],
            [
                {"x": 1.25, "y": 0.9, "z": 2.0},
                {"x": 2.0, "y": 0.9, "z": 2.0},
            ],
        )

    def test_position_match_returns_pose_alignment_action(self) -> None:
        controller = FakeController(
            [{**self.pose, "rotation": 180.0}]
        )
        result = self.verifier.verify(
            controller,
            mode="default",
            metadata={"agent": self.agent},
            object_id="Sofa|1",
        )

        self.assertFalse(result.verified)
        self.assertEqual(result.path_status, "PoseAlignment")
        self.assertEqual(
            result.recommended_action,
            {"type": "TURN_RIGHT", "args": {"angle": 90.0}},
        )

    def test_missing_posture_does_not_verify_approach(self) -> None:
        controller = FakeController([self.pose])
        agent = dict(self.agent)
        agent.pop("isStanding")
        result = self.verifier.verify(
            controller,
            mode="default",
            metadata={"agent": agent},
            object_id="Sofa|1",
        )

        self.assertFalse(result.verified)
        self.assertEqual(controller.calls, [])

    def test_failed_pose_query_does_not_verify_approach(self) -> None:
        controller = FakeController([], success=False)
        result = self.verifier.verify(
            controller,
            mode="default",
            metadata={"agent": self.agent},
            object_id="Sofa|1",
        )

        self.assertFalse(result.verified)
        self.assertIn("query failed", result.reason)


if __name__ == "__main__":
    unittest.main()
