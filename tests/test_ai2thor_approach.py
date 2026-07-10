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
    ) -> None:
        self.poses = poses
        self.success = success
        self.calls: list[dict[str, object]] = []

    def step(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            metadata={
                "lastActionSuccess": self.success,
                "errorMessage": "" if self.success else "query failed",
                "actionReturn": self.poses,
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
