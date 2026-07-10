from __future__ import annotations

import unittest
from types import SimpleNamespace

from tools.validate_ai2thor_sofa_approximation import (
    _interactable_pose,
    _select_sofa,
    _select_standing_pose,
)


class SofaApproximationValidationTests(unittest.TestCase):
    def test_select_sofa_is_deterministic(self) -> None:
        selected = _select_sofa(
            {
                "objects": [
                    {"objectId": "Sofa|2", "objectType": "Sofa"},
                    {"objectId": "Chair|1", "objectType": "Chair"},
                    {"objectId": "Sofa|1", "objectType": "Sofa"},
                ]
            }
        )
        self.assertEqual(selected["objectId"], "Sofa|1")

    def test_select_sofa_rejects_scene_without_sofa(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "does not contain a Sofa"):
            _select_sofa({"objects": [{"objectType": "Chair"}]})

    def test_select_standing_pose_skips_crouched_pose(self) -> None:
        selected = _select_standing_pose(
            [
                {
                    "x": 0.0,
                    "y": 0.9,
                    "z": 0.0,
                    "rotation": 0.0,
                    "horizon": 0.0,
                    "standing": False,
                },
                {
                    "x": 0.0,
                    "y": 0.9,
                    "z": 0.25,
                    "rotation": 0.0,
                    "horizon": 0.0,
                    "standing": True,
                },
            ]
        )
        self.assertTrue(selected["standing"])
        self.assertEqual(selected["z"], 0.25)

    def test_select_standing_pose_rejects_crouched_only_candidates(self) -> None:
        with self.assertRaisesRegex(
            RuntimeError,
            "No standing interactable pose",
        ):
            _select_standing_pose([{"standing": False}])

    def test_interactable_pose_requests_standing_candidates(self) -> None:
        pose = {
            "x": 0.0,
            "y": 0.9,
            "z": 0.25,
            "rotation": 0.0,
            "horizon": 0.0,
            "standing": True,
        }

        class FakeController:
            def __init__(self) -> None:
                self.kwargs = {}

            def step(self, **kwargs):
                self.kwargs = kwargs
                return SimpleNamespace(
                    metadata={
                        "lastActionSuccess": True,
                        "actionReturn": [pose],
                    }
                )

        controller = FakeController()
        selected = _interactable_pose(controller, "Sofa|1")
        self.assertEqual(selected, pose)
        self.assertEqual(controller.kwargs["standings"], [True])
        self.assertEqual(controller.kwargs["maxPoses"], 64)


if __name__ == "__main__":
    unittest.main()
