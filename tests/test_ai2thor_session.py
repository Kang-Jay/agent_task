from __future__ import annotations

import unittest
import threading

import numpy as np

from src.simulation.ai2thor_session import AI2ThorSessionManager
from src.simulation.ai2thor_runtime import (
    is_grid_aligned_rotation,
    should_snap_to_grid,
)


class _FakeEvent:
    def __init__(self, action: str = "Initialize"):
        self.frame = np.zeros((120, 160, 3), dtype=np.uint8)
        self.metadata = {
            "lastAction": action,
            "lastActionSuccess": True,
            "errorMessage": "",
            "actionReturn": None,
            "agent": {
                "position": {"x": 0.0, "y": 0.9, "z": 0.0},
                "rotation": {"x": 0.0, "y": 0.0, "z": 0.0},
            },
            "inventoryObjects": [],
            "objects": [
                {
                    "objectId": "Sofa|1",
                    "objectType": "Sofa",
                    "visible": True,
                    "distance": 1.2,
                    "pickupable": False,
                    "receptacle": False,
                    "openable": False,
                    "toggleable": False,
                }
            ],
        }


class _FakeController:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.created_thread = threading.get_ident()
        self.last_event = _FakeEvent()
        self.calls: list[dict[str, object]] = []
        self.call_threads: list[int] = []
        self.stopped = False
        self.stop_thread: int | None = None

    def step(self, **kwargs):
        self.call_threads.append(threading.get_ident())
        self.calls.append(kwargs)
        self.last_event = _FakeEvent(str(kwargs["action"]))
        if kwargs["action"] == "MoveAhead":
            self.last_event.metadata["agent"]["position"]["z"] = 0.25
        return self.last_event

    def stop(self):
        self.stop_thread = threading.get_ident()
        self.stopped = True


class AI2ThorSessionManagerTests(unittest.TestCase):
    def setUp(self):
        self.controllers: list[_FakeController] = []

        def factory(**kwargs):
            controller = _FakeController(**kwargs)
            self.controllers.append(controller)
            return controller

        self.manager = AI2ThorSessionManager(controller_factory=factory)

    def tearDown(self):
        self.manager.close_all()

    def test_start_returns_real_session_snapshot_shape(self):
        snapshot = self.manager.start(
            session_id="demo",
            scene="FloorPlan211",
            mode="default",
            width=320,
            height=240,
        )
        self.assertEqual(snapshot["mode"], "default")
        self.assertTrue(snapshot["frame"].startswith("data:image/"))
        self.assertEqual(snapshot["visible_objects"][0]["objectType"], "Sofa")
        self.assertEqual(self.controllers[0].kwargs["agentMode"], "default")
        self.assertTrue(self.controllers[0].kwargs["snapToGrid"])

    def test_thirty_degree_rotation_disables_grid_snapping(self):
        self.manager.start(
            session_id="continuous-turn",
            scene="FloorPlan211",
            mode="default",
            rotate_step_degrees=30,
        )
        self.assertFalse(self.controllers[0].kwargs["snapToGrid"])

    def test_grid_snapping_requires_default_mode_and_grid_aligned_rotation(self):
        self.assertTrue(is_grid_aligned_rotation(90))
        self.assertTrue(is_grid_aligned_rotation(-90))
        self.assertFalse(is_grid_aligned_rotation(30))
        self.assertTrue(
            should_snap_to_grid(mode="default", rotate_step_degrees=90)
        )
        self.assertFalse(
            should_snap_to_grid(mode="default", rotate_step_degrees=30)
        )
        self.assertFalse(
            should_snap_to_grid(mode="locobot", rotate_step_degrees=90)
        )

    def test_execute_preserves_session_state(self):
        self.manager.start(session_id="demo", scene="FloorPlan211", mode="default")
        result = self.manager.execute(
            session_id="demo",
            action="MOVE_FORWARD",
            actor="agent",
        )
        self.assertEqual(result["last_action"], "MoveAhead")
        self.assertTrue(result["execution"]["success"])
        self.assertTrue(result["postcondition"]["passed"])
        self.assertEqual(self.controllers[0].calls, [{"action": "MoveAhead"}])

    def test_mode_specific_action_is_rejected(self):
        self.manager.start(session_id="demo", scene="FloorPlan211", mode="default")
        with self.assertRaisesRegex(ValueError, "not available"):
            self.manager.execute(
                session_id="demo",
                action="FlyUp",
                args={"moveMagnitude": 0.25},
                actor="agent",
            )

    def test_close_stops_controller(self):
        self.manager.start(session_id="demo", scene="FloorPlan211", mode="default")
        self.assertTrue(self.manager.close("demo"))
        self.assertTrue(self.controllers[0].stopped)

    def test_controller_lifecycle_stays_on_one_worker_thread(self):
        self.manager.start(session_id="demo", scene="FloorPlan211", mode="default")
        self.manager.execute(
            session_id="demo",
            action="MOVE_FORWARD",
            actor="agent",
        )
        self.manager.close("demo")

        controller = self.controllers[0]
        self.assertEqual(controller.call_threads, [controller.created_thread])
        self.assertEqual(controller.stop_thread, controller.created_thread)


if __name__ == "__main__":
    unittest.main()
