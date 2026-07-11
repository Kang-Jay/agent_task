from __future__ import annotations

import copy
import unittest
import threading

import numpy as np

from src.simulation.ai2thor_session import AI2ThorSessionManager
from src.simulation.ai2thor_runtime import (
    is_grid_aligned_rotation,
    should_snap_to_grid,
)


class _FakeEvent:
    def __init__(
        self,
        action: str = "Initialize",
        *,
        agent: dict[str, object] | None = None,
        inventory: list[dict[str, object]] | None = None,
        objects: list[dict[str, object]] | None = None,
        success: bool = True,
        error_message: str = "",
    ):
        self.frame = np.zeros((120, 160, 3), dtype=np.uint8)
        self.metadata = {
            "lastAction": action,
            "lastActionSuccess": success,
            "errorMessage": error_message,
            "actionReturn": None,
            "agent": copy.deepcopy(agent) if agent is not None else {
                "position": {"x": 0.0, "y": 0.9, "z": 0.0},
                "rotation": {"x": 0.0, "y": 0.0, "z": 0.0},
            },
            "inventoryObjects": copy.deepcopy(inventory or []),
            "objects": copy.deepcopy(objects) if objects is not None else [
                {
                    "objectId": "Sofa|1",
                    "objectType": "Sofa",
                    "visible": True,
                    "distance": 1.2,
                    "pickupable": False,
                    "receptacle": False,
                    "openable": False,
                    "toggleable": False,
                },
                {
                    "objectId": "Box|1",
                    "objectType": "Box",
                    "visible": True,
                    "distance": 0.8,
                    "pickupable": False,
                    "receptacle": True,
                    "openable": True,
                    "isOpen": False,
                    "toggleable": False,
                    "receptacleObjectIds": [],
                    "parentReceptacles": [],
                    "isPickedUp": False,
                },
                {
                    "objectId": "Vase|1",
                    "objectType": "Vase",
                    "visible": True,
                    "distance": 0.7,
                    "pickupable": True,
                    "receptacle": False,
                    "openable": False,
                    "toggleable": False,
                    "receptacleObjectIds": [],
                    "parentReceptacles": [],
                    "isPickedUp": False,
                },
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
        self.agent = copy.deepcopy(self.last_event.metadata["agent"])
        self.inventory: list[dict[str, object]] = []
        self.objects = copy.deepcopy(self.last_event.metadata["objects"])
        self.fail_actions: set[str] = set()
        self.suppress_state_change_actions: set[str] = set()

    def step(self, **kwargs):
        self.call_threads.append(threading.get_ident())
        self.calls.append(kwargs)
        action = str(kwargs["action"])
        success = action not in self.fail_actions
        if success and action not in self.suppress_state_change_actions:
            self._apply_action(action, kwargs)
        self.last_event = _FakeEvent(
            action,
            agent=self.agent,
            inventory=self.inventory,
            objects=self.objects,
            success=success,
            error_message="" if success else "forced fake controller failure",
        )
        return self.last_event

    def _apply_action(self, action: str, args: dict[str, object]) -> None:
        if action == "MoveAhead":
            self.agent["position"]["z"] = 0.25
            return
        object_id = args.get("objectId")
        if action == "OpenObject":
            self._object(str(object_id))["isOpen"] = True
            return
        if action == "PickupObject":
            target = self._object(str(object_id))
            target["isPickedUp"] = True
            target["parentReceptacles"] = []
            self.inventory = [
                {
                    "objectId": target["objectId"],
                    "objectType": target["objectType"],
                }
            ]
            return
        if action == "PutObject":
            receptacle = self._object(str(object_id))
            held_ids = [str(item["objectId"]) for item in self.inventory]
            self.inventory = []
            receptacle["receptacleObjectIds"] = held_ids
            for held_id in held_ids:
                held = self._object(held_id)
                held["isPickedUp"] = False
                held["parentReceptacles"] = [receptacle["objectId"]]

    def _object(self, object_id: str) -> dict[str, object]:
        return next(
            item for item in self.objects if item["objectId"] == object_id
        )

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
        self.assertTrue(result["execution"]["committed"])
        self.assertTrue(result["committed"])
        self.assertEqual(self.controllers[0].calls, [{"action": "MoveAhead"}])

    def test_runtime_success_without_postcondition_does_not_commit(self):
        self.manager.start(session_id="demo", scene="FloorPlan211", mode="default")
        self.controllers[0].suppress_state_change_actions.add("OpenObject")

        result = self.manager.execute(
            session_id="demo",
            action="OpenObject",
            args={"objectId": "Box|1"},
            actor="agent",
        )

        self.assertTrue(result["execution"]["success"])
        self.assertFalse(result["postcondition"]["passed"])
        self.assertFalse(result["execution"]["committed"])
        self.assertFalse(result["committed"])
        self.assertFalse(
            next(
                item
                for item in result["visible_objects"]
                if item["objectId"] == "Box|1"
            )["isOpen"]
        )

    def test_runtime_failure_does_not_commit(self):
        self.manager.start(session_id="demo", scene="FloorPlan211", mode="default")
        self.controllers[0].fail_actions.add("OpenObject")

        result = self.manager.execute(
            session_id="demo",
            action="OpenObject",
            args={"objectId": "Box|1"},
            actor="agent",
        )

        self.assertFalse(result["execution"]["success"])
        self.assertFalse(result["postcondition"]["passed"])
        self.assertFalse(result["committed"])

    def test_open_pickup_put_chain_preserves_after_state(self):
        self.manager.start(session_id="demo", scene="FloorPlan211", mode="default")

        opened = self.manager.execute(
            session_id="demo",
            action="OpenObject",
            args={"objectId": "Box|1"},
            actor="agent",
        )
        self.assertTrue(opened["committed"])
        opened_box = next(
            item
            for item in opened["visible_objects"]
            if item["objectId"] == "Box|1"
        )
        self.assertTrue(opened_box["isOpen"])

        picked_up = self.manager.execute(
            session_id="demo",
            action="PickupObject",
            args={"objectId": "Vase|1"},
            actor="agent",
        )
        self.assertTrue(picked_up["committed"])
        self.assertEqual(
            picked_up["inventory_objects"],
            [{"objectId": "Vase|1", "objectType": "Vase"}],
        )
        picked_vase = next(
            item
            for item in picked_up["visible_objects"]
            if item["objectId"] == "Vase|1"
        )
        self.assertTrue(picked_vase["isPickedUp"])

        put = self.manager.execute(
            session_id="demo",
            action="PutObject",
            args={"objectId": "Box|1"},
            actor="agent",
        )
        self.assertTrue(put["committed"])
        self.assertEqual(put["inventory_objects"], [])
        put_box = next(
            item
            for item in put["visible_objects"]
            if item["objectId"] == "Box|1"
        )
        put_vase = next(
            item
            for item in put["visible_objects"]
            if item["objectId"] == "Vase|1"
        )
        self.assertEqual(put_box["receptacleObjectIds"], ["Vase|1"])
        self.assertEqual(put_vase["parentReceptacles"], ["Box|1"])
        self.assertFalse(put_vase["isPickedUp"])

        final_snapshot = self.manager.snapshot("demo")
        self.assertEqual(final_snapshot["inventory_objects"], [])
        final_box = next(
            item
            for item in final_snapshot["visible_objects"]
            if item["objectId"] == "Box|1"
        )
        final_vase = next(
            item
            for item in final_snapshot["visible_objects"]
            if item["objectId"] == "Vase|1"
        )
        self.assertEqual(final_box["receptacleObjectIds"], ["Vase|1"])
        self.assertEqual(final_vase["parentReceptacles"], ["Box|1"])

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
