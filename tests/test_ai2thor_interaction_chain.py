from __future__ import annotations

import copy
import threading
import unittest

import numpy as np

from src.simulation.ai2thor_interactions import AI2ThorInteractionResolver
from src.simulation.ai2thor_session import AI2ThorSessionManager


CABINET_ID = "Cabinet|1"
BOX_ID = "Box|1"
MUG_ID = "Mug|1"
PLATE_ID = "Plate|1"


def _object(
    *,
    object_id: str,
    object_type: str,
    distance: float,
    pickupable: bool = False,
    receptacle: bool = False,
    openable: bool = False,
    is_open: bool = False,
) -> dict[str, object]:
    return {
        "objectId": object_id,
        "objectType": object_type,
        "name": f"{object_type}_1",
        "distance": distance,
        "visible": True,
        "pickupable": pickupable,
        "receptacle": receptacle,
        "openable": openable,
        "isOpen": is_open,
        "parentReceptacles": [],
        "receptacleObjectIds": [],
    }


class _InteractionEvent:
    def __init__(self, metadata: dict[str, object]):
        self.frame = np.zeros((120, 160, 3), dtype=np.uint8)
        self.metadata = copy.deepcopy(metadata)


class _StatefulInteractionController:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.created_thread = threading.get_ident()
        self.calls: list[dict[str, object]] = []
        self.put_target_override: str | None = None
        self.metadata: dict[str, object] = {
            "lastAction": "Initialize",
            "lastActionSuccess": True,
            "errorMessage": "",
            "actionReturn": None,
            "agent": {
                "position": {"x": 0.0, "y": 0.9, "z": 0.0},
                "rotation": {"x": 0.0, "y": 0.0, "z": 0.0},
            },
            "inventoryObjects": [],
            "objects": [
                _object(
                    object_id=CABINET_ID,
                    object_type="Cabinet",
                    distance=0.8,
                    receptacle=True,
                    openable=True,
                ),
                _object(
                    object_id=BOX_ID,
                    object_type="Box",
                    distance=1.0,
                    receptacle=True,
                ),
                _object(
                    object_id=MUG_ID,
                    object_type="Mug",
                    distance=0.6,
                    pickupable=True,
                ),
                _object(
                    object_id=PLATE_ID,
                    object_type="Plate",
                    distance=0.7,
                    pickupable=True,
                ),
            ],
        }
        self.last_event = _InteractionEvent(self.metadata)

    def step(self, **kwargs):
        self.calls.append(dict(kwargs))
        action = str(kwargs["action"])
        object_id = str(kwargs.get("objectId") or "")
        self.metadata["lastAction"] = action
        self.metadata["lastActionSuccess"] = True
        self.metadata["errorMessage"] = ""

        if action == "OpenObject":
            self._find_object(object_id)["isOpen"] = True
        elif action == "PickupObject":
            target = self._find_object(object_id)
            self.metadata["inventoryObjects"] = [
                {
                    "objectId": object_id,
                    "objectType": target["objectType"],
                }
            ]
        elif action == "PutObject":
            inventory = list(self.metadata["inventoryObjects"])
            if not inventory:
                raise AssertionError("PutObject reached Unity without a held object")
            held_id = str(inventory[0]["objectId"])
            actual_receptacle_id = self.put_target_override or object_id
            held = self._find_object(held_id)
            actual_receptacle = self._find_object(actual_receptacle_id)
            held["parentReceptacles"] = [actual_receptacle_id]
            actual_receptacle["receptacleObjectIds"] = [held_id]
            self.metadata["inventoryObjects"] = []

        self.last_event = _InteractionEvent(self.metadata)
        return self.last_event

    def stop(self):
        return None

    def _find_object(self, object_id: str) -> dict[str, object]:
        for item in self.metadata["objects"]:
            if item["objectId"] == object_id:
                return item
        raise AssertionError(f"Unknown test objectId: {object_id}")


class AI2ThorInteractionChainTests(unittest.TestCase):
    def setUp(self):
        self.controllers: list[_StatefulInteractionController] = []

        def factory(**kwargs):
            controller = _StatefulInteractionController(**kwargs)
            self.controllers.append(controller)
            return controller

        self.manager = AI2ThorSessionManager(controller_factory=factory)
        self.resolver = AI2ThorInteractionResolver()
        self.manager.start(
            session_id="interaction-chain",
            scene="FloorPlan1",
            mode="default",
            width=320,
            height=240,
        )

    def tearDown(self):
        self.manager.close_all()

    @property
    def controller(self) -> _StatefulInteractionController:
        return self.controllers[0]

    def _resolve_and_execute(
        self,
        *,
        action: str,
        args: dict[str, object],
        instruction: str,
    ) -> dict[str, object]:
        binding = self.resolver.resolve(
            action=action,
            args=args,
            instruction=instruction,
            metadata=self.controller.last_event.metadata,
        )
        self.assertTrue(binding.valid, binding.errors)
        result = self.manager.execute(
            session_id="interaction-chain",
            action=action,
            args=binding.args,
            actor="agent",
        )
        self.assertTrue(result["execution"]["success"])
        return result

    def _execute_until_held(self) -> tuple[dict[str, object], dict[str, object]]:
        instruction = "Open the cabinet, pick up the mug, and put it in the cabinet"
        opened = self._resolve_and_execute(
            action="OpenObject",
            args={"objectType": "Cabinet"},
            instruction=instruction,
        )
        picked_up = self._resolve_and_execute(
            action="PickupObject",
            args={"objectType": "Mug"},
            instruction=instruction,
        )
        return opened, picked_up

    def test_open_pickup_put_chain_binds_objects_and_verifies_every_transition(self):
        opened, picked_up = self._execute_until_held()

        self.assertEqual(opened["execution"]["args"], {"objectId": CABINET_ID})
        self.assertTrue(opened["postcondition"]["checked"])
        self.assertTrue(opened["postcondition"]["passed"])
        self.assertEqual(
            opened["postcondition"]["evidence"],
            {
                "objectId": CABINET_ID,
                "beforeIsOpen": False,
                "afterIsOpen": True,
            },
        )

        self.assertEqual(picked_up["execution"]["args"], {"objectId": MUG_ID})
        self.assertEqual(
            picked_up["inventory_objects"],
            [{"objectId": MUG_ID, "objectType": "Mug"}],
        )
        self.assertTrue(picked_up["postcondition"]["checked"])
        self.assertTrue(picked_up["postcondition"]["passed"])
        self.assertEqual(
            picked_up["postcondition"]["evidence"]["inventoryObjectIds"],
            [MUG_ID],
        )

        placed = self._resolve_and_execute(
            action="PutObject",
            args={"receptacleType": "Cabinet"},
            instruction="Put the held mug in the cabinet",
        )

        self.assertEqual(placed["execution"]["args"], {"objectId": CABINET_ID})
        self.assertEqual(placed["inventory_objects"], [])
        self.assertTrue(placed["postcondition"]["checked"])
        self.assertTrue(placed["postcondition"]["passed"])
        self.assertEqual(
            placed["postcondition"]["evidence"]["releasedObjectIds"],
            [MUG_ID],
        )
        self.assertEqual(
            placed["postcondition"]["evidence"]["placedObjectIds"],
            [MUG_ID],
        )
        self.assertEqual(
            placed["postcondition"]["evidence"]["receptacleObjectIds"],
            [MUG_ID],
        )
        self.assertEqual(
            self.controller.calls,
            [
                {"action": "OpenObject", "objectId": CABINET_ID},
                {"action": "PickupObject", "objectId": MUG_ID},
                {"action": "PutObject", "objectId": CABINET_ID},
            ],
        )

    def test_put_postcondition_rejects_runtime_success_in_wrong_receptacle(self):
        self._execute_until_held()
        self.controller.put_target_override = BOX_ID

        placed = self._resolve_and_execute(
            action="PutObject",
            args={"receptacleType": "Cabinet"},
            instruction="Put the held mug in the cabinet",
        )

        self.assertTrue(placed["execution"]["success"])
        self.assertFalse(placed["postcondition"]["passed"])
        self.assertEqual(
            placed["postcondition"]["evidence"]["receptacleObjectId"],
            CABINET_ID,
        )
        self.assertEqual(
            placed["postcondition"]["evidence"]["releasedObjectIds"],
            [MUG_ID],
        )
        self.assertEqual(
            placed["postcondition"]["evidence"]["placedObjectIds"],
            [],
        )
        box = self.controller._find_object(BOX_ID)
        self.assertEqual(box["receptacleObjectIds"], [MUG_ID])

    def test_put_resolution_rejects_when_requested_object_is_not_held(self):
        self.controller.metadata["inventoryObjects"] = [
            {"objectId": PLATE_ID, "objectType": "Plate"}
        ]
        self.controller.last_event = _InteractionEvent(self.controller.metadata)

        binding = self.resolver.resolve(
            action="PutObject",
            args={
                "object": "Mug",
                "receptacleType": "Cabinet",
            },
            instruction="Put the mug in the cabinet",
            metadata=self.controller.last_event.metadata,
        )

        self.assertFalse(
            binding.valid,
            "PutObject must reject a task that requests a mug while a plate is held",
        )


if __name__ == "__main__":
    unittest.main()
