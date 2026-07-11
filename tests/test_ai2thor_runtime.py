from __future__ import annotations

import unittest

from src.simulation.ai2thor_runtime import (
    create_controller_safely,
    execute_controller_action,
)


class _FailingController:
    last_instance = None

    def __init__(self, **kwargs):
        del kwargs
        type(self).last_instance = self
        self.stopped = False
        raise RuntimeError("initialize failed")

    def stop(self):
        self.stopped = True


class _Event:
    def __init__(
        self,
        *,
        action: str,
        success: bool = True,
        inventory: list[dict[str, object]] | None = None,
        error_message: str = "",
        action_return: object = None,
    ):
        self.metadata = {
            "lastAction": action,
            "lastActionSuccess": success,
            "errorMessage": error_message,
            "actionReturn": action_return,
            "inventoryObjects": list(inventory or []),
            "objects": [],
        }


class _RecordingController:
    def __init__(self):
        self.calls: list[dict[str, object]] = []
        self.last_event = _Event(action="Initialize")

    def step(self, **kwargs):
        self.calls.append(kwargs)
        action = str(kwargs["action"])
        inventory = self.last_event.metadata["inventoryObjects"]
        if action == "PickupObject":
            inventory = [{"objectId": kwargs["objectId"]}]
        elif action == "PutObject":
            inventory = []
        self.last_event = _Event(
            action=action,
            inventory=inventory,
            action_return={"received": dict(kwargs)},
        )
        return self.last_event


class _UnityFailureController(_RecordingController):
    def step(self, **kwargs):
        self.calls.append(kwargs)
        self.last_event = _Event(
            action=str(kwargs["action"]),
            success=False,
            inventory=[{"objectId": "Vase|1"}],
            error_message="Object is not interactable",
        )
        return self.last_event


class _RaisingController(_RecordingController):
    def step(self, **kwargs):
        self.calls.append(kwargs)
        raise ConnectionError("Unity process disconnected")


class AI2ThorRuntimeTests(unittest.TestCase):
    def test_failed_initialization_stops_partial_controller(self):
        with self.assertRaisesRegex(RuntimeError, "initialize failed"):
            create_controller_safely(_FailingController, scene="FloorPlan1")
        self.assertIsNotNone(_FailingController.last_instance)
        self.assertTrue(_FailingController.last_instance.stopped)

    def test_open_object_forwards_native_name_and_args_unchanged(self):
        controller = _RecordingController()
        args = {
            "objectId": "Cabinet|1",
            "openness": 0.75,
            "forceAction": False,
        }

        execution = execute_controller_action(
            controller,
            action="OpenObject",
            args=args,
        )

        self.assertEqual(
            controller.calls,
            [{"action": "OpenObject", **args}],
        )
        self.assertEqual(execution.action, "OpenObject")
        self.assertEqual(execution.args, args)
        self.assertTrue(execution.success)
        self.assertEqual(execution.before_metadata["lastAction"], "Initialize")
        self.assertEqual(execution.after_metadata["lastAction"], "OpenObject")

    def test_pickup_object_records_inventory_transition(self):
        controller = _RecordingController()

        execution = execute_controller_action(
            controller,
            action="PickupObject",
            args={"objectId": "Vase|1", "forceAction": False},
        )

        self.assertEqual(execution.inventory_before, [])
        self.assertEqual(
            execution.inventory_after,
            [{"objectId": "Vase|1"}],
        )
        self.assertTrue(execution.to_dict()["last_action_success"])

    def test_put_object_preserves_receptacle_id_and_inventory_release(self):
        controller = _RecordingController()
        controller.last_event = _Event(
            action="PickupObject",
            inventory=[{"objectId": "Vase|1"}],
        )
        args = {"objectId": "Box|1", "forceAction": False}

        execution = execute_controller_action(
            controller,
            action="PutObject",
            args=args,
        )

        self.assertEqual(
            controller.calls,
            [{"action": "PutObject", **args}],
        )
        self.assertEqual(
            execution.inventory_before,
            [{"objectId": "Vase|1"}],
        )
        self.assertEqual(execution.inventory_after, [])

    def test_open_pickup_put_chain_retains_each_audit_boundary(self):
        controller = _RecordingController()

        opened = execute_controller_action(
            controller,
            action="OpenObject",
            args={"objectId": "Box|1", "forceAction": False},
        )
        picked = execute_controller_action(
            controller,
            action="PickupObject",
            args={"objectId": "Vase|1", "forceAction": False},
        )
        placed = execute_controller_action(
            controller,
            action="PutObject",
            args={"objectId": "Box|1", "forceAction": False},
        )

        self.assertTrue(opened.success)
        self.assertEqual(picked.inventory_after, [{"objectId": "Vase|1"}])
        self.assertEqual(placed.inventory_before, picked.inventory_after)
        self.assertEqual(placed.inventory_after, [])
        self.assertEqual(
            [call["action"] for call in controller.calls],
            ["OpenObject", "PickupObject", "PutObject"],
        )

    def test_unity_failure_is_returned_with_error_and_inventory(self):
        controller = _UnityFailureController()

        execution = execute_controller_action(
            controller,
            action="PutObject",
            args={"objectId": "Box|1"},
        )

        self.assertFalse(execution.success)
        self.assertEqual(
            execution.error_message,
            "Object is not interactable",
        )
        self.assertEqual(
            execution.inventory_after,
            [{"objectId": "Vase|1"}],
        )

    def test_controller_exception_propagates_without_rewriting(self):
        controller = _RaisingController()

        with self.assertRaisesRegex(
            ConnectionError,
            "Unity process disconnected",
        ):
            execute_controller_action(
                controller,
                action="PickupObject",
                args={"objectId": "Vase|1"},
            )

    def test_runtime_detaches_args_and_metadata_from_mutation(self):
        controller = _RecordingController()
        args = {"objectId": "Cabinet|1", "metadata": {"source": "planner"}}

        execution = execute_controller_action(
            controller,
            action="OpenObject",
            args=args,
        )
        args["metadata"]["source"] = "mutated"
        controller.last_event.metadata["inventoryObjects"].append(
            {"objectId": "Injected|1"}
        )

        self.assertEqual(execution.args["metadata"]["source"], "planner")
        self.assertEqual(execution.inventory_after, [])

    def test_reserved_action_argument_is_rejected_before_unity_call(self):
        controller = _RecordingController()

        with self.assertRaisesRegex(ValueError, "reserved 'action' key"):
            execute_controller_action(
                controller,
                action="OpenObject",
                args={"action": "PickupObject", "objectId": "Cabinet|1"},
            )

        self.assertEqual(controller.calls, [])


if __name__ == "__main__":
    unittest.main()
