from __future__ import annotations

import unittest

from src.simulation.ai2thor_actions import AI2ThorActionCatalog, AI2ThorActionExecutor


class _FakeEvent:
    def __init__(self, success: bool = True):
        self.metadata = {
            "lastActionSuccess": success,
            "errorMessage": "" if success else "failed",
            "actionReturn": {"ok": success},
        }


class _FakeController:
    def __init__(self):
        self.calls: list[dict[str, object]] = []

    def step(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeEvent()


class AI2ThorActionCatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = AI2ThorActionCatalog()

    def test_catalog_contains_all_supported_modes(self):
        metadata = self.catalog.summary()
        self.assertIn("default", metadata["mode_controllers"])
        self.assertIn("locobot", metadata["mode_controllers"])
        self.assertIn("drone", metadata["mode_controllers"])
        self.assertGreater(metadata["counts"]["actions"], 300)

    def test_default_navigation_alias_validates(self):
        validation = self.catalog.validate(
            mode="default",
            action="TURN_RIGHT",
            args={},
            actor="agent",
        )
        self.assertTrue(validation.valid, validation.errors)
        self.assertEqual(validation.action, "RotateRight")
        self.assertFalse(
            [
                parameter
                for parameter in validation.matched_overload["parameters"]
                if parameter["required"]
            ]
        )

    def test_abstract_navigation_arguments_are_normalized(self):
        move = self.catalog.validate(
            mode="default",
            action="MOVE_FORWARD",
            args={"distance": 0.25},
            actor="agent",
        )
        turn = self.catalog.validate(
            mode="default",
            action="TURN_RIGHT",
            args={"angle": 30},
            actor="agent",
        )

        self.assertTrue(move.valid, move.errors)
        self.assertEqual(move.normalized_args, {"moveMagnitude": 0.25})
        self.assertTrue(turn.valid, turn.errors)
        self.assertEqual(turn.normalized_args, {"degrees": 30})

    def test_drone_only_action_is_mode_gated(self):
        default_validation = self.catalog.validate(
            mode="default",
            action="FlyUp",
            args={"moveMagnitude": 0.25},
            actor="agent",
        )
        drone_validation = self.catalog.validate(
            mode="drone",
            action="FlyUp",
            args={"moveMagnitude": 0.25},
            actor="manual",
        )
        self.assertFalse(default_validation.valid)
        self.assertTrue(drone_validation.valid, drone_validation.errors)

    def test_system_action_is_not_available_to_agent(self):
        validation = self.catalog.validate(
            mode="default",
            action="TeleportFull",
            args={},
            actor="agent",
        )
        self.assertFalse(validation.valid)
        self.assertIn("requires actor level", " ".join(validation.errors))

    def test_internal_method_is_never_executable(self):
        validation = self.catalog.validate(
            mode="default",
            action="TestActionDispatchNoop",
            args={},
            actor="system",
        )
        self.assertFalse(validation.valid)
        self.assertIn("internal Unity method", " ".join(validation.errors))

    def test_parameter_type_validation(self):
        validation = self.catalog.validate(
            mode="drone",
            action="FlyUp",
            args={"moveMagnitude": "fast"},
            actor="manual",
        )
        self.assertFalse(validation.valid)
        self.assertIn("must be numeric", " ".join(validation.errors))

    def test_executor_uses_native_action_and_arguments(self):
        controller = _FakeController()
        execution = AI2ThorActionExecutor(self.catalog).execute(
            controller,
            mode="drone",
            action="FlyUp",
            args={"moveMagnitude": 0.25},
            actor="manual",
        )
        self.assertTrue(execution.success)
        self.assertEqual(
            controller.calls,
            [{"action": "FlyUp", "moveMagnitude": 0.25}],
        )

    def test_executor_translates_abstract_navigation_arguments(self):
        controller = _FakeController()
        execution = AI2ThorActionExecutor(self.catalog).execute(
            controller,
            mode="default",
            action="MOVE_FORWARD",
            args={"distance": 0.25},
            actor="agent",
        )

        self.assertTrue(execution.success)
        self.assertEqual(
            controller.calls,
            [{"action": "MoveAhead", "moveMagnitude": 0.25}],
        )

    def test_runtime_identity_must_match_catalog(self):
        result = self.catalog.verify_runtime(
            ai2thor_version="5.0.0",
            build_commit="f0825767cd50d69f666c7f282e54abfe58f1e917",
        )
        self.assertTrue(result["matched"])
        with self.assertRaisesRegex(RuntimeError, "Unity build mismatch"):
            self.catalog.verify_runtime(
                ai2thor_version="5.0.0",
                build_commit="wrong-build",
            )

    def test_all_public_runtime_actions_are_listed_in_at_least_one_mode(self):
        listed_actions = {
            action["name"]
            for mode in self.catalog.summary()["mode_controllers"]
            for action in self.catalog.list_actions(
                mode=mode,
                actor="system",
                include_internal=False,
            )
        }
        expected_actions = {
            action["name"]
            for action in self.catalog.actions
            if action["runtime_available"] and action["exposure"] != "internal"
        }

        self.assertEqual(listed_actions, expected_actions)
        self.assertEqual(len(expected_actions), 333)


if __name__ == "__main__":
    unittest.main()
