from __future__ import annotations

import unittest

from src.simulation.ai2thor_postconditions import AI2ThorPostconditionVerifier


class AI2ThorPostconditionVerifierTests(unittest.TestCase):
    def setUp(self):
        self.verifier = AI2ThorPostconditionVerifier()

    def test_pickup_requires_inventory_change(self):
        before = {"inventoryObjects": [], "objects": []}
        after = {
            "inventoryObjects": [{"objectId": "Mug|1"}],
            "objects": [],
        }
        result = self.verifier.verify(
            action="PickupObject",
            args={"objectId": "Mug|1"},
            before=before,
            after=after,
            runtime_success=True,
        )
        self.assertTrue(result.checked)
        self.assertTrue(result.passed)

    def test_put_requires_object_in_requested_receptacle(self):
        result = self.verifier.verify(
            action="PutObject",
            args={"objectId": "Bowl|1"},
            before={
                "inventoryObjects": [{"objectId": "Egg|1"}],
                "objects": [],
            },
            after={
                "inventoryObjects": [],
                "objects": [
                    {
                        "objectId": "Egg|1",
                        "parentReceptacles": ["Bowl|1", "CounterTop|1"],
                    },
                    {
                        "objectId": "Bowl|1",
                        "receptacleObjectIds": ["Egg|1"],
                    },
                ],
            },
            runtime_success=True,
        )
        self.assertTrue(result.passed)
        self.assertEqual(result.evidence["placedObjectIds"], ["Egg|1"])

    def test_put_fails_when_inventory_releases_into_different_receptacle(self):
        result = self.verifier.verify(
            action="PutObject",
            args={"objectId": "Bowl|1"},
            before={
                "inventoryObjects": [{"objectId": "Egg|1"}],
                "objects": [],
            },
            after={
                "inventoryObjects": [],
                "objects": [
                    {
                        "objectId": "Egg|1",
                        "parentReceptacles": ["Plate|1"],
                    },
                    {
                        "objectId": "Bowl|1",
                        "receptacleObjectIds": [],
                    },
                    {
                        "objectId": "Plate|1",
                        "receptacleObjectIds": ["Egg|1"],
                    },
                ],
            },
            runtime_success=True,
        )
        self.assertFalse(result.passed)

    def test_open_requires_object_state_change(self):
        before = {"objects": [{"objectId": "Cabinet|1", "isOpen": False}]}
        after = {"objects": [{"objectId": "Cabinet|1", "isOpen": True}]}
        result = self.verifier.verify(
            action="OpenObject",
            args={"objectId": "Cabinet|1"},
            before=before,
            after=after,
            runtime_success=True,
        )
        self.assertTrue(result.passed)
        self.assertEqual(result.evidence["actual"], True)

    def test_runtime_failure_always_fails(self):
        result = self.verifier.verify(
            action="MoveAhead",
            args={},
            before={},
            after={"errorMessage": "collision"},
            runtime_success=False,
        )
        self.assertTrue(result.checked)
        self.assertFalse(result.passed)

    def test_crouch_requires_explicit_boolean_posture(self):
        missing = self.verifier.verify(
            action="Crouch",
            args={},
            before={"agent": {"isStanding": True}},
            after={"agent": {}},
            runtime_success=True,
        )
        crouched = self.verifier.verify(
            action="Crouch",
            args={},
            before={"agent": {"isStanding": True}},
            after={"agent": {"isStanding": False}},
            runtime_success=True,
        )

        self.assertFalse(missing.passed)
        self.assertTrue(crouched.passed)

    def test_unregistered_postcondition_is_explicitly_unchecked(self):
        result = self.verifier.verify(
            action="GetSceneBounds",
            args={},
            before={},
            after={},
            runtime_success=True,
        )
        self.assertFalse(result.checked)
        self.assertTrue(result.passed)


if __name__ == "__main__":
    unittest.main()
