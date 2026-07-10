from __future__ import annotations

import unittest

from src.agent.model_adapter import ModelAdapter
from src.simulation.ai2thor_interactions import AI2ThorInteractionResolver


class AI2ThorInteractionResolverTests(unittest.TestCase):
    def setUp(self):
        self.resolver = AI2ThorInteractionResolver()
        self.metadata = {
            "agent": {"position": {"x": 0, "y": 0.9, "z": 0}},
            "inventoryObjects": [],
            "objects": [
                {
                    "objectId": "Mug|near",
                    "objectType": "Mug",
                    "name": "Mug_1",
                    "distance": 0.8,
                    "visible": True,
                    "pickupable": True,
                    "receptacle": False,
                    "openable": False,
                },
                {
                    "objectId": "Mug|far",
                    "objectType": "Mug",
                    "name": "Mug_2",
                    "distance": 2.0,
                    "visible": False,
                    "pickupable": True,
                    "receptacle": False,
                    "openable": False,
                },
                {
                    "objectId": "Cabinet|1",
                    "objectType": "Cabinet",
                    "name": "Cabinet_1",
                    "distance": 1.1,
                    "visible": True,
                    "pickupable": False,
                    "receptacle": True,
                    "openable": True,
                    "isOpen": False,
                },
            ],
        }

    def test_context_prioritizes_visible_near_objects(self):
        context = self.resolver.build_context(self.metadata)
        self.assertEqual(context["objects"][0]["objectId"], "Mug|near")
        self.assertEqual(context["inventoryObjects"], [])

    def test_pickup_binds_nearest_visible_matching_object(self):
        binding = self.resolver.resolve(
            action="PickupObject",
            args={"objectType": "Mug"},
            instruction="Pick up the mug",
            metadata=self.metadata,
        )
        self.assertTrue(binding.valid, binding.errors)
        self.assertEqual(binding.args, {"objectId": "Mug|near"})

    def test_open_rejects_invented_object_id(self):
        binding = self.resolver.resolve(
            action="OpenObject",
            args={"objectId": "Cabinet|missing"},
            instruction="Open the cabinet",
            metadata=self.metadata,
        )
        self.assertFalse(binding.valid)
        self.assertIn("not present", binding.errors[0])

    def test_open_binds_openable_closed_object(self):
        binding = self.resolver.resolve(
            action="OpenObject",
            args={"target": "Cabinet"},
            instruction="Open the cabinet",
            metadata=self.metadata,
        )
        self.assertTrue(binding.valid, binding.errors)
        self.assertEqual(binding.args, {"objectId": "Cabinet|1"})

    def test_put_requires_inventory_object(self):
        binding = self.resolver.resolve(
            action="PutObject",
            args={"receptacleType": "Cabinet"},
            instruction="Put it in the cabinet",
            metadata=self.metadata,
        )
        self.assertFalse(binding.valid)
        self.assertIn("inventory", binding.errors[0])

    def test_hidden_target_is_not_interacted_with_without_force_action(self):
        binding = self.resolver.resolve(
            action="PickupObject",
            args={"objectId": "Mug|far"},
            instruction="Pick up the mug",
            metadata=self.metadata,
        )
        self.assertFalse(binding.valid)
        self.assertIn("not visible", binding.errors[0])

    def test_planner_prompt_contains_environment_object_ids_and_rules(self):
        prompt = ModelAdapter()._build_planner_prompt(
            {
                "instruction": "Pick up the mug",
                "allowed_actions": ["MoveAhead", "PickupObject"],
                "action_specs": [],
                "environment_context": {
                    "objects": [
                        {
                            "objectId": "Mug|near",
                            "objectType": "Mug",
                            "visible": True,
                            "pickupable": True,
                        }
                    ]
                },
            }
        )
        self.assertIn("Mug|near", prompt)
        self.assertIn("Never invent an objectId", prompt)


if __name__ == "__main__":
    unittest.main()
