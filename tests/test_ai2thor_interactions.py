from __future__ import annotations

import copy
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
                {
                    "objectId": "Box|1",
                    "objectType": "Box",
                    "name": "Box_1",
                    "distance": 1.2,
                    "visible": True,
                    "pickupable": False,
                    "receptacle": True,
                    "openable": False,
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

    def test_put_object_id_is_the_receptacle_not_the_held_object(self):
        metadata = copy.deepcopy(self.metadata)
        metadata["inventoryObjects"] = [
            {"objectId": "Mug|near", "objectType": "Mug"}
        ]
        metadata["objects"][2]["isOpen"] = True

        binding = self.resolver.resolve(
            action="PutObject",
            args={
                "object": "Mug",
                "receptacleType": "Cabinet",
            },
            instruction="Put the held mug in the cabinet",
            metadata=metadata,
        )

        self.assertTrue(binding.valid, binding.errors)
        self.assertEqual(binding.args, {"objectId": "Cabinet|1"})
        self.assertEqual(binding.target_object["objectType"], "Cabinet")
        self.assertEqual(
            metadata["inventoryObjects"],
            [{"objectId": "Mug|near", "objectType": "Mug"}],
        )

    def test_put_rejects_when_requested_object_is_not_held(self):
        metadata = copy.deepcopy(self.metadata)
        metadata["inventoryObjects"] = [
            {"objectId": "Plate|1", "objectType": "Plate"}
        ]
        metadata["objects"][2]["isOpen"] = True

        binding = self.resolver.resolve(
            action="PutObject",
            args={
                "object": "Mug",
                "receptacleType": "Cabinet",
            },
            instruction="Put the mug in the cabinet",
            metadata=metadata,
        )

        self.assertFalse(binding.valid)
        self.assertIn("does not match inventory", binding.errors[0])
        self.assertIn("Plate", binding.errors[0])

    def test_put_accepts_native_object_id_for_visible_open_receptacle(self):
        metadata = copy.deepcopy(self.metadata)
        metadata["inventoryObjects"] = [
            {"objectId": "Mug|near", "objectType": "Mug"}
        ]
        metadata["objects"][2]["isOpen"] = True

        binding = self.resolver.resolve(
            action="PutObject",
            args={"objectId": "Cabinet|1"},
            instruction="Put it in the cabinet",
            metadata=metadata,
        )

        self.assertTrue(binding.valid, binding.errors)
        self.assertEqual(binding.args, {"objectId": "Cabinet|1"})

    def test_put_accepts_receptacle_object_id_alias(self):
        metadata = copy.deepcopy(self.metadata)
        metadata["inventoryObjects"] = [
            {"objectId": "Mug|near", "objectType": "Mug"}
        ]

        binding = self.resolver.resolve(
            action="PutObject",
            args={"receptacleObjectId": "Box|1"},
            instruction="Put it in the box",
            metadata=metadata,
        )

        self.assertTrue(binding.valid, binding.errors)
        self.assertEqual(binding.args, {"objectId": "Box|1"})

    def test_put_allows_redundant_receptacle_object_type_and_target(self):
        metadata = copy.deepcopy(self.metadata)
        metadata["inventoryObjects"] = [
            {"objectId": "Mug|near", "objectType": "Mug"}
        ]
        metadata["objects"][2]["isOpen"] = True

        binding = self.resolver.resolve(
            action="PutObject",
            args={
                "objectType": "Cabinet",
                "target": "Cabinet",
            },
            instruction="Put it in the cabinet",
            metadata=metadata,
        )

        self.assertTrue(binding.valid, binding.errors)
        self.assertEqual(binding.args, {"objectId": "Cabinet|1"})

    def test_put_rejects_conflicting_receptacle_ids(self):
        metadata = copy.deepcopy(self.metadata)
        metadata["inventoryObjects"] = [
            {"objectId": "Mug|near", "objectType": "Mug"}
        ]

        binding = self.resolver.resolve(
            action="PutObject",
            args={
                "objectId": "Cabinet|1",
                "receptacleObjectId": "Box|1",
            },
            instruction="Put it away",
            metadata=metadata,
        )

        self.assertFalse(binding.valid)
        self.assertIn("conflicting receptacle identifiers", binding.errors[0])

    def test_put_rejects_closed_openable_receptacle(self):
        metadata = copy.deepcopy(self.metadata)
        metadata["inventoryObjects"] = [
            {"objectId": "Mug|near", "objectType": "Mug"}
        ]

        binding = self.resolver.resolve(
            action="PutObject",
            args={"receptacleType": "Cabinet"},
            instruction="Put it in the cabinet",
            metadata=metadata,
        )

        self.assertFalse(binding.valid)
        self.assertIn("open receptacle", binding.errors[0])

    def test_put_rejects_ambiguous_inventory(self):
        metadata = copy.deepcopy(self.metadata)
        metadata["inventoryObjects"] = [
            {"objectId": "Mug|near", "objectType": "Mug"},
            {"objectId": "Plate|1", "objectType": "Plate"},
        ]

        binding = self.resolver.resolve(
            action="PutObject",
            args={"receptacleType": "Box"},
            instruction="Put it in the box",
            metadata=metadata,
        )

        self.assertFalse(binding.valid)
        self.assertIn("exactly one", binding.errors[0])

    def test_pickup_rejects_nonempty_inventory(self):
        metadata = copy.deepcopy(self.metadata)
        metadata["inventoryObjects"] = [
            {"objectId": "Plate|1", "objectType": "Plate"}
        ]

        binding = self.resolver.resolve(
            action="PickupObject",
            args={"objectType": "Mug"},
            instruction="Pick up the mug",
            metadata=metadata,
        )

        self.assertFalse(binding.valid)
        self.assertIn("empty inventory", binding.errors[0])

    def test_hidden_target_is_not_interacted_with_without_force_action(self):
        binding = self.resolver.resolve(
            action="PickupObject",
            args={"objectId": "Mug|far"},
            instruction="Pick up the mug",
            metadata=self.metadata,
        )
        self.assertFalse(binding.valid)
        self.assertIn("not visible", binding.errors[0])

    def test_visible_target_requires_valid_distance_metadata(self):
        metadata = copy.deepcopy(self.metadata)
        metadata["objects"][2]["distance"] = float("nan")

        binding = self.resolver.resolve(
            action="OpenObject",
            args={"objectId": "Cabinet|1"},
            instruction="Open the cabinet",
            metadata=metadata,
        )

        self.assertFalse(binding.valid)
        self.assertIn("finite non-negative distance", binding.errors[0])

    def test_force_action_explicitly_bypasses_visibility_and_distance_checks(self):
        metadata = copy.deepcopy(self.metadata)
        metadata["objects"][1].pop("distance")

        binding = self.resolver.resolve(
            action="PickupObject",
            args={"objectId": "Mug|far", "forceAction": True},
            instruction="Force pickup the mug",
            metadata=metadata,
        )

        self.assertTrue(binding.valid, binding.errors)
        self.assertEqual(
            binding.args,
            {"objectId": "Mug|far", "forceAction": True},
        )

    def test_explicit_selector_does_not_fall_back_to_instruction_object(self):
        binding = self.resolver.resolve(
            action="OpenObject",
            args={"target": "Drawer"},
            instruction="Open the cabinet",
            metadata=self.metadata,
        )

        self.assertFalse(binding.valid)
        self.assertIn("Drawer", binding.errors[0])

    def test_open_pickup_put_sequence_preserves_role_separation(self):
        metadata = copy.deepcopy(self.metadata)

        opened = self.resolver.resolve(
            action="OpenObject",
            args={"objectType": "Cabinet"},
            instruction="Open the cabinet, pick up the mug, and put it inside",
            metadata=metadata,
        )
        self.assertTrue(opened.valid, opened.errors)
        self.assertEqual(opened.args, {"objectId": "Cabinet|1"})
        metadata["objects"][2]["isOpen"] = True

        picked = self.resolver.resolve(
            action="PickupObject",
            args={"target": "Mug"},
            instruction="Pick up the mug",
            metadata=metadata,
        )
        self.assertTrue(picked.valid, picked.errors)
        self.assertEqual(picked.args, {"objectId": "Mug|near"})
        metadata["inventoryObjects"] = [
            {"objectId": "Mug|near", "objectType": "Mug"}
        ]

        placed = self.resolver.resolve(
            action="PutObject",
            args={
                "object": "Mug",
                "target": "Cabinet",
            },
            instruction="Put the held mug in the cabinet",
            metadata=metadata,
        )
        self.assertTrue(placed.valid, placed.errors)
        self.assertEqual(placed.args, {"objectId": "Cabinet|1"})
        self.assertEqual(placed.target_object["objectId"], "Cabinet|1")

    def test_vase_into_box_binds_held_object_and_receptacle_roles(self):
        metadata = copy.deepcopy(self.metadata)
        metadata["objects"].extend(
            [
                {
                    "objectId": "Vase|1",
                    "objectType": "Vase",
                    "name": "Vase_1",
                    "distance": 0.5,
                    "visible": True,
                    "pickupable": True,
                    "receptacle": False,
                    "openable": False,
                    "isOpen": False,
                    "isPickedUp": False,
                    "parentReceptacles": [],
                    "receptacleObjectIds": [],
                },
                {
                    "objectId": "CardboardBox|1",
                    "objectType": "CardboardBox",
                    "name": "CardboardBox_1",
                    "distance": 0.9,
                    "visible": True,
                    "pickupable": False,
                    "receptacle": True,
                    "openable": False,
                    "isOpen": False,
                    "isPickedUp": False,
                    "parentReceptacles": [],
                    "receptacleObjectIds": [],
                },
            ]
        )

        picked = self.resolver.resolve(
            action="PickupObject",
            args={"objectType": "Vase"},
            instruction="把花瓶放到纸箱里",
            metadata=metadata,
        )
        self.assertTrue(picked.valid, picked.errors)
        self.assertEqual(picked.args, {"objectId": "Vase|1"})
        metadata["inventoryObjects"] = [
            {"objectId": "Vase|1", "objectType": "Vase"}
        ]

        placed = self.resolver.resolve(
            action="PutObject",
            args={
                "object": "Vase",
                "receptacleType": "CardboardBox",
            },
            instruction="把花瓶放到纸箱里",
            metadata=metadata,
        )
        self.assertTrue(placed.valid, placed.errors)
        self.assertEqual(placed.args, {"objectId": "CardboardBox|1"})
        self.assertEqual(placed.target_object["objectType"], "CardboardBox")

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
