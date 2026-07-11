from __future__ import annotations

import unittest

from tools.validate_final_demo_tasks import (
    RIGHT_DOOR_ID,
    door_crossing_evidence,
    object_position_from_id,
    select_object,
)


class FinalDemoValidationTests(unittest.TestCase):
    def test_object_position_is_parsed_from_ai2thor_id(self) -> None:
        self.assertEqual(
            object_position_from_id("Box|-00.22|+00.08|-02.04"),
            {"x": -0.22, "y": 0.08, "z": -2.04},
        )

    def test_door_crossing_requires_side_change_across_threshold(self) -> None:
        crossed = door_crossing_evidence(
            door_object_id=RIGHT_DOOR_ID,
            start_metadata={"agent": {"position": {"x": -0.75, "y": 0.9, "z": 1.25}}},
            final_metadata={"agent": {"position": {"x": -0.75, "y": 0.9, "z": 1.80}}},
        )
        self.assertTrue(crossed["crossed_threshold"])
        self.assertEqual(crossed["start_side"], -1)
        self.assertEqual(crossed["final_side"], 1)

        not_crossed = door_crossing_evidence(
            door_object_id=RIGHT_DOOR_ID,
            start_metadata={"agent": {"position": {"x": -0.75, "y": 0.9, "z": 1.25}}},
            final_metadata={"agent": {"position": {"x": -0.75, "y": 0.9, "z": 1.45}}},
        )
        self.assertFalse(not_crossed["crossed_threshold"])

    def test_select_object_prefers_known_ids_and_predicate(self) -> None:
        metadata = {
            "objects": [
                {
                    "objectId": "Box|bad",
                    "objectType": "Box",
                    "receptacle": False,
                },
                {
                    "objectId": "Box|-00.22|+00.08|-02.04",
                    "objectType": "Box",
                    "receptacle": True,
                },
            ]
        }
        selected = select_object(
            metadata,
            preferred_ids=("Box|-00.22|+00.08|-02.04",),
            object_type="Box",
            predicate=lambda item: bool(item.get("receptacle")),
        )
        self.assertEqual(selected["objectId"], "Box|-00.22|+00.08|-02.04")


if __name__ == "__main__":
    unittest.main()
