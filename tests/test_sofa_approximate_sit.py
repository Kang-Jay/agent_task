from __future__ import annotations

import unittest

from tools.validate_ai2thor_sofa_approximation import _select_sofa


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


if __name__ == "__main__":
    unittest.main()
