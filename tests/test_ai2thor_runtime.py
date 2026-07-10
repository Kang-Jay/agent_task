from __future__ import annotations

import unittest

from src.simulation.ai2thor_runtime import create_controller_safely


class _FailingController:
    last_instance = None

    def __init__(self, **kwargs):
        del kwargs
        type(self).last_instance = self
        self.stopped = False
        raise RuntimeError("initialize failed")

    def stop(self):
        self.stopped = True


class AI2ThorRuntimeTests(unittest.TestCase):
    def test_failed_initialization_stops_partial_controller(self):
        with self.assertRaisesRegex(RuntimeError, "initialize failed"):
            create_controller_safely(_FailingController, scene="FloorPlan1")
        self.assertIsNotNone(_FailingController.last_instance)
        self.assertTrue(_FailingController.last_instance.stopped)


if __name__ == "__main__":
    unittest.main()
