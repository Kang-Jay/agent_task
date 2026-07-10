"""Integration tests for object closeup rendering in AI2-THOR adapter.

According to ChangeRecord/1-9/10016_object_click_closeup_render.md (Stage D).

Tests that the adapter correctly prepares a closeup reference image when a
user clicks an object on the first step, and falls back gracefully when
closeup rendering fails.
"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from src.simulation.ai2thor_adapter import AI2ThorVisualSearchDemo


class FakeMetadata:
    def __init__(self, objects):
        self.objects = objects


class FakeEvent:
    def __init__(self, frame, instance_masks, objects):
        self.frame = frame
        self.instance_masks = instance_masks
        self.metadata = {"objects": objects, "lastActionSuccess": True}


def _mask(height, width, region):
    m = np.zeros((height, width), dtype=bool)
    y0, y1, x0, x1 = region
    m[y0:y1, x0:x1] = True
    return m


class CloseupIntegrationTests(unittest.TestCase):
    def test_prepare_click_target_with_point_renders_closeup(self) -> None:
        """When clicked_point hits an object, prepare_click_target renders closeup."""
        demo = AI2ThorVisualSearchDemo(scene="FloorPlan1")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        masks = {"Television|1": _mask(480, 640, (100, 300, 200, 400))}
        objects = [
            {
                "objectId": "Television|1",
                "objectType": "Television",
                "position": {"x": 1.0, "y": 0.5, "z": 2.0},
                "toggleable": True,
                "visible": True,
            }
        ]
        event = FakeEvent(frame, masks, objects)
        mock_controller = MagicMock()

        # Mock render_closeup to return a successful image
        closeup_frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        mock_render_event = FakeEvent(closeup_frame, {}, [])
        mock_render_event.third_party_camera_frames = [closeup_frame]

        with patch.object(
            demo.action_executor, "execute", return_value=MagicMock(
                success=True, event=mock_render_event
            )
        ):
            target_crop_url, object_id, binding = demo._prepare_click_target(
                mock_controller, event, clicked_point=[300, 200], clicked_object_id=None
            )

        self.assertIsNotNone(target_crop_url)
        self.assertTrue(target_crop_url.startswith("data:image/"))
        self.assertEqual(object_id, "Television|1")
        self.assertIsNotNone(binding)
        self.assertEqual(binding["object_type"], "Television")
        self.assertEqual(binding["closeup_source"], "third_party_camera")

    def test_prepare_click_target_falls_back_on_render_failure(self) -> None:
        """When closeup render fails, prepare_click_target returns None target_crop."""
        demo = AI2ThorVisualSearchDemo(scene="FloorPlan1")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        masks = {"Box|1": _mask(480, 640, (50, 150, 50, 150))}
        objects = [
            {
                "objectId": "Box|1",
                "objectType": "Box",
                "position": {"x": 0.5, "y": 0.0, "z": 1.0},
                "pickupable": True,
            }
        ]
        event = FakeEvent(frame, masks, objects)
        mock_controller = MagicMock()

        # Mock render_closeup to fail (no frames)
        mock_render_event = FakeEvent(frame, {}, [])
        mock_render_event.third_party_camera_frames = []

        with patch.object(
            demo.action_executor, "execute", return_value=MagicMock(
                success=False, event=mock_render_event
            )
        ):
            target_crop_url, object_id, binding = demo._prepare_click_target(
                mock_controller, event, clicked_point=[100, 100], clicked_object_id=None
            )

        self.assertIsNone(target_crop_url)
        self.assertEqual(object_id, "Box|1")
        self.assertIsNotNone(binding)
        self.assertEqual(binding["closeup_source"], "pov_crop_fallback")

    def test_prepare_click_target_with_explicit_object_id(self) -> None:
        """When clicked_object_id is provided directly, it takes precedence."""
        demo = AI2ThorVisualSearchDemo(scene="FloorPlan1")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        objects = [
            {
                "objectId": "Sofa|9",
                "objectType": "Sofa",
                "position": {"x": -1.0, "y": 0.0, "z": 3.0},
                "receptacle": True,
            }
        ]
        event = FakeEvent(frame, {}, objects)
        mock_controller = MagicMock()

        closeup_frame = np.ones((480, 640, 3), dtype=np.uint8) * 64
        mock_render_event = FakeEvent(frame, {}, [])
        mock_render_event.third_party_camera_frames = [closeup_frame]

        with patch.object(
            demo.action_executor, "execute", return_value=MagicMock(
                success=True, event=mock_render_event
            )
        ):
            target_crop_url, object_id, binding = demo._prepare_click_target(
                mock_controller,
                event,
                clicked_point=None,
                clicked_object_id="Sofa|9",
            )

        self.assertIsNotNone(target_crop_url)
        self.assertEqual(object_id, "Sofa|9")
        self.assertEqual(binding["object_type"], "Sofa")

    def test_prepare_click_target_miss_returns_none(self) -> None:
        """When click misses all objects, prepare_click_target returns all None."""
        demo = AI2ThorVisualSearchDemo(scene="FloorPlan1")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        masks = {"Sofa|1": _mask(480, 640, (0, 50, 0, 50))}
        objects = [{"objectId": "Sofa|1", "objectType": "Sofa"}]
        event = FakeEvent(frame, masks, objects)
        mock_controller = MagicMock()

        target_crop_url, object_id, binding = demo._prepare_click_target(
            mock_controller, event, clicked_point=[500, 400], clicked_object_id=None
        )

        self.assertIsNone(target_crop_url)
        self.assertIsNone(object_id)
        self.assertIsNone(binding)


if __name__ == "__main__":
    unittest.main()
