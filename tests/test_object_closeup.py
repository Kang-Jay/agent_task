"""Tests for object-level click resolution and close-up rendering.

According to ChangeRecord/1-9/10016_object_click_closeup_render.md (Stage C).

All AI2-THOR interactions are mocked; no real simulator is required.
"""
from __future__ import annotations

import unittest

import numpy as np
from PIL import Image

from src.simulation.object_closeup import (
    object_id_at_pixel,
    resolve_clicked_object,
    render_closeup,
)
from src.task.config import load_config


STRUCTURAL_TYPES = {"floor", "wall", "ceiling", "window", "painting"}


class FakeEvent:
    """Minimal stand-in for an AI2-THOR event."""

    def __init__(self, instance_masks, objects, third_party_frames=None):
        self.instance_masks = instance_masks
        self.metadata = {"objects": objects}
        if third_party_frames is not None:
            self.third_party_camera_frames = third_party_frames


class FakeExecution:
    def __init__(self, success, event):
        self.success = success
        self.event = event


class FakeExecutor:
    """Records the last execute() call and returns a scripted result."""

    def __init__(self, result_event, success=True, raise_exc=None):
        self._result_event = result_event
        self._success = success
        self._raise_exc = raise_exc
        self.calls = []

    def execute(self, controller, *, mode, action, args=None, actor="agent"):
        self.calls.append({"mode": mode, "action": action, "args": args, "actor": actor})
        if self._raise_exc is not None:
            raise self._raise_exc
        return FakeExecution(self._success, self._result_event)


def _mask(height, width, region):
    """Build a boolean mask with region=(y0, y1, x0, x1) set True."""
    m = np.zeros((height, width), dtype=bool)
    y0, y1, x0, x1 = region
    m[y0:y1, x0:x1] = True
    return m


class ObjectIdAtPixelTests(unittest.TestCase):
    def test_hits_object_under_pixel(self) -> None:
        masks = {
            "Television|1": _mask(100, 100, (10, 40, 10, 40)),
            "Sofa|2": _mask(100, 100, (60, 90, 60, 90)),
        }
        objects = [
            {"objectId": "Television|1", "objectType": "Television"},
            {"objectId": "Sofa|2", "objectType": "Sofa"},
        ]
        event = FakeEvent(masks, objects)
        self.assertEqual(
            object_id_at_pixel(event, 25, 25, structural_types=STRUCTURAL_TYPES),
            "Television|1",
        )
        self.assertEqual(
            object_id_at_pixel(event, 75, 75, structural_types=STRUCTURAL_TYPES),
            "Sofa|2",
        )

    def test_returns_none_on_empty_pixel(self) -> None:
        masks = {"Television|1": _mask(100, 100, (10, 40, 10, 40))}
        objects = [{"objectId": "Television|1", "objectType": "Television"}]
        event = FakeEvent(masks, objects)
        self.assertIsNone(
            object_id_at_pixel(event, 80, 80, structural_types=STRUCTURAL_TYPES)
        )

    def test_out_of_bounds_returns_none(self) -> None:
        masks = {"Television|1": _mask(100, 100, (10, 40, 10, 40))}
        objects = [{"objectId": "Television|1", "objectType": "Television"}]
        event = FakeEvent(masks, objects)
        self.assertIsNone(
            object_id_at_pixel(event, 200, 200, structural_types=STRUCTURAL_TYPES)
        )

    def test_structural_object_filtered(self) -> None:
        masks = {"Wall|1": _mask(100, 100, (0, 100, 0, 100))}
        objects = [{"objectId": "Wall|1", "objectType": "Wall"}]
        event = FakeEvent(masks, objects)
        self.assertIsNone(
            object_id_at_pixel(event, 50, 50, structural_types=STRUCTURAL_TYPES)
        )

    def test_overlap_picks_largest_mask(self) -> None:
        # Both masks cover pixel (50,50); larger area should win.
        masks = {
            "SmallBox|1": _mask(100, 100, (45, 55, 45, 55)),   # 100 px
            "BigTable|2": _mask(100, 100, (20, 80, 20, 80)),   # 3600 px
        }
        objects = [
            {"objectId": "SmallBox|1", "objectType": "Box"},
            {"objectId": "BigTable|2", "objectType": "Table"},
        ]
        event = FakeEvent(masks, objects)
        self.assertEqual(
            object_id_at_pixel(event, 50, 50, structural_types=STRUCTURAL_TYPES),
            "BigTable|2",
        )

    def test_min_mask_pixels_filters_tiny(self) -> None:
        masks = {"Speck|1": _mask(100, 100, (50, 52, 50, 52))}  # 4 px
        objects = [{"objectId": "Speck|1", "objectType": "Vase"}]
        event = FakeEvent(masks, objects)
        self.assertIsNone(
            object_id_at_pixel(
                event, 50, 50, structural_types=STRUCTURAL_TYPES, min_mask_pixels=50
            )
        )

    def test_no_masks_returns_none(self) -> None:
        event = FakeEvent({}, [])
        self.assertIsNone(
            object_id_at_pixel(event, 10, 10, structural_types=STRUCTURAL_TYPES)
        )


class ResolveClickedObjectTests(unittest.TestCase):
    def test_resolve_by_pixel_extracts_affordances(self) -> None:
        masks = {"Box|1": _mask(100, 100, (10, 60, 10, 60))}
        objects = [
            {
                "objectId": "Box|1",
                "objectType": "Box",
                "position": {"x": 1.0, "y": 0.5, "z": 2.0},
                "pickupable": True,
                "receptacle": True,
                "visible": True,
            }
        ]
        event = FakeEvent(masks, objects)
        binding = resolve_clicked_object(
            event, x=30, y=30, structural_types=STRUCTURAL_TYPES
        )
        self.assertIsNotNone(binding)
        self.assertEqual(binding.object_id, "Box|1")
        self.assertEqual(binding.object_type, "Box")
        self.assertEqual(binding.world_position, {"x": 1.0, "y": 0.5, "z": 2.0})
        self.assertTrue(binding.affordances["pickupable"])
        self.assertTrue(binding.affordances["receptacle"])
        self.assertEqual(binding.closeup_source, "")

    def test_resolve_by_explicit_object_id(self) -> None:
        objects = [
            {
                "objectId": "Television|9",
                "objectType": "Television",
                "position": {"x": 0.0, "y": 1.2, "z": 3.0},
                "toggleable": True,
            }
        ]
        event = FakeEvent({}, objects)
        binding = resolve_clicked_object(
            event, object_id="Television|9", structural_types=STRUCTURAL_TYPES
        )
        self.assertIsNotNone(binding)
        self.assertEqual(binding.object_id, "Television|9")
        self.assertEqual(binding.object_type, "Television")
        self.assertTrue(binding.affordances["toggleable"])

    def test_resolve_unknown_object_id_returns_none(self) -> None:
        event = FakeEvent({}, [{"objectId": "Sofa|1", "objectType": "Sofa"}])
        self.assertIsNone(
            resolve_clicked_object(event, object_id="Ghost|0")
        )

    def test_resolve_pixel_miss_returns_none(self) -> None:
        masks = {"Box|1": _mask(100, 100, (10, 20, 10, 20))}
        objects = [{"objectId": "Box|1", "objectType": "Box"}]
        event = FakeEvent(masks, objects)
        self.assertIsNone(
            resolve_clicked_object(event, x=90, y=90, structural_types=STRUCTURAL_TYPES)
        )

    def test_resolve_without_coords_or_id_returns_none(self) -> None:
        event = FakeEvent({}, [])
        self.assertIsNone(resolve_clicked_object(event))


class RenderCloseupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config()

    def test_success_returns_image_and_source(self) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result_event = FakeEvent({}, [], third_party_frames=[frame])
        executor = FakeExecutor(result_event, success=True)
        image, source, bbox = render_closeup(
            executor,
            controller=object(),
            mode="default",
            target_position={"x": 1.0, "y": 0.5, "z": 2.0},
            config=self.config,
        )
        self.assertIsInstance(image, Image.Image)
        self.assertEqual(source, "third_party_camera")
        self.assertIsNone(bbox)
        # Verify camera geometry came from config, not hard-coded.
        call = executor.calls[-1]
        self.assertEqual(call["action"], "AddThirdPartyCamera")
        self.assertEqual(call["actor"], "system")
        cfg = self.config.raw["closeup"]
        self.assertAlmostEqual(
            call["args"]["position"]["y"], 0.5 + cfg["camera_height_offset"]
        )
        self.assertAlmostEqual(
            call["args"]["position"]["z"], 2.0 - cfg["camera_back_distance"]
        )
        self.assertAlmostEqual(call["args"]["fieldOfView"], cfg["field_of_view"])

    def test_execution_failure_falls_back(self) -> None:
        result_event = FakeEvent({}, [], third_party_frames=[])
        executor = FakeExecutor(result_event, success=False)
        image, source, bbox = render_closeup(
            executor,
            controller=object(),
            mode="default",
            target_position={"x": 0.0, "y": 0.0, "z": 0.0},
            config=self.config,
        )
        self.assertIsNone(image)
        self.assertEqual(source, "pov_crop_fallback")
        self.assertIsNone(bbox)

    def test_no_frames_falls_back(self) -> None:
        result_event = FakeEvent({}, [], third_party_frames=[])
        executor = FakeExecutor(result_event, success=True)
        image, source, _ = render_closeup(
            executor,
            controller=object(),
            mode="default",
            target_position={"x": 0.0, "y": 0.0, "z": 0.0},
            config=self.config,
        )
        self.assertIsNone(image)
        self.assertEqual(source, "pov_crop_fallback")

    def test_exception_falls_back(self) -> None:
        executor = FakeExecutor(None, raise_exc=ValueError("camera error"))
        image, source, _ = render_closeup(
            executor,
            controller=object(),
            mode="default",
            target_position={"x": 0.0, "y": 0.0, "z": 0.0},
            config=self.config,
        )
        self.assertIsNone(image)
        self.assertEqual(source, "pov_crop_fallback")


if __name__ == "__main__":
    unittest.main()
