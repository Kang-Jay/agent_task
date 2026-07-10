"""Object-level click resolution and close-up rendering (ChangeRecord 10016).

Turns a user click (pixel coordinate on the POV frame or a resolved objectId)
into: the clicked object's objectId, its affordance context, and a close-up
reference image rendered near the object via an AI2-THOR third-party camera.

Design constraints:
- This module MUST NOT import ai2thor_adapter (that module imports this one),
  so the caller passes in the structural-object filter set and the config.
- All camera geometry parameters come from configs/agent_config.json ["closeup"];
  nothing is hard-coded here.
- Every path has a graceful fallback so the demo never crashes when the
  simulator lacks segmentation or the camera cannot be placed.
"""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np
from PIL import Image

from src.simulation.ai2thor_interactions import CONTEXT_OBJECT_KEYS
from src.types.schema import ClickedObjectBinding


def _normalize_type(object_id: str, obj: dict[str, Any] | None) -> str:
    if obj and obj.get("objectType"):
        return str(obj["objectType"]).lower()
    # objectId looks like "Television|+00.00|+00.50|+01.00"
    return str(object_id).split("|", 1)[0].lower()


def object_id_at_pixel(
    event: Any,
    x: int,
    y: int,
    *,
    structural_types: Iterable[str] = (),
    min_mask_pixels: int = 50,
) -> str | None:
    """Return the objectId whose instance mask covers pixel (x, y).

    Uses event.instance_masks (requires renderInstanceSegmentation=True).
    Filters out structural objects (wall/floor/...) and masks smaller than
    min_mask_pixels. When several objects overlap the pixel, the one with the
    largest mask is returned. Returns None when nothing valid is under (x, y).
    """
    masks = getattr(event, "instance_masks", None) or {}
    if not masks:
        return None
    metadata_by_id = {
        str(item.get("objectId") or item.get("name") or ""): item
        for item in getattr(event, "metadata", {}).get("objects", [])
    }
    structural = {str(term).lower() for term in structural_types}

    best_id: str | None = None
    best_area = -1
    for object_id, raw_mask in masks.items():
        mask = np.asarray(raw_mask).astype(bool)
        if mask.ndim != 2:
            continue
        height, width = mask.shape[:2]
        if not (0 <= y < height and 0 <= x < width):
            continue
        if not mask[y, x]:
            continue
        area = int(mask.sum())
        if area < int(min_mask_pixels):
            continue
        normalized = _normalize_type(str(object_id), metadata_by_id.get(str(object_id)))
        if normalized in structural:
            continue
        if area > best_area:
            best_area = area
            best_id = str(object_id)
    return best_id


def _affordances(obj: dict[str, Any]) -> dict[str, Any]:
    return {key: obj.get(key) for key in CONTEXT_OBJECT_KEYS if key in obj}


def resolve_clicked_object(
    event: Any,
    *,
    x: int | None = None,
    y: int | None = None,
    object_id: str | None = None,
    structural_types: Iterable[str] = (),
    min_mask_pixels: int = 50,
) -> ClickedObjectBinding | None:
    """Resolve a click (pixel or explicit objectId) into a ClickedObjectBinding.

    The close-up fields are left empty here; render_closeup() fills them once
    the reference image has actually been rendered.
    """
    resolved_id = object_id
    if not resolved_id:
        if x is None or y is None:
            return None
        resolved_id = object_id_at_pixel(
            event,
            int(x),
            int(y),
            structural_types=structural_types,
            min_mask_pixels=min_mask_pixels,
        )
    if not resolved_id:
        return None

    obj = next(
        (
            item
            for item in getattr(event, "metadata", {}).get("objects", [])
            if str(item.get("objectId")) == str(resolved_id)
        ),
        None,
    )
    if obj is None:
        return None

    position = obj.get("position")
    world_position = (
        {
            "x": float(position["x"]),
            "y": float(position["y"]),
            "z": float(position["z"]),
        }
        if isinstance(position, dict) and {"x", "y", "z"} <= set(position)
        else None
    )
    return ClickedObjectBinding(
        object_id=str(resolved_id),
        object_type=str(obj.get("objectType") or _normalize_type(str(resolved_id), obj)),
        affordances=_affordances(obj),
        closeup_source="",
        closeup_bbox=None,
        world_position=world_position,
    )


def render_closeup(
    action_executor: Any,
    controller: Any,
    *,
    mode: str,
    target_position: dict[str, float],
    config: Any,
) -> tuple[Image.Image | None, str, list[int] | None]:
    """Render a close-up image of the object at target_position.

    Places a perspective third-party camera above/behind the object looking
    down at it (geometry from config["closeup"]), then reads the last
    third_party_camera_frames entry. On any failure returns
    (None, "pov_crop_fallback", None) so the caller can fall back to the
    legacy point-crop path.
    """
    closeup_cfg = config.raw["closeup"]
    height_offset = float(closeup_cfg["camera_height_offset"])
    back_distance = float(closeup_cfg["camera_back_distance"])
    pitch = float(closeup_cfg["camera_pitch_degrees"])
    field_of_view = float(closeup_cfg["field_of_view"])

    camera_position = {
        "x": float(target_position["x"]),
        "y": float(target_position["y"]) + height_offset,
        "z": float(target_position["z"]) - back_distance,
    }
    camera_rotation = {"x": pitch, "y": 0.0, "z": 0.0}

    try:
        execution = action_executor.execute(
            controller,
            mode=mode,
            action="AddThirdPartyCamera",
            args={
                "position": camera_position,
                "rotation": camera_rotation,
                "fieldOfView": field_of_view,
                "orthographic": False,
                "antiAliasing": "fxaa",
                "skyboxColor": "white",
            },
            actor="system",
        )
        if not execution.success:
            return None, "pov_crop_fallback", None
        frames = getattr(execution.event, "third_party_camera_frames", None) or []
        if not frames:
            return None, "pov_crop_fallback", None
        image = Image.fromarray(np.asarray(frames[-1])).convert("RGB")
        return image, "third_party_camera", None
    except Exception:
        return None, "pov_crop_fallback", None
