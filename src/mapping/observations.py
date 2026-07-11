from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole camera intrinsics for planner-safe RGB-D projection."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    depth_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("camera dimensions must be positive")
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError("focal lengths must be positive")
        if self.depth_scale <= 0:
            raise ValueError("depth_scale must be positive")


@dataclass(frozen=True)
class CameraPose:
    """World-space camera pose using AI2-THOR-compatible yaw convention.

    yaw_degrees=0 faces +Z, yaw_degrees=90 faces +X.
    """

    x: float
    y: float
    z: float
    yaw_degrees: float
    pitch_degrees: float = 0.0
    roll_degrees: float = 0.0

    @property
    def yaw_radians(self) -> float:
        return math.radians(self.yaw_degrees)


@dataclass(frozen=True)
class RGBDObservation:
    """Planner-safe RGB-D observation.

    This packet intentionally excludes simulator object metadata, instance masks,
    target coordinates, interactable poses, and reachable-position oracle data.
    """

    depth_meters: Sequence[Sequence[float]]
    intrinsics: CameraIntrinsics
    camera_pose: CameraPose
    step_id: int = 0
    rgb_image: object | None = None
    last_action: str | None = None
    last_action_success: bool | None = None


def intrinsics_from_vertical_fov(
    *,
    width: int,
    height: int,
    vertical_fov_degrees: float,
    depth_scale: float = 1.0,
) -> CameraIntrinsics:
    """Create pinhole intrinsics from image size and vertical field of view."""

    if vertical_fov_degrees <= 0 or vertical_fov_degrees >= 180:
        raise ValueError("vertical_fov_degrees must be between 0 and 180")
    fy = (height / 2.0) / math.tan(math.radians(vertical_fov_degrees) / 2.0)
    fx = fy
    return CameraIntrinsics(
        width=width,
        height=height,
        fx=fx,
        fy=fy,
        cx=(width - 1) / 2.0,
        cy=(height - 1) / 2.0,
        depth_scale=depth_scale,
    )
