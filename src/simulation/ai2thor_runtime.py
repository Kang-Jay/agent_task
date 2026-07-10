from __future__ import annotations

import math
from typing import Any


GRID_ALIGNED_ROTATIONS = (0.0, 90.0, 180.0, 270.0)


def is_grid_aligned_rotation(rotate_step_degrees: float) -> bool:
    """Return whether AI2-THOR can combine this rotation with snapToGrid."""
    normalized = float(rotate_step_degrees) % 360.0
    return any(
        math.isclose(normalized, allowed, abs_tol=1e-6)
        for allowed in GRID_ALIGNED_ROTATIONS
    )


def should_snap_to_grid(*, mode: str, rotate_step_degrees: float) -> bool:
    """Enable grid snapping only for compatible default-agent rotations."""
    return mode.lower() == "default" and is_grid_aligned_rotation(
        rotate_step_degrees
    )


def create_controller_safely(
    controller_type: type[Any],
    **kwargs: Any,
) -> Any:
    """Retain the partial Controller so failed initialization can be cleaned up."""
    controller = controller_type.__new__(controller_type)
    try:
        controller_type.__init__(controller, **kwargs)
    except BaseException:
        stop = getattr(controller, "stop", None)
        if callable(stop):
            try:
                stop()
            except BaseException:
                pass
        raise
    return controller
