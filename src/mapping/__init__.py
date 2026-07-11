"""Planner-safe spatial mapping primitives for non-oracle navigation."""

from src.mapping.depth_projector import project_depth_to_grid
from src.mapping.frontier import Frontier, extract_frontiers
from src.mapping.observations import (
    CameraIntrinsics,
    CameraPose,
    RGBDObservation,
    intrinsics_from_vertical_fov,
)
from src.mapping.occupancy_grid import (
    GridCellState,
    GridSpec,
    MapUpdateResult,
    OccupancyGrid,
)

__all__ = [
    "CameraIntrinsics",
    "CameraPose",
    "Frontier",
    "GridCellState",
    "GridSpec",
    "MapUpdateResult",
    "OccupancyGrid",
    "RGBDObservation",
    "extract_frontiers",
    "intrinsics_from_vertical_fov",
    "project_depth_to_grid",
]
