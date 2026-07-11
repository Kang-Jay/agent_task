from __future__ import annotations

import math
from collections.abc import Sequence

from src.mapping.observations import RGBDObservation
from src.mapping.occupancy_grid import GridCell, MapUpdateResult, OccupancyGrid


def project_depth_to_grid(
    grid: OccupancyGrid,
    observation: RGBDObservation,
    *,
    max_depth_m: float = 5.0,
    sample_stride: int = 4,
    obstacle_depth_margin_m: float = 0.05,
) -> MapUpdateResult:
    """Project a planner-safe depth image into an occupancy grid.

    The projection uses only depth, camera intrinsics, and camera pose. It never
    reads simulator object metadata, masks, target positions, or reachable cells.
    """

    if max_depth_m <= 0:
        raise ValueError("max_depth_m must be positive")
    if sample_stride <= 0:
        raise ValueError("sample_stride must be positive")

    depth = observation.depth_meters
    height = len(depth)
    width = len(depth[0]) if height else 0
    if height == 0 or width == 0:
        return MapUpdateResult()
    if height != observation.intrinsics.height or width != observation.intrinsics.width:
        raise ValueError("depth dimensions must match camera intrinsics")

    start = grid.world_to_grid(observation.camera_pose.x, observation.camera_pose.z)
    visited = [start]
    free_cells: set[GridCell] = set()
    occupied_cells: set[GridCell] = set()
    ignored = 0
    projected = 0

    for v in range(0, height, sample_stride):
        row_values: Sequence[float] = depth[v]
        for u in range(0, width, sample_stride):
            raw_depth = float(row_values[u])
            if not math.isfinite(raw_depth) or raw_depth <= 0:
                ignored += 1
                continue
            depth_m = raw_depth * observation.intrinsics.depth_scale
            if depth_m <= 0:
                ignored += 1
                continue
            clipped_depth = min(depth_m, max_depth_m)
            endpoint = _project_pixel_to_world(
                u=u,
                v=v,
                depth_m=clipped_depth,
                observation=observation,
            )
            end_cell = grid.world_to_grid(*endpoint)
            ray_cells = _bresenham_cells(start, end_cell)
            if ray_cells:
                terminal = ray_cells[-1]
                free_cells.update(ray_cells[:-1])
                if depth_m <= max_depth_m - obstacle_depth_margin_m:
                    occupied_cells.add(terminal)
                else:
                    free_cells.add(terminal)
            projected += 1

    update = grid.apply_updates(
        free_cells=free_cells,
        occupied_cells=occupied_cells,
        visited_cells=visited,
    )
    return MapUpdateResult(
        free_cells=update.free_cells,
        occupied_cells=update.occupied_cells,
        visited_cells=update.visited_cells,
        blocked_cells=update.blocked_cells,
        ignored_depth_pixels=ignored,
        projected_depth_pixels=projected,
    )


def _project_pixel_to_world(
    *,
    u: int,
    v: int,
    depth_m: float,
    observation: RGBDObservation,
) -> tuple[float, float]:
    intr = observation.intrinsics
    pose = observation.camera_pose
    x_cam = (u - intr.cx) * depth_m / intr.fx
    z_cam = depth_m * math.cos(math.radians(pose.pitch_degrees))

    yaw = pose.yaw_radians
    right_x = math.cos(yaw)
    right_z = -math.sin(yaw)
    forward_x = math.sin(yaw)
    forward_z = math.cos(yaw)
    world_x = pose.x + right_x * x_cam + forward_x * z_cam
    world_z = pose.z + right_z * x_cam + forward_z * z_cam
    return world_x, world_z


def _bresenham_cells(start: GridCell, end: GridCell) -> list[GridCell]:
    r0, c0 = start
    r1, c1 = end
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    step_r = 1 if r0 < r1 else -1
    step_c = 1 if c0 < c1 else -1
    row, col = r0, c0
    cells: list[GridCell] = [(row, col)]
    if dc > dr:
        err = dc / 2
        while col != c1:
            col += step_c
            err -= dr
            if err < 0:
                row += step_r
                err += dc
            cells.append((row, col))
    else:
        err = dr / 2
        while row != r1:
            row += step_r
            err -= dc
            if err < 0:
                col += step_c
                err += dr
            cells.append((row, col))
    return cells
