from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Iterable


class GridCellState(IntEnum):
    UNKNOWN = -1
    FREE = 0
    OCCUPIED = 1


GridCell = tuple[int, int]


@dataclass(frozen=True)
class GridSpec:
    resolution_m: float
    width: int
    height: int
    origin_x: float
    origin_z: float

    def __post_init__(self) -> None:
        if self.resolution_m <= 0:
            raise ValueError("resolution_m must be positive")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("grid dimensions must be positive")


@dataclass(frozen=True)
class MapUpdateResult:
    free_cells: int = 0
    occupied_cells: int = 0
    visited_cells: int = 0
    blocked_cells: int = 0
    ignored_depth_pixels: int = 0
    projected_depth_pixels: int = 0

    def merged(self, other: "MapUpdateResult") -> "MapUpdateResult":
        return MapUpdateResult(
            free_cells=self.free_cells + other.free_cells,
            occupied_cells=self.occupied_cells + other.occupied_cells,
            visited_cells=self.visited_cells + other.visited_cells,
            blocked_cells=self.blocked_cells + other.blocked_cells,
            ignored_depth_pixels=self.ignored_depth_pixels
            + other.ignored_depth_pixels,
            projected_depth_pixels=self.projected_depth_pixels
            + other.projected_depth_pixels,
        )


@dataclass
class OccupancyGrid:
    spec: GridSpec
    cells: list[list[GridCellState]] = field(init=False)
    visited: list[list[bool]] = field(init=False)
    blocked: list[list[bool]] = field(init=False)

    def __post_init__(self) -> None:
        self.cells = [
            [GridCellState.UNKNOWN for _ in range(self.spec.width)]
            for _ in range(self.spec.height)
        ]
        self.visited = [
            [False for _ in range(self.spec.width)]
            for _ in range(self.spec.height)
        ]
        self.blocked = [
            [False for _ in range(self.spec.width)]
            for _ in range(self.spec.height)
        ]

    def in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.spec.height and 0 <= col < self.spec.width

    def world_to_grid(self, x: float, z: float) -> GridCell:
        col = int((x - self.spec.origin_x) // self.spec.resolution_m)
        row = int((z - self.spec.origin_z) // self.spec.resolution_m)
        return row, col

    def grid_to_world(self, row: int, col: int) -> tuple[float, float]:
        if not self.in_bounds(row, col):
            raise IndexError(f"grid cell out of bounds: {(row, col)}")
        return (
            self.spec.origin_x + (col + 0.5) * self.spec.resolution_m,
            self.spec.origin_z + (row + 0.5) * self.spec.resolution_m,
        )

    def get(self, row: int, col: int) -> GridCellState:
        if not self.in_bounds(row, col):
            raise IndexError(f"grid cell out of bounds: {(row, col)}")
        return self.cells[row][col]

    def mark_free(self, row: int, col: int) -> bool:
        if not self.in_bounds(row, col) or self.blocked[row][col]:
            return False
        if self.cells[row][col] == GridCellState.UNKNOWN:
            self.cells[row][col] = GridCellState.FREE
            return True
        return False

    def mark_occupied(self, row: int, col: int) -> bool:
        if not self.in_bounds(row, col):
            return False
        changed = self.cells[row][col] != GridCellState.OCCUPIED
        self.cells[row][col] = GridCellState.OCCUPIED
        return changed

    def mark_visited(self, row: int, col: int) -> bool:
        if not self.in_bounds(row, col):
            return False
        changed = not self.visited[row][col]
        self.visited[row][col] = True
        self.mark_free(row, col)
        return changed

    def mark_blocked(self, row: int, col: int) -> bool:
        if not self.in_bounds(row, col):
            return False
        changed = not self.blocked[row][col]
        self.blocked[row][col] = True
        self.cells[row][col] = GridCellState.OCCUPIED
        return changed

    def is_traversable(self, row: int, col: int, *, allow_unknown: bool = False) -> bool:
        if not self.in_bounds(row, col) or self.blocked[row][col]:
            return False
        state = self.cells[row][col]
        if state == GridCellState.OCCUPIED:
            return False
        return allow_unknown or state == GridCellState.FREE

    def neighbors4(self, row: int, col: int) -> list[GridCell]:
        candidates = ((row + 1, col), (row, col + 1), (row - 1, col), (row, col - 1))
        return [(r, c) for r, c in candidates if self.in_bounds(r, c)]

    def apply_updates(
        self,
        *,
        free_cells: Iterable[GridCell] = (),
        occupied_cells: Iterable[GridCell] = (),
        visited_cells: Iterable[GridCell] = (),
        blocked_cells: Iterable[GridCell] = (),
    ) -> MapUpdateResult:
        free_count = 0
        occupied_count = 0
        visited_count = 0
        blocked_count = 0
        for row, col in free_cells:
            free_count += int(self.mark_free(row, col))
        for row, col in occupied_cells:
            occupied_count += int(self.mark_occupied(row, col))
        for row, col in visited_cells:
            visited_count += int(self.mark_visited(row, col))
        for row, col in blocked_cells:
            blocked_count += int(self.mark_blocked(row, col))
        return MapUpdateResult(
            free_cells=free_count,
            occupied_cells=occupied_count,
            visited_cells=visited_count,
            blocked_cells=blocked_count,
        )

    def summary(self) -> dict[str, int | float]:
        counts = {state: 0 for state in GridCellState}
        for row in self.cells:
            for cell in row:
                counts[cell] += 1
        return {
            "resolution_m": self.spec.resolution_m,
            "width": self.spec.width,
            "height": self.spec.height,
            "unknown_cells": counts[GridCellState.UNKNOWN],
            "free_cells": counts[GridCellState.FREE],
            "occupied_cells": counts[GridCellState.OCCUPIED],
            "visited_cells": sum(sum(1 for value in row if value) for row in self.visited),
            "blocked_cells": sum(sum(1 for value in row if value) for row in self.blocked),
        }
