from __future__ import annotations

from dataclasses import dataclass

from src.mapping.occupancy_grid import GridCell, GridCellState, OccupancyGrid


@dataclass(frozen=True)
class Frontier:
    row: int
    col: int
    unknown_neighbors: int
    free_neighbors: int
    distance_cells: int

    @property
    def cell(self) -> GridCell:
        return self.row, self.col

    def to_dict(self) -> dict[str, int]:
        return {
            "row": self.row,
            "col": self.col,
            "unknown_neighbors": self.unknown_neighbors,
            "free_neighbors": self.free_neighbors,
            "distance_cells": self.distance_cells,
        }


def extract_frontiers(
    grid: OccupancyGrid,
    *,
    start: GridCell | None = None,
    reachable_cells: set[GridCell] | None = None,
    min_unknown_neighbors: int = 1,
) -> list[Frontier]:
    """Return free cells adjacent to unknown space, filtered deterministically."""

    if min_unknown_neighbors <= 0:
        raise ValueError("min_unknown_neighbors must be positive")
    result: list[Frontier] = []
    for row in range(grid.spec.height):
        for col in range(grid.spec.width):
            if not grid.is_traversable(row, col):
                continue
            if reachable_cells is not None and (row, col) not in reachable_cells:
                continue
            unknown_neighbors = 0
            free_neighbors = 0
            for nr, nc in grid.neighbors4(row, col):
                state = grid.get(nr, nc)
                if state == GridCellState.UNKNOWN:
                    unknown_neighbors += 1
                elif state == GridCellState.FREE:
                    free_neighbors += 1
            if unknown_neighbors < min_unknown_neighbors:
                continue
            result.append(
                Frontier(
                    row=row,
                    col=col,
                    unknown_neighbors=unknown_neighbors,
                    free_neighbors=free_neighbors,
                    distance_cells=_manhattan(start, (row, col)) if start else 0,
                )
            )
    return sorted(
        result,
        key=lambda item: (
            item.distance_cells,
            -item.unknown_neighbors,
            -item.free_neighbors,
            item.row,
            item.col,
        ),
    )


def _manhattan(a: GridCell | None, b: GridCell) -> int:
    if a is None:
        return 0
    return abs(a[0] - b[0]) + abs(a[1] - b[1])
