from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

from src.exploration.frontier_policy import RankedFrontier, rank_frontiers
from src.mapping.frontier import Frontier
from src.mapping.occupancy_grid import GridCell, OccupancyGrid


@dataclass(frozen=True)
class GridPose:
    row: int
    col: int
    heading_degrees: float

    @property
    def cell(self) -> GridCell:
        return self.row, self.col


@dataclass(frozen=True)
class PathResult:
    status: str
    path: list[GridCell]
    cost: int
    goal: GridCell | None = None

    @property
    def ok(self) -> bool:
        return self.status in {"ok", "already_there"}


@dataclass(frozen=True)
class ActionCommand:
    type: str
    args: dict[str, float | int | str]
    reason: str = ""

    def to_dict(self) -> dict[str, object]:
        return {"type": self.type, "args": dict(self.args), "reason": self.reason}


class GridPlanner:
    """Deterministic frontier planner over the planner-safe occupancy grid."""

    def __init__(self, grid: OccupancyGrid):
        self.grid = grid

    def plan_path(
        self,
        start: GridCell,
        goal: GridCell,
        *,
        allow_unknown: bool = False,
    ) -> PathResult:
        if not self.grid.in_bounds(*start) or not self.grid.in_bounds(*goal):
            return PathResult(status="out_of_bounds", path=[], cost=0, goal=goal)
        if start == goal:
            return PathResult(status="already_there", path=[start], cost=0, goal=goal)
        if not self.grid.is_traversable(*start, allow_unknown=allow_unknown):
            return PathResult(status="start_blocked", path=[], cost=0, goal=goal)
        if not self.grid.is_traversable(*goal, allow_unknown=allow_unknown):
            return PathResult(status="goal_blocked", path=[], cost=0, goal=goal)

        frontier: list[tuple[int, int, GridCell]] = []
        heapq.heappush(frontier, (0, 0, start))
        came_from: dict[GridCell, GridCell | None] = {start: None}
        cost_so_far: dict[GridCell, int] = {start: 0}
        counter = 0

        while frontier:
            _, _, current = heapq.heappop(frontier)
            if current == goal:
                path = _reconstruct_path(came_from, current)
                return PathResult(status="ok", path=path, cost=len(path) - 1, goal=goal)
            for neighbor in self._ordered_neighbors(current):
                if not self.grid.is_traversable(*neighbor, allow_unknown=allow_unknown):
                    continue
                new_cost = cost_so_far[current] + 1
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + _manhattan(neighbor, goal)
                    counter += 1
                    heapq.heappush(frontier, (priority, counter, neighbor))
                    came_from[neighbor] = current

        return PathResult(status="no_path", path=[], cost=0, goal=goal)

    def plan_to_frontier(
        self,
        start: GridCell,
        frontiers: list[Frontier],
        *,
        ranked_frontiers: list[RankedFrontier] | None = None,
    ) -> PathResult:
        candidates = ranked_frontiers or rank_frontiers(frontiers)
        for candidate in candidates:
            frontier = candidate.frontier if isinstance(candidate, RankedFrontier) else candidate
            path = self.plan_path(start, frontier.cell)
            if path.ok:
                return path
        return PathResult(status="no_reachable_frontier", path=[], cost=0)

    def next_navigation_action(
        self,
        pose: GridPose,
        path: PathResult,
        *,
        turn_angle_degrees: float = 30.0,
        heading_tolerance_degrees: float = 1.0,
    ) -> ActionCommand:
        if not path.ok or not path.path:
            return ActionCommand("INSPECT", {"reason": path.status}, "no traversable path")
        if len(path.path) == 1:
            return ActionCommand("INSPECT", {"reason": "frontier reached"}, "frontier reached")
        current = path.path[0]
        next_cell = path.path[1]
        if current != pose.cell:
            return ActionCommand(
                "INSPECT",
                {"reason": "pose/path mismatch"},
                "planner pose does not match path start",
            )
        target_heading = _heading_for_step(current, next_cell)
        delta = _signed_heading_delta(pose.heading_degrees, target_heading)
        if abs(delta) <= heading_tolerance_degrees:
            return ActionCommand("MOVE_FORWARD", {"distance": 1}, "advance along planned path")
        turn = min(abs(delta), turn_angle_degrees)
        if delta > 0:
            return ActionCommand("TURN_RIGHT", {"angle": turn}, "rotate toward planned path")
        return ActionCommand("TURN_LEFT", {"angle": turn}, "rotate toward planned path")

    def _ordered_neighbors(self, cell: GridCell) -> list[GridCell]:
        row, col = cell
        candidates = ((row + 1, col), (row, col + 1), (row - 1, col), (row, col - 1))
        return [(r, c) for r, c in candidates if self.grid.in_bounds(r, c)]


def _heading_for_step(current: GridCell, next_cell: GridCell) -> float:
    row, col = current
    next_row, next_col = next_cell
    if next_row > row:
        return 0.0
    if next_col > col:
        return 90.0
    if next_row < row:
        return 180.0
    if next_col < col:
        return 270.0
    raise ValueError("next cell must differ from current cell")


def _signed_heading_delta(current: float, target: float) -> float:
    return (target - current + 180.0) % 360.0 - 180.0


def _manhattan(a: GridCell, b: GridCell) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _reconstruct_path(
    came_from: dict[GridCell, GridCell | None],
    current: GridCell,
) -> list[GridCell]:
    path = [current]
    while came_from[current] is not None:
        current = came_from[current]  # type: ignore[assignment]
        path.append(current)
    path.reverse()
    return path
