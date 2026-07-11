from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from src.exploration.frontier_policy import (
    RankedFrontier,
    SemanticValueMap,
    rank_frontiers,
    reject_oracle_fields,
)
from src.mapping.depth_projector import project_depth_to_grid
from src.mapping.frontier import Frontier, extract_frontiers
from src.mapping.observations import RGBDObservation
from src.mapping.occupancy_grid import GridCell, MapUpdateResult, OccupancyGrid
from src.planning.grid_planner import ActionCommand, GridPlanner, GridPose, PathResult


@dataclass(frozen=True)
class ExplorationPlannerInput:
    observation: RGBDObservation | None
    pose: GridPose
    target_terms: tuple[str, ...] = ()
    rgb_candidates: tuple[dict[str, Any], ...] = ()
    failed_forward_cell: GridCell | None = None


@dataclass(frozen=True)
class ExplorationDecision:
    action: ActionCommand
    map_update: MapUpdateResult
    path: PathResult
    frontiers: tuple[Frontier, ...]
    ranked_frontiers: tuple[RankedFrontier, ...]
    selected_frontier: Frontier | None
    planner_source: str = "non_oracle_frontier"
    summary: dict[str, Any] = field(default_factory=dict)


class ExplorationPlanner:
    """VLFM-lite deterministic planner built from planner-safe observations.

    Inputs are limited to RGB-D, camera/agent pose, action feedback, failed
    movement hints, and RGB-derived semantic candidates. Simulator object
    metadata, hidden target positions, instance masks, and interactable poses are
    rejected before they can affect ranking or action selection.
    """

    def __init__(
        self,
        grid: OccupancyGrid,
        *,
        max_depth_m: float = 5.0,
        depth_sample_stride: int = 4,
        turn_angle_degrees: float = 30.0,
    ):
        if max_depth_m <= 0:
            raise ValueError("max_depth_m must be positive")
        if depth_sample_stride <= 0:
            raise ValueError("depth_sample_stride must be positive")
        if turn_angle_degrees <= 0:
            raise ValueError("turn_angle_degrees must be positive")
        self.grid = grid
        self.semantic_map = SemanticValueMap(grid)
        self.grid_planner = GridPlanner(grid)
        self.max_depth_m = max_depth_m
        self.depth_sample_stride = depth_sample_stride
        self.turn_angle_degrees = turn_angle_degrees
        self.attempted_frontiers: set[GridCell] = set()

    def decide(self, planner_input: ExplorationPlannerInput) -> ExplorationDecision:
        map_update = self._update_map(planner_input)
        self.semantic_map.decay()
        updated_semantics = self.semantic_map.update_from_candidates(
            planner_input.rgb_candidates,
            target_terms=planner_input.target_terms,
            current_cell=planner_input.pose.cell,
        )
        frontiers = extract_frontiers(self.grid, start=planner_input.pose.cell)
        ranked = self._rank_unattempted(frontiers)
        path = self.grid_planner.plan_to_frontier(
            planner_input.pose.cell,
            frontiers,
            ranked_frontiers=ranked,
        )
        action = self.grid_planner.next_navigation_action(
            planner_input.pose,
            path,
            turn_angle_degrees=self.turn_angle_degrees,
        )
        selected = path.goal
        selected_frontier = next(
            (frontier for frontier in frontiers if frontier.cell == selected),
            None,
        )
        if selected_frontier is not None and action.type == "INSPECT":
            self.attempted_frontiers.add(selected_frontier.cell)
        return ExplorationDecision(
            action=action,
            map_update=map_update,
            path=path,
            frontiers=tuple(frontiers),
            ranked_frontiers=tuple(ranked),
            selected_frontier=selected_frontier,
            summary={
                "grid": self.grid.summary(),
                "frontier_count": len(frontiers),
                "ranked_frontier_count": len(ranked),
                "selected_frontier": selected_frontier.to_dict()
                if selected_frontier
                else None,
                "semantic_updates": updated_semantics,
                "attempted_frontiers": len(self.attempted_frontiers),
            },
        )

    def _update_map(self, planner_input: ExplorationPlannerInput) -> MapUpdateResult:
        visited = self.grid.apply_updates(visited_cells=[planner_input.pose.cell])
        failed = MapUpdateResult()
        if planner_input.failed_forward_cell is not None:
            failed = self.grid.apply_updates(
                blocked_cells=[planner_input.failed_forward_cell]
            )
        projected = MapUpdateResult()
        if planner_input.observation is not None:
            projected = project_depth_to_grid(
                self.grid,
                planner_input.observation,
                max_depth_m=self.max_depth_m,
                sample_stride=self.depth_sample_stride,
            )
        return visited.merged(failed).merged(projected)

    def _rank_unattempted(self, frontiers: Iterable[Frontier]) -> list[RankedFrontier]:
        ranked = rank_frontiers(frontiers, semantic_map=self.semantic_map)
        return [
            item
            for item in ranked
            if item.frontier.cell not in self.attempted_frontiers
        ]

    @staticmethod
    def validate_planner_context(context: dict[str, Any]) -> None:
        reject_oracle_fields(context)
