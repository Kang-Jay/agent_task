from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Iterable

from src.mapping.frontier import Frontier
from src.mapping.occupancy_grid import GridCell, OccupancyGrid


FORBIDDEN_ORACLE_KEYS = {
    "objects",
    "objectid",
    "objecttype",
    "objectposition",
    "targetpose",
    "targetposition",
    "matchedpose",
    "recommendedaction",
    "interactableposes",
    "instancemasks",
}


class OracleLeakageError(ValueError):
    """Raised when planner-safe evidence contains simulator oracle fields."""


@dataclass(frozen=True)
class RankedFrontier:
    frontier: Frontier
    score: float
    semantic_score: float
    information_score: float
    distance_penalty: float

    def to_dict(self) -> dict[str, float | int]:
        data = self.frontier.to_dict()
        data.update(
            {
                "score": self.score,
                "semantic_score": self.semantic_score,
                "information_score": self.information_score,
                "distance_penalty": self.distance_penalty,
            }
        )
        return data


class SemanticValueMap:
    """Target-evidence value map built only from planner-safe detections."""

    def __init__(self, grid: OccupancyGrid):
        self.grid = grid
        self.values = [
            [0.0 for _ in range(grid.spec.width)]
            for _ in range(grid.spec.height)
        ]

    def decay(self, factor: float = 0.92) -> None:
        if not 0.0 <= factor <= 1.0:
            raise ValueError("decay factor must be in [0, 1]")
        for row in range(self.grid.spec.height):
            for col in range(self.grid.spec.width):
                self.values[row][col] *= factor

    def update_from_candidates(
        self,
        candidates: Iterable[dict[str, Any]],
        *,
        target_terms: Iterable[str],
        current_cell: GridCell | None = None,
        default_range_m: float = 1.0,
    ) -> int:
        terms = [_normalize(term) for term in target_terms if _normalize(term)]
        updated = 0
        for candidate in candidates:
            reject_oracle_fields(candidate)
            label = _normalize(str(candidate.get("label") or candidate.get("name") or ""))
            if terms and not any(term in label or label in term for term in terms):
                continue
            confidence = _clamp(float(candidate.get("confidence", 0.0)), 0.0, 1.0)
            if confidence <= 0:
                continue
            cell = _candidate_cell(self.grid, candidate, current_cell, default_range_m)
            if cell is None:
                continue
            row, col = cell
            self.values[row][col] = min(1.0, self.values[row][col] + confidence)
            updated += 1
        return updated

    def value_at(self, row: int, col: int, *, radius: int = 1) -> float:
        value = 0.0
        for r in range(max(0, row - radius), min(self.grid.spec.height, row + radius + 1)):
            for c in range(max(0, col - radius), min(self.grid.spec.width, col + radius + 1)):
                dist = abs(row - r) + abs(col - c)
                value = max(value, self.values[r][c] / (1 + dist))
        return value


def rank_frontiers(
    frontiers: Iterable[Frontier],
    *,
    semantic_map: SemanticValueMap | None = None,
    information_weight: float = 1.0,
    semantic_weight: float = 2.0,
    distance_weight: float = 0.25,
) -> list[RankedFrontier]:
    ranked: list[RankedFrontier] = []
    for frontier in frontiers:
        semantic_score = (
            semantic_map.value_at(frontier.row, frontier.col)
            if semantic_map is not None
            else 0.0
        )
        information_score = float(frontier.unknown_neighbors)
        distance_penalty = float(frontier.distance_cells)
        score = (
            information_weight * information_score
            + semantic_weight * semantic_score
            - distance_weight * distance_penalty
        )
        ranked.append(
            RankedFrontier(
                frontier=frontier,
                score=score,
                semantic_score=semantic_score,
                information_score=information_score,
                distance_penalty=distance_penalty,
            )
        )
    return sorted(
        ranked,
        key=lambda item: (-item.score, item.frontier.distance_cells, item.frontier.row, item.frontier.col),
    )


def reject_oracle_fields(value: Any, *, path: str = "$") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            normalized_key = _normalize_key(str(key))
            if normalized_key in FORBIDDEN_ORACLE_KEYS:
                raise OracleLeakageError(f"oracle field is not planner-safe: {path}.{key}")
            reject_oracle_fields(item, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            reject_oracle_fields(item, path=f"{path}[{index}]")


def _candidate_cell(
    grid: OccupancyGrid,
    candidate: dict[str, Any],
    current_cell: GridCell | None,
    default_range_m: float,
) -> GridCell | None:
    if "world_x" in candidate and "world_z" in candidate:
        cell = grid.world_to_grid(float(candidate["world_x"]), float(candidate["world_z"]))
        return cell if grid.in_bounds(*cell) else None
    if current_cell is None:
        return None
    bearing = math.radians(float(candidate.get("bearing_degrees", 0.0)))
    range_m = float(candidate.get("range_m", default_range_m))
    if range_m <= 0:
        return current_cell
    current_x, current_z = grid.grid_to_world(*current_cell)
    cell = grid.world_to_grid(
        current_x + math.sin(bearing) * range_m,
        current_z + math.cos(bearing) * range_m,
    )
    return cell if grid.in_bounds(*cell) else None


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", text.lower())


def _normalize_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
