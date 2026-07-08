from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Action:
    type: str
    args: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Candidate:
    label: str
    bbox: list[int]
    confidence: float
    color_name: str
    region: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ObservationAnalysis:
    image_size: tuple[int, int]
    scene_summary: str
    candidates: list[Candidate]
    best_candidate: Candidate | None
    target_visible: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_size": list(self.image_size),
            "scene_summary": self.scene_summary,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "best_candidate": self.best_candidate.to_dict() if self.best_candidate else None,
            "target_visible": self.target_visible,
        }


@dataclass(frozen=True)
class AgentRequest:
    session_id: str
    instruction: str
    observation_image: str
    step_id: int = 0
    target_crop: str | None = None
    clicked_point: list[int] | None = None


@dataclass(frozen=True)
class AgentResponse:
    session_id: str
    step_id: int
    thought: str
    action: Action
    confidence: float
    done: bool
    observation: ObservationAnalysis
    retrieved_hints: list[str]
    memory_summary: str
    replay: list[dict[str, Any]]
    search_map: dict[str, Any] = field(default_factory=dict)
    confidence_trace: list[float] = field(default_factory=list)
    target_binding: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "step_id": self.step_id,
            "thought": self.thought,
            "action": self.action.to_dict(),
            "confidence": self.confidence,
            "done": self.done,
            "observation": self.observation.to_dict(),
            "retrieved_hints": self.retrieved_hints,
            "memory_summary": self.memory_summary,
            "replay": self.replay,
            "search_map": self.search_map,
            "confidence_trace": self.confidence_trace,
            "target_binding": self.target_binding,
        }
