from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class Action:
    type: str
    args: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SkillCall:
    """Structured skill call representation.

    The skill name must be in the current task-conditioned action set.
    """
    name: str
    args: dict[str, Any] = field(default_factory=dict)
    preconditions: list[str] = field(default_factory=list)
    expected_observation: str = ""

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
    agent_mode: str = "default"
    environment_context: dict[str, Any] | None = None


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
    recalled_memories: list[dict[str, Any]] = field(default_factory=list)
    search_map: dict[str, Any] = field(default_factory=dict)
    confidence_trace: list[float] = field(default_factory=list)
    target_binding: dict[str, Any] = field(default_factory=dict)
    structured_thought: dict[str, str] = field(default_factory=dict)
    skill_call: SkillCall | None = None
    planner_source: Literal["model_planner", "rule_fallback", "simulator_oracle", "human_manual"] = "rule_fallback"
    model_info: dict[str, Any] = field(default_factory=dict)
    fallback_reason: str | None = None
    task_plan: dict[str, Any] = field(default_factory=dict)
    completion_status: dict[str, Any] = field(default_factory=dict)

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
            "recalled_memories": self.recalled_memories,
            "search_map": self.search_map,
            "confidence_trace": self.confidence_trace,
            "target_binding": self.target_binding,
            "structured_thought": self.structured_thought,
            "skill_call": self.skill_call.to_dict() if self.skill_call else None,
            "planner_source": self.planner_source,
            "model_info": self.model_info,
            "fallback_reason": self.fallback_reason,
            "task_plan": self.task_plan,
            "completion_status": self.completion_status,
        }
