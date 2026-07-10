from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


SubgoalStatus = Literal["pending", "in_progress", "completed"]
ExecutionPlanStatus = Literal["in_progress", "completed", "failed", "terminated"]
ExecutionPlanSource = Literal["model_planner", "semantic_fallback"]


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
class ClickedObjectBinding:
    """Result of resolving a user click to a concrete scene object.

    Produced when the user clicks an object in the viewport/map and the
    simulator grounds it to an AI2-THOR objectId, then renders a close-up
    reference image near the object to feed the vision model.
    """
    object_id: str
    object_type: str
    affordances: dict[str, Any] = field(default_factory=dict)
    closeup_source: str = ""
    closeup_bbox: list[int] | None = None
    world_position: dict[str, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "object_type": self.object_type,
            "affordances": self.affordances,
            "closeup_source": self.closeup_source,
            "closeup_bbox": self.closeup_bbox,
            "world_position": self.world_position,
        }


@dataclass
class ExecutionSubgoal:
    id: str
    description: str
    success_evidence: str
    status: SubgoalStatus = "pending"
    evidence: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TaskExecutionPlan:
    plan_id: str
    instruction: str
    task_summary: str
    task_types: list[str]
    completion_mode: str
    subgoals: list[ExecutionSubgoal]
    current_subgoal_id: str | None
    status: ExecutionPlanStatus
    source: ExecutionPlanSource
    failure_policy: str
    limitations: list[str] = field(default_factory=list)
    revision: int = 1
    replan_count: int = 0
    vision_input_used: bool = False
    last_updated_step: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "instruction": self.instruction,
            "task_summary": self.task_summary,
            "task_types": list(self.task_types),
            "completion_mode": self.completion_mode,
            "subgoals": [subgoal.to_dict() for subgoal in self.subgoals],
            "current_subgoal_id": self.current_subgoal_id,
            "status": self.status,
            "source": self.source,
            "failure_policy": self.failure_policy,
            "limitations": list(self.limitations),
            "revision": self.revision,
            "replan_count": self.replan_count,
            "vision_input_used": self.vision_input_used,
            "last_updated_step": self.last_updated_step,
        }


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
    clicked_object_id: str | None = None
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
    execution_plan: dict[str, Any] = field(default_factory=dict)
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
            "execution_plan": self.execution_plan,
            "completion_status": self.completion_status,
        }
