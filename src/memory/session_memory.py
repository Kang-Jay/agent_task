from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.memory.episodic_store import EpisodicMemoryStore
from src.task.config import AgentConfig
from src.types.schema import TaskExecutionPlan


MEMORY_NAMESPACE = "visual_search"


@dataclass
class SessionState:
    session_id: str
    instruction: str
    steps: list[dict[str, Any]] = field(default_factory=list)
    negative_memory: list[str] = field(default_factory=list)
    explored_regions: dict[str, int] = field(default_factory=dict)
    retrieved_memories: list[dict[str, Any]] = field(default_factory=list)
    execution_plan: TaskExecutionPlan | None = None

    def recent_steps(self, window: int) -> list[dict[str, Any]]:
        return self.steps[-window:]


class SessionMemory:
    def __init__(self, config: AgentConfig):
        self.config = config
        self.sessions: dict[str, SessionState] = {}
        self.long_term_events: list[dict[str, Any]] = []
        self._lock = threading.RLock()
        self.trace_dir = config.trajectory_dir
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.episodic_store = EpisodicMemoryStore(
            self.trace_dir.parent / "memory" / "episodic_memory.sqlite3",
            capacity=int(config.raw["memory"]["long_term_capacity"]),
        )

    def get_or_create(self, session_id: str, instruction: str) -> SessionState:
        with self._lock:
            self._validate_session_id(session_id)
            if session_id not in self.sessions:
                self.sessions[session_id] = SessionState(session_id=session_id, instruction=instruction)
            elif self.sessions[session_id].instruction != instruction:
                raise ValueError("session_id is already bound to a different instruction")
            return self.sessions[session_id]

    def reset(self, session_id: str) -> None:
        with self._lock:
            self._validate_session_id(session_id)
            self.sessions.pop(session_id, None)
            trace_path = self._trace_path(session_id)
            if trace_path.exists():
                trace_path.unlink()

    def record_step(self, state: SessionState, step: dict[str, Any]) -> None:
        with self._lock:
            state.steps.append(step)
            best = step.get("best_candidate")
            if best:
                region = best.get("region", "unknown")
                state.explored_regions[region] = state.explored_regions.get(region, 0) + 1
            if not step.get("done") and best and step.get("confidence", 0.0) < self.config.target_visible_threshold:
                state.negative_memory.append(f"Searched {best.get('region', 'unknown')} but confidence stayed low.")
                capacity = int(self.config.raw["memory"]["negative_memory_capacity"])
                state.negative_memory = state.negative_memory[-capacity:]
            self.long_term_events.append(
                {
                    "session_id": state.session_id,
                    "instruction": state.instruction,
                    "action": step.get("action", {}).get("type"),
                    "confidence": step.get("confidence"),
                    "region": best.get("region") if best else None,
                }
            )
            capacity = int(self.config.raw["memory"]["long_term_capacity"])
            self.long_term_events = self.long_term_events[-capacity:]
            if self.config.raw["memory"]["persist_traces"]:
                self._persist_trace(state)

    def retrieve_relevant(self, state: SessionState) -> list[dict[str, Any]]:
        state.retrieved_memories = self.episodic_store.search(
            state.instruction,
            namespace=MEMORY_NAMESPACE,
            top_k=int(self.config.raw["memory"]["retrieval_top_k"]),
            exclude_session_id=state.session_id,
        )
        return list(state.retrieved_memories)

    def set_execution_plan(
        self,
        state: SessionState,
        plan: TaskExecutionPlan,
    ) -> TaskExecutionPlan:
        with self._lock:
            if state.execution_plan is not None:
                if state.execution_plan.plan_id != plan.plan_id:
                    raise ValueError("session already has a different execution plan")
                return state.execution_plan
            state.execution_plan = plan
            if self.config.raw["memory"]["persist_traces"]:
                self._persist_trace(state)
            return plan

    def update_execution_plan(
        self,
        state: SessionState,
        completion_status: dict[str, Any],
        *,
        step_id: int,
    ) -> TaskExecutionPlan | None:
        with self._lock:
            plan = state.execution_plan
            if plan is None:
                return None

            explicit_progress = {
                str(item.get("id")): item
                for item in completion_status.get("subgoal_progress", [])
                if item.get("id")
            }
            successful_actions = set(completion_status.get("successful_actions", []))
            target_located = bool(completion_status.get("target_located"))
            approach_verified = bool(completion_status.get("approach_verified"))

            for subgoal in plan.subgoals:
                progress = explicit_progress.get(subgoal.id)
                completed = bool(progress and progress.get("complete"))
                evidence = progress.get("evidence") if progress else None
                if subgoal.id == "locate_target":
                    completed = target_located
                    evidence = evidence or (
                        "target grounded by visual or simulator evidence"
                        if completed
                        else None
                    )
                elif subgoal.id == "approach_target":
                    completed = approach_verified
                    evidence = evidence or (
                        f"AI2-THOR target distance={completion_status.get('target_distance')}m"
                        if completed
                        else None
                    )
                elif subgoal.id.startswith("execute_"):
                    action_name = subgoal.id[len("execute_"):]
                    completed = completed or any(
                        action.lower() == action_name for action in successful_actions
                    )
                    evidence = evidence or (
                        f"{action_name} execution verified"
                        if completed
                        else None
                    )
                if completed:
                    subgoal.status = "completed"
                    subgoal.evidence = str(evidence) if evidence is not None else None
                else:
                    subgoal.status = "pending"
                    subgoal.evidence = None

            first_incomplete = next(
                (subgoal for subgoal in plan.subgoals if subgoal.status != "completed"),
                None,
            )
            if completion_status.get("complete"):
                plan.status = "completed"
                plan.current_subgoal_id = None
            else:
                plan.status = "in_progress"
                plan.current_subgoal_id = first_incomplete.id if first_incomplete else None
                if first_incomplete is not None:
                    first_incomplete.status = "in_progress"
            plan.last_updated_step = step_id
            if self.config.raw["memory"]["persist_traces"]:
                self._persist_trace(state)
            return plan

    def commit_execution(
        self,
        session_id: str,
        *,
        step_id: int | None = None,
        executed_action: dict[str, Any],
        done: bool,
        confidence: float,
        planner_source: str,
        skill_call: dict[str, Any] | None,
        action_success: bool,
        robot_before: dict[str, float] | None = None,
        robot_after: dict[str, float] | None = None,
        environment: dict[str, Any] | None = None,
    ) -> SessionState:
        with self._lock:
            return self._commit_execution_unlocked(
                session_id,
                step_id=step_id,
                executed_action=executed_action,
                done=done,
                confidence=confidence,
                planner_source=planner_source,
                skill_call=skill_call,
                action_success=action_success,
                robot_before=robot_before,
                robot_after=robot_after,
                environment=environment,
            )

    def _commit_execution_unlocked(
        self,
        session_id: str,
        *,
        step_id: int | None = None,
        executed_action: dict[str, Any],
        done: bool,
        confidence: float,
        planner_source: str,
        skill_call: dict[str, Any] | None,
        action_success: bool,
        robot_before: dict[str, float] | None = None,
        robot_after: dict[str, float] | None = None,
        environment: dict[str, Any] | None = None,
    ) -> SessionState:
        self._validate_session_id(session_id)
        state = self.sessions.get(session_id)
        if state is None or not state.steps:
            raise ValueError(f"session has no pending step to commit: {session_id}")
        if step_id is None:
            step = state.steps[-1]
        else:
            step = next(
                (
                    candidate
                    for candidate in reversed(state.steps)
                    if candidate.get("step_id") == step_id
                ),
                None,
            )
            if step is None:
                raise ValueError(
                    f"session {session_id} has no proposal for step_id={step_id}"
                )
            if "action_success" in step:
                raise ValueError(
                    f"session {session_id} step_id={step_id} was already committed"
                )
        step.setdefault("proposed_action", step.get("action"))
        step["action"] = dict(executed_action)
        step["executed_action"] = dict(executed_action)
        step["done"] = bool(done)
        step["confidence"] = float(confidence)
        step["planner_source"] = planner_source
        step["skill_call"] = skill_call
        step["action_success"] = bool(action_success)
        if robot_before is not None:
            step["robot_before"] = dict(robot_before)
        if robot_after is not None:
            step["robot_after"] = dict(robot_after)
        if environment is not None:
            step["environment"] = dict(environment)

        if self.long_term_events:
            event = self.long_term_events[-1]
            if event.get("session_id") == session_id:
                event["action"] = executed_action.get("type")
                event["confidence"] = float(confidence)
                event["action_success"] = bool(action_success)

        if "episodic_memory_id" not in step:
            best = step.get("best_candidate") or {}
            action_type = str(executed_action.get("type", "UNKNOWN"))
            memory_id = self.episodic_store.add(
                namespace=MEMORY_NAMESPACE,
                session_id=state.session_id,
                instruction=state.instruction,
                action=action_type,
                action_success=bool(action_success),
                confidence=float(confidence),
                region=best.get("region"),
                lesson=self._build_lesson(
                    action=action_type,
                    action_success=bool(action_success),
                    done=bool(done),
                    confidence=float(confidence),
                    region=best.get("region"),
                ),
                metadata={
                    "step_id": step.get("step_id"),
                    "planner_source": planner_source,
                    "skill_call": skill_call,
                    "robot_before": robot_before,
                    "robot_after": robot_after,
                    "environment": environment or {},
                },
            )
            step["episodic_memory_id"] = memory_id

        if self.config.raw["memory"]["persist_traces"]:
            self._persist_trace(state)
        return state

    def summarize(self, state: SessionState) -> str:
        if not state.steps:
            return "No prior steps in this session."
        last = state.steps[-1]
        explored = ", ".join(f"{region}:{count}" for region, count in sorted(state.explored_regions.items()))
        current_subgoal = (
            state.execution_plan.current_subgoal_id
            if state.execution_plan is not None
            else None
        )
        return (
            f"{len(state.steps)} steps recorded. Last action={last.get('action', {}).get('type')}; "
            f"explored regions={explored or 'none'}; negative memories={len(state.negative_memory)}; "
            f"recalled episodes={len(state.retrieved_memories)}; "
            f"current subgoal={current_subgoal or 'none'}."
        )

    def search_map(self, state: SessionState) -> dict[str, Any]:
        regions = [
            "upper left",
            "upper center",
            "upper right",
            "middle left",
            "middle center",
            "middle right",
            "lower left",
            "lower center",
            "lower right",
        ]
        visited = {region: state.explored_regions.get(region, 0) for region in regions}
        confidence_by_region: dict[str, float] = {}
        for step in state.steps:
            best = step.get("best_candidate") or {}
            region = best.get("region")
            if region:
                confidence_by_region[region] = max(
                    confidence_by_region.get(region, 0.0),
                    float(step.get("confidence", 0.0)),
                )
        unexplored = [region for region, count in visited.items() if count == 0]
        return {
            "visited_counts": visited,
            "unexplored_regions": unexplored,
            "confidence_by_region": confidence_by_region,
            "negative_memory": list(state.negative_memory),
            "recalled_memories": list(state.retrieved_memories),
        }

    def confidence_trace(self, state: SessionState) -> list[float]:
        return [float(step.get("confidence", 0.0)) for step in state.steps]

    def export_trace(self, session_id: str) -> dict[str, Any]:
        self._validate_session_id(session_id)
        state = self.sessions.get(session_id)
        if state is None:
            trace_path = self._trace_path(session_id)
            if trace_path.exists():
                return json.loads(trace_path.read_text(encoding="utf-8"))
            return {"session_id": session_id, "steps": []}
        return {
            "session_id": state.session_id,
            "instruction": state.instruction,
            "steps": state.steps,
            "negative_memory": state.negative_memory,
            "explored_regions": state.explored_regions,
            "retrieved_memories": state.retrieved_memories,
            "execution_plan": (
                state.execution_plan.to_dict()
                if state.execution_plan is not None
                else None
            ),
        }

    def _persist_trace(self, state: SessionState) -> None:
        path = self._trace_path(state.session_id)
        payload = {
            "session_id": state.session_id,
            "instruction": state.instruction,
            "steps": state.steps,
            "negative_memory": state.negative_memory,
            "explored_regions": state.explored_regions,
            "retrieved_memories": state.retrieved_memories,
            "execution_plan": (
                state.execution_plan.to_dict()
                if state.execution_plan is not None
                else None
            ),
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _build_lesson(
        *,
        action: str,
        action_success: bool,
        done: bool,
        confidence: float,
        region: str | None,
    ) -> str:
        location = region or "unknown region"
        if not action_success:
            return (
                f"{action} failed in {location}; verify action preconditions, collision state, "
                "and choose an alternative before repeating it."
            )
        if done:
            return (
                f"{action} completed the search in {location} with confidence "
                f"{confidence:.3f}; preserve the final visual evidence before stopping."
            )
        return (
            f"{action} executed successfully while searching {location} with confidence "
            f"{confidence:.3f}; compare the next observation before selecting another action."
        )

    def _trace_path(self, session_id: str) -> Path:
        self._validate_session_id(session_id)
        return self.trace_dir / f"{session_id}.json"

    @staticmethod
    def _validate_session_id(session_id: str) -> None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", session_id):
            raise ValueError(
                "session_id must be 1-128 characters using letters, digits, '.', '_' or '-'"
            )
