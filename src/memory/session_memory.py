from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.memory.episodic_store import EpisodicMemoryStore
from src.memory.hierarchical_memory import (
    EvidenceReference,
    HierarchicalMemoryStore,
    MEMORY_LAYERS,
    normalize_identity,
)
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
    layered_memories: dict[str, list[dict[str, Any]]] = field(
        default_factory=lambda: {layer: [] for layer in MEMORY_LAYERS}
    )
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
        self.hierarchical_store = HierarchicalMemoryStore(
            self.trace_dir.parent / "memory" / "hierarchical_memory.sqlite3",
            capacity=int(config.raw["memory"]["long_term_capacity"]),
            failure_capacity=int(
                config.raw["memory"]["negative_memory_capacity"]
            ),
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
                    "step_id": step.get("step_id"),
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

    def retrieve_relevant(
        self,
        state: SessionState,
        environment_context: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        query = self._memory_query(
            state.instruction,
            environment_context=environment_context,
        )
        state.retrieved_memories = self.episodic_store.search(
            query,
            namespace=MEMORY_NAMESPACE,
            top_k=int(self.config.raw["memory"]["retrieval_top_k"]),
            exclude_session_id=state.session_id,
        )
        state.layered_memories = self.hierarchical_store.search_grouped(
            query,
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
            if "action_success" in step:
                raise ValueError(
                    f"session {session_id} latest step was already committed"
                )
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

        committed_step_id = step.get("step_id")
        event = next(
            (
                candidate
                for candidate in reversed(self.long_term_events)
                if candidate.get("session_id") == session_id
                and candidate.get("step_id") == committed_step_id
            ),
            None,
        )
        if event is not None:
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
        if "hierarchical_memory_ids" not in step:
            step["hierarchical_memory_ids"] = self._record_hierarchical_commit(
                state,
                step,
                executed_action=executed_action,
                action_success=action_success,
                confidence=confidence,
                planner_source=planner_source,
                skill_call=skill_call,
                robot_before=robot_before,
                robot_after=robot_after,
                environment=environment,
            )

        if self.config.raw["memory"]["persist_traces"]:
            self._persist_trace(state)
        return state

    def finalize_execution(
        self,
        state: SessionState,
        *,
        step_id: int,
        completion_status: dict[str, Any],
        done: bool,
        environment_context: dict[str, Any] | None = None,
    ) -> SessionState:
        with self._lock:
            step = next(
                (
                    candidate
                    for candidate in reversed(state.steps)
                    if candidate.get("step_id") == step_id
                ),
                None,
            )
            if step is None or "action_success" not in step:
                raise ValueError(
                    f"session {state.session_id} step_id={step_id} "
                    "must be committed before finalization"
                )
            step["completion_status"] = dict(completion_status)
            step["done"] = bool(done)
            if environment_context is not None:
                step["environment_context"] = dict(environment_context)
            self.update_execution_plan(
                state,
                completion_status,
                step_id=step_id,
            )
            finalized_ids = self._record_hierarchical_finalization(
                state,
                step,
                completion_status=completion_status,
                environment_context=environment_context,
            )
            step.setdefault("hierarchical_memory_ids", {}).update(
                finalized_ids
            )
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
        recalled_layered = sum(
            len(items) for items in state.layered_memories.values()
        )
        return (
            f"{len(state.steps)} steps recorded. Last action={last.get('action', {}).get('type')}; "
            f"explored regions={explored or 'none'}; negative memories={len(state.negative_memory)}; "
            f"recalled episodes={len(state.retrieved_memories)}; "
            f"recalled layered memories={recalled_layered}; "
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
            "layered_memories": {
                layer: list(items)
                for layer, items in state.layered_memories.items()
            },
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
            "layered_memories": state.layered_memories,
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
            "layered_memories": state.layered_memories,
            "execution_plan": (
                state.execution_plan.to_dict()
                if state.execution_plan is not None
                else None
            ),
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _record_hierarchical_commit(
        self,
        state: SessionState,
        step: dict[str, Any],
        *,
        executed_action: dict[str, Any],
        action_success: bool,
        confidence: float,
        planner_source: str,
        skill_call: dict[str, Any] | None,
        robot_before: dict[str, float] | None,
        robot_after: dict[str, float] | None,
        environment: dict[str, Any] | None,
    ) -> dict[str, int]:
        step_id = int(step.get("step_id", 0))
        action_type = str(executed_action.get("type") or "UNKNOWN")
        best = step.get("best_candidate") or {}
        scene = self._scene_name(environment)
        evidence = self._evidence_reference(
            state,
            step,
            source="execution_commit",
            environment=environment,
        )
        common_metadata = {
            "action": action_type,
            "action_args": dict(executed_action.get("args") or {}),
            "planner_source": planner_source,
            "region": best.get("region"),
            "scene": scene,
        }
        memory_ids: dict[str, int] = {}

        memory_ids["episode"] = self.hierarchical_store.upsert(
            layer="episode",
            identity_key=f"{state.session_id}:{step_id}",
            session_id=state.session_id,
            instruction=state.instruction,
            subject=f"step {step_id}",
            summary=self._build_lesson(
                action=action_type,
                action_success=bool(action_success),
                done=bool(step.get("done")),
                confidence=float(confidence),
                region=best.get("region"),
            ),
            evidence=evidence,
            success=bool(action_success),
            confidence=float(confidence),
            metadata=common_metadata,
        )

        task_subgoal = (
            state.execution_plan.current_subgoal_id
            if state.execution_plan is not None
            else "unplanned"
        )
        memory_ids["task"] = self.hierarchical_store.upsert(
            layer="task",
            identity_key=normalize_identity(
                {
                    "instruction": state.instruction,
                    "subgoal": task_subgoal,
                }
            ),
            session_id=state.session_id,
            instruction=state.instruction,
            subject=str(task_subgoal),
            summary=(
                f"Task subgoal {task_subgoal} executed {action_type}; "
                f"environment success={bool(action_success)}."
            ),
            evidence=evidence,
            success=bool(action_success),
            confidence=float(confidence),
            metadata={
                **common_metadata,
                "subgoal": task_subgoal,
                "execution_plan_status": (
                    state.execution_plan.status
                    if state.execution_plan is not None
                    else None
                ),
            },
        )

        if best:
            object_subject = str(
                best.get("objectId")
                or best.get("object_id")
                or best.get("label")
                or "visual_candidate"
            )
            memory_ids["object"] = self.hierarchical_store.upsert(
                layer="object",
                identity_key=normalize_identity(
                    {
                        "scene": scene,
                        "subject": object_subject,
                        "region": best.get("region"),
                    }
                ),
                session_id=state.session_id,
                instruction=state.instruction,
                subject=object_subject,
                summary=(
                    f"Observed {object_subject} in "
                    f"{best.get('region') or 'unknown region'} with "
                    f"confidence {float(confidence):.3f}."
                ),
                evidence=evidence,
                confidence=float(confidence),
                metadata={
                    **common_metadata,
                    "candidate": dict(best),
                },
            )

        if robot_after or best.get("region") or scene:
            spatial_identity = {
                "scene": scene,
                "robot_after": robot_after,
                "region": best.get("region"),
            }
            memory_ids["spatial"] = self.hierarchical_store.upsert(
                layer="spatial",
                identity_key=normalize_identity(spatial_identity),
                session_id=state.session_id,
                instruction=state.instruction,
                subject=scene or str(best.get("region") or "unknown"),
                summary=(
                    f"After {action_type}, robot pose={robot_after or 'unknown'}; "
                    f"observed region={best.get('region') or 'unknown'}."
                ),
                evidence=evidence,
                success=bool(action_success),
                confidence=float(confidence),
                metadata={
                    **common_metadata,
                    "robot_before": robot_before,
                    "robot_after": robot_after,
                },
            )

        if skill_call:
            skill_name = str(skill_call.get("name") or action_type)
            skill_args = dict(skill_call.get("args") or {})
            memory_ids["skill"] = self.hierarchical_store.upsert(
                layer="skill",
                identity_key=normalize_identity(
                    {
                        "name": skill_name,
                        "args": skill_args,
                        "success": bool(action_success),
                    }
                ),
                session_id=state.session_id,
                instruction=state.instruction,
                subject=skill_name,
                summary=(
                    f"Skill {skill_name} executed with "
                    f"success={bool(action_success)}."
                ),
                evidence=evidence,
                success=bool(action_success),
                confidence=float(confidence),
                metadata={
                    **common_metadata,
                    "skill_call": dict(skill_call),
                },
            )

        if not action_success:
            failure_reason = self._failure_reason(
                step=step,
                environment=environment,
            )
            memory_ids["failure"] = self.hierarchical_store.upsert(
                layer="failure",
                identity_key=normalize_identity(
                    {
                        "scene": scene,
                        "action": action_type,
                        "args": executed_action.get("args") or {},
                        "reason": failure_reason,
                    }
                ),
                session_id=state.session_id,
                instruction=state.instruction,
                subject=action_type,
                summary=(
                    f"{action_type} failed: {failure_reason}. "
                    "Verify preconditions before retrying."
                ),
                evidence=evidence,
                success=False,
                confidence=float(confidence),
                metadata={
                    **common_metadata,
                    "failure_reason": failure_reason,
                },
            )
        return memory_ids

    def _record_hierarchical_finalization(
        self,
        state: SessionState,
        step: dict[str, Any],
        *,
        completion_status: dict[str, Any],
        environment_context: dict[str, Any] | None,
    ) -> dict[str, int]:
        step_id = int(step.get("step_id", 0))
        action = step.get("executed_action") or step.get("action") or {}
        action_type = str(action.get("type") or "UNKNOWN")
        confidence = float(step.get("confidence", 0.0))
        scene = self._scene_name(environment_context)
        evidence = self._evidence_reference(
            state,
            step,
            source="execution_finalization",
            environment=environment_context,
        )
        current_subgoal = (
            state.execution_plan.current_subgoal_id
            if state.execution_plan is not None
            else None
        )
        memory_ids: dict[str, int] = {}

        memory_ids["task_final"] = self.hierarchical_store.upsert(
            layer="task",
            identity_key=normalize_identity(
                {
                    "instruction": state.instruction,
                    "subgoal": current_subgoal or "completed",
                }
            ),
            session_id=state.session_id,
            instruction=state.instruction,
            subject=current_subgoal or "completed",
            summary=(
                f"Task completion={bool(completion_status.get('complete'))}; "
                f"reason={completion_status.get('reason') or 'not provided'}."
            ),
            evidence=evidence,
            success=bool(completion_status.get("complete")),
            confidence=confidence,
            metadata={
                "action": action_type,
                "scene": scene,
                "completion_status": dict(completion_status),
            },
        )
        memory_ids["episode_final"] = self.hierarchical_store.upsert(
            layer="episode",
            identity_key=f"{state.session_id}:{step_id}",
            session_id=state.session_id,
            instruction=state.instruction,
            subject=f"step {step_id}",
            summary=(
                f"{action_type} environment success="
                f"{bool(step.get('action_success'))}; task completion="
                f"{bool(completion_status.get('complete'))}."
            ),
            evidence=evidence,
            success=bool(step.get("action_success")),
            confidence=confidence,
            metadata={
                "action": action_type,
                "scene": scene,
                "completion_status": dict(completion_status),
            },
        )

        context = environment_context or {}
        agent_pose = context.get("agent")
        if isinstance(agent_pose, dict):
            memory_ids["spatial_final"] = self.hierarchical_store.upsert(
                layer="spatial",
                identity_key=normalize_identity(
                    {
                        "scene": scene,
                        "agent": agent_pose,
                    }
                ),
                session_id=state.session_id,
                instruction=state.instruction,
                subject=scene or "agent_pose",
                summary=(
                    f"Verified agent state in {scene or 'unknown scene'}: "
                    f"{agent_pose}."
                ),
                evidence=evidence,
                success=bool(step.get("action_success")),
                confidence=confidence,
                metadata={
                    "action": action_type,
                    "scene": scene,
                    "agent": dict(agent_pose),
                },
            )

        objects = context.get("objects")
        if isinstance(objects, list):
            for index, item in enumerate(objects):
                if not isinstance(item, dict):
                    continue
                if item.get("visible") is False:
                    continue
                object_id = str(
                    item.get("objectId")
                    or item.get("object_id")
                    or item.get("objectType")
                    or item.get("object_type")
                    or f"object-{index}"
                )
                object_type = str(
                    item.get("objectType")
                    or item.get("object_type")
                    or object_id
                )
                memory_ids[f"object_final_{index}"] = (
                    self.hierarchical_store.upsert(
                        layer="object",
                        identity_key=normalize_identity(
                            {
                                "scene": scene,
                                "object_id": object_id,
                            }
                        ),
                        session_id=state.session_id,
                        instruction=state.instruction,
                        subject=object_id,
                        summary=(
                            f"Verified visible {object_type} "
                            f"with objectId={object_id}."
                        ),
                        evidence=evidence,
                        confidence=confidence,
                        metadata={
                            "action": action_type,
                            "scene": scene,
                            "object": dict(item),
                        },
                    )
                )
        return memory_ids

    def _evidence_reference(
        self,
        state: SessionState,
        step: dict[str, Any],
        *,
        source: str,
        environment: dict[str, Any] | None,
    ) -> EvidenceReference:
        step_id = int(step.get("step_id", 0))
        context = environment or {}
        details = {
            "action": (
                step.get("executed_action")
                or step.get("action")
                or {}
            ),
            "action_success": step.get("action_success"),
            "planner_source": step.get("planner_source"),
            "scene": self._scene_name(context),
        }
        for field in (
            "observation_path",
            "observation_phase",
            "backend",
        ):
            if field in context:
                details[field] = context[field]
        return EvidenceReference(
            session_id=state.session_id,
            step_id=step_id,
            source=source,
            reference=f"{self._trace_path(state.session_id)}#step={step_id}",
            details=details,
        )

    @staticmethod
    def _scene_name(context: dict[str, Any] | None) -> str | None:
        payload = context or {}
        scene = payload.get("scene") or payload.get("scene_name")
        if scene:
            return str(scene)
        environment = payload.get("environment")
        if isinstance(environment, dict):
            nested = environment.get("scene") or environment.get("scene_name")
            if nested:
                return str(nested)
        return None

    @staticmethod
    def _failure_reason(
        *,
        step: dict[str, Any],
        environment: dict[str, Any] | None,
    ) -> str:
        payload = environment or {}
        for key in (
            "errorMessage",
            "error_message",
            "failure_reason",
            "lastActionError",
        ):
            if payload.get(key):
                return str(payload[key])
        completion = step.get("completion_status") or {}
        if completion.get("reason"):
            return str(completion["reason"])
        return "environment reported action_success=false"

    @staticmethod
    def _memory_query(
        instruction: str,
        *,
        environment_context: dict[str, Any] | None,
    ) -> str:
        context = environment_context or {}
        parts = [instruction]
        scene = context.get("scene") or context.get("scene_name")
        if scene:
            parts.append(str(scene))
        objects = context.get("objects")
        if isinstance(objects, list):
            for item in objects:
                if not isinstance(item, dict):
                    continue
                if item.get("visible") is False:
                    continue
                object_type = item.get("objectType") or item.get("object_type")
                object_id = item.get("objectId") or item.get("object_id")
                if object_type:
                    parts.append(str(object_type))
                if object_id:
                    parts.append(str(object_id))
        return " ".join(part for part in parts if part)

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
