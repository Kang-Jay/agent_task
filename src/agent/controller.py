from __future__ import annotations

import math

from PIL import Image

from src.agent.model_adapter import ModelAdapter
from src.agent.task_semantics import TaskPlan, TaskSemantics
from src.memory.session_memory import SessionMemory
from src.rag.retriever import HintRetriever
from src.simulation.task_verifier import TaskVerifier
from src.task.config import AgentConfig, load_config
from src.types.schema import (
    Action,
    AgentRequest,
    AgentResponse,
    ExecutionSubgoal,
    SkillCall,
    TaskExecutionPlan,
)
from src.vision.heuristic_vision import HeuristicVision
from src.vision.image_io import crop_from_point, image_to_data_url, load_image_from_any


class EmbodiedSearchAgent:
    def __init__(
        self,
        config: AgentConfig | None = None,
        model_adapter: ModelAdapter | None = None,
        strict_vlm: bool = False,
    ):
        self.config = config or load_config()
        self.strict_vlm = bool(strict_vlm)
        self.vision = HeuristicVision(self.config)
        self.memory = SessionMemory(self.config)
        self.retriever = HintRetriever(self.config)
        self.model_adapter = model_adapter or ModelAdapter()
        self.task_semantics = TaskSemantics()
        self.task_verifier = TaskVerifier()

    def reset(self, session_id: str) -> dict[str, str]:
        self.memory.reset(session_id)
        return {"status": "reset", "session_id": session_id}

    def step(self, request: AgentRequest) -> AgentResponse:
        self._validate_request(request)
        observation = load_image_from_any(request.observation_image, root=self.config.path.parent.parent)
        target_crop = self._resolve_target_crop(observation, request)
        state = self.memory.get_or_create(request.session_id, request.instruction)
        task_plan = self.task_semantics.analyze(
            request.instruction,
            mode=request.agent_mode,
            legacy_actions=self.config.allowed_actions,
        )
        recalled_memories = self.memory.retrieve_relevant(
            state,
            request.environment_context,
        )
        analysis = self.vision.analyze(observation, request.instruction, target_crop)
        hints = self.retriever.retrieve(request.instruction, state)

        vision_confidence = analysis.best_candidate.confidence if analysis.best_candidate else 0.0
        completion_status = self.task_verifier.verify(
            task_plan,
            steps=state.steps,
            target_visible=analysis.target_visible,
            confidence=vision_confidence,
            stop_confidence_threshold=self.config.stop_confidence_threshold,
            environment_context=request.environment_context,
        ).to_dict()
        execution_plan = self._ensure_execution_plan(
            state=state,
            task_plan=task_plan,
            request=request,
            analysis=analysis,
            observation=observation,
            target_crop=target_crop,
        )
        execution_plan = self.memory.update_execution_plan(
            state,
            completion_status,
            step_id=request.step_id,
        ) or execution_plan

        has_successful_real_vlm_step = self._has_successful_real_vlm_step(state)
        approach_guidance_should_yield = (
            has_successful_real_vlm_step
            and self._approach_guidance_should_yield(
                task_plan=task_plan,
                completion_status=completion_status,
                state=state,
            )
        )
        interaction_ready_for_model = (
            approach_guidance_should_yield
            and self._interaction_continuation_ready(
                task_plan=task_plan,
                completion_status=completion_status,
                environment_context=request.environment_context,
            )
        )
        approach_context = (
            (request.environment_context or {}).get("approach") or {}
        )
        approach_action = None
        if has_successful_real_vlm_step and not interaction_ready_for_model:
            approach_action = self._verified_approach_action(
                task_plan=task_plan,
                completion_status=completion_status,
                environment_context=request.environment_context,
                state=state,
                allow_yield=interaction_ready_for_model,
            )
        if approach_action is not None and not self.strict_vlm:
            action = approach_action
            skill_call = SkillCall(
                name="APPROACH_TARGET",
                args={
                    "objectId": approach_context.get("objectId"),
                    "action": action.to_dict(),
                },
                preconditions=[
                    "target object is grounded",
                    "AI2-THOR returned a complete path to an interactable pose",
                ],
                expected_observation=(
                    "agent pose advances toward the verified target "
                    "interaction pose"
                ),
            )
            planner_source = "simulator_oracle"
            fallback_reason = "verified_approach_navigation"
            planner_confidence = None
            model_info = {
                "status": "skill_planner",
                "skill": "APPROACH_TARGET",
                "vision_input_used": False,
                "path_status": approach_context.get("path_status"),
            }
        else:
            action, skill_call, planner_source, fallback_reason, planner_confidence, model_info = self._plan_with_model(
                analysis,
                vision_confidence,
                hints,
                state,
                request,
                observation,
                target_crop,
                task_plan,
                execution_plan,
                completion_status,
            )
            if interaction_ready_for_model and planner_source == "model_planner":
                model_info["interaction_decision_owner"] = "vlm_planner"
                model_info["prior_real_vlm_step"] = True
                model_info["missing_actions"] = list(
                    completion_status.get("missing_actions") or []
                )
                model_info["path_status"] = approach_context.get("path_status")
        confidence = (
            min(vision_confidence, planner_confidence)
            if planner_source == "model_planner" and planner_confidence is not None
            else vision_confidence
        )

        # Validate action
        if not task_plan.supported:
            if self.strict_vlm:
                raise RuntimeError(
                    "strict VLM mode rejected an unsupported task: "
                    f"{task_plan.clarification or request.instruction}"
                )
            action = Action(
                "ASK_CLARIFY",
                {"reason": task_plan.clarification or "unsupported embodied capability"},
            )
            skill_call = SkillCall(
                name="ASK_CLARIFY",
                args=action.args,
                preconditions=[],
                expected_observation="user confirms a supported task formulation",
            )
            planner_source = "rule_fallback"
            fallback_reason = "unsupported_task_capability"
            model_info["decision_status"] = "rejected_unsupported_task"
        elif action.type not in task_plan.action_candidates:
            if self.strict_vlm:
                raise RuntimeError(
                    "strict VLM mode rejected illegal action: "
                    f"{action.type} not in {task_plan.action_candidates}"
                )
            action = Action("ASK_CLARIFY", {"reason": f"illegal action blocked: {action.type}"})
            skill_call = SkillCall(name="ASK_CLARIFY", args={}, preconditions=[], expected_observation="request clarification")
            planner_source = "rule_fallback"
            fallback_reason = "illegal_action"
            model_info["decision_status"] = "rejected_illegal_action"

        completion_status = self.task_verifier.verify(
            task_plan,
            steps=state.steps,
            target_visible=analysis.target_visible,
            confidence=confidence,
            stop_confidence_threshold=self.config.stop_confidence_threshold,
            environment_context=request.environment_context,
        ).to_dict()
        execution_plan = self.memory.update_execution_plan(
            state,
            completion_status,
            step_id=request.step_id,
        ) or execution_plan
        done = self._action_finishes_step(
            action,
            completion_status=completion_status,
        )
        premature_sit_action = (
            task_plan.completion_mode == "approximate_sit"
            and action.type == "Crouch"
            and not completion_status.get("approach_verified")
        )
        if (
            action.type in {"STOP", "Done"}
            and not completion_status["complete"]
        ) or premature_sit_action:
            keep_model_replan_source = False
            if task_plan.is_visual_search and not self.strict_vlm:
                action = self._rule_fallback_planner(confidence, analysis.target_visible, state)
                skill_call = SkillCall(
                    name=action.type,
                    args=action.args,
                    preconditions=[],
                    expected_observation=f"Execute {action.type}",
                )
                fallback_reason = "stop_confidence_too_low"
            elif task_plan.completion_mode == "approximate_sit" and not self.strict_vlm:
                action = self._continue_approximate_sit(
                    completion_status=completion_status,
                    confidence=confidence,
                    target_visible=analysis.target_visible,
                    state=state,
                )
                skill_call = SkillCall(
                    name=action.type,
                    args=action.args,
                    preconditions=[],
                    expected_observation=f"Execute {action.type} and verify simulator state",
                )
                fallback_reason = (
                    "premature_crouch_replanned"
                    if premature_sit_action
                    else "premature_done_replanned"
                )
            elif self.model_adapter.available() and planner_source == "model_planner":
                rejected_action = action.to_dict()
                replan_completion_status = dict(completion_status)
                replan_completion_status["last_rejected_action"] = rejected_action
                replan_completion_status["rejection_reason"] = (
                    "Verifier rejected terminal action because the task is not complete. "
                    "Use missing_actions, current_subgoal_id, and AI2-THOR environment "
                    "context to choose the next non-terminal action."
                )
                (
                    action,
                    skill_call,
                    planner_source,
                    fallback_reason,
                    planner_confidence,
                    model_info,
                ) = self._plan_with_model(
                    analysis,
                    vision_confidence,
                    hints,
                    state,
                    request,
                    observation,
                    target_crop,
                    task_plan,
                    execution_plan,
                    replan_completion_status,
                )
                model_info["decision_status"] = "vlm_replanned_after_rejected_done"
                model_info["rejected_action"] = rejected_action
                model_info["rejection_reason"] = replan_completion_status[
                    "rejection_reason"
                ]
                keep_model_replan_source = planner_source == "model_planner"
                if action.type in {"STOP", "Done"}:
                    if self.strict_vlm:
                        raise RuntimeError(
                            "strict VLM mode received a second premature terminal action"
                        )
                    action = Action(
                        "INSPECT",
                        {
                            "reason": (
                                "VLM returned a terminal action again after verifier "
                                "rejection; refresh evidence before the next VLM plan."
                            )
                        },
                    )
                    skill_call = SkillCall(
                        name=action.type,
                        args=action.args,
                        preconditions=[
                            "VLM terminal action was rejected twice by verifier"
                        ],
                        expected_observation=(
                            "refresh current evidence so the next action is planned by VLM"
                        ),
                    )
                    planner_source = "rule_fallback"
                    fallback_reason = "premature_done_blocked_after_vlm_replan"
                    keep_model_replan_source = False
            elif not self.strict_vlm:
                action = self._continue_supported_task(
                    task_plan=task_plan,
                    completion_status=completion_status,
                    confidence=confidence,
                    target_visible=analysis.target_visible,
                    environment_context=request.environment_context,
                    state=state,
                )
                skill_call = SkillCall(
                    name=action.type,
                    args=action.args,
                    preconditions=[],
                    expected_observation=(
                        "execute the next task action and verify simulator state"
                    ),
                )
                fallback_reason = "premature_done_replanned"
            else:
                raise RuntimeError(
                    "strict VLM mode could not obtain a non-terminal action after verifier rejection"
                )
            if not keep_model_replan_source:
                planner_source = "rule_fallback"
            model_info.setdefault("decision_status", "rejected_premature_termination")
            done = self._action_finishes_step(
                action,
                completion_status=completion_status,
            )

        model_summary = str(model_info.get("thought_summary") or "").strip()
        thought = self._build_thought(
            analysis.scene_summary,
            action,
            confidence,
            hints,
            done,
            completion_status=completion_status,
        )
        structured_thought = self._build_structured_thought(
            analysis,
            action,
            confidence,
            hints,
            done,
            completion_status=completion_status,
        )
        structured_thought["decision_trace"] = self._build_decision_trace(
            planner_source=planner_source,
            model_info=model_info,
            action=action,
            completion_status=completion_status,
        )

        layered_memory_ids = {
            layer: [int(memory["id"]) for memory in memories]
            for layer, memories in state.layered_memories.items()
        }
        model_info["memory_context_ids_by_layer"] = layered_memory_ids
        step_record = {
            "step_id": request.step_id,
            "thought": thought,
            "action": action.to_dict(),
            "confidence": confidence,
            "done": done,
            "best_candidate": analysis.best_candidate.to_dict() if analysis.best_candidate else None,
            "candidate_count": len(analysis.candidates),
            "retrieved_hints": hints,
            "planner_source": planner_source,
            "fallback_reason": fallback_reason if planner_source != "model_planner" else None,
            "model_info": model_info,
            "recalled_memory_ids": [memory["id"] for memory in recalled_memories],
            "recalled_layered_memory_ids": layered_memory_ids,
            "task_plan": task_plan.to_dict(),
            "execution_plan": execution_plan.to_dict(),
            "completion_status": completion_status,
        }
        self.memory.record_step(state, step_record)

        return AgentResponse(
            session_id=request.session_id,
            step_id=request.step_id,
            thought=thought,
            action=action,
            confidence=confidence,
            done=done,
            observation=analysis,
            retrieved_hints=hints,
            memory_summary=self.memory.summarize(state),
            replay=state.recent_steps(self.config.history_window),
            recalled_memories=recalled_memories,
            search_map=self.memory.search_map(state),
            confidence_trace=self.memory.confidence_trace(state),
            target_binding={
                "language": bool(request.instruction.strip()),
                "clicked_point": request.clicked_point,
                "clicked_object_id": request.clicked_object_id,
                "target_crop": bool(request.target_crop or request.clicked_point),
                "crop_source": (
                    "closeup_render" if (request.target_crop and request.clicked_object_id)
                    else "point_crop" if request.target_crop
                    else None
                ),
                "mode": "multimodal" if request.target_crop or request.clicked_point else "language_only",
            },
            structured_thought=structured_thought,
            skill_call=skill_call,
            planner_source=planner_source,
            model_info=model_info,
            fallback_reason=fallback_reason,
            task_plan=task_plan.to_dict(),
            execution_plan=execution_plan.to_dict(),
            completion_status=completion_status,
        )

    def audit(self) -> dict[str, object]:
        return {
            "config_path": str(self.config.path),
            "pipeline": self.config.raw["pipeline"]["stages"],
            "allowed_actions": self.config.allowed_actions,
            "ai2thor_action_catalog": self.task_semantics.catalog.summary(),
            "max_steps": self.config.max_steps,
            "stop_confidence_threshold": self.config.stop_confidence_threshold,
            "image_size": list(self.config.image_size),
            "memory": {
                "backend": "sqlite",
                "capacity": int(self.config.raw["memory"]["long_term_capacity"]),
                "retrieval_top_k": int(self.config.raw["memory"]["retrieval_top_k"]),
                "database_path": str(self.memory.episodic_store.db_path),
                "stored_episodes": self.memory.episodic_store.count(
                    namespace="visual_search"
                ),
                "hierarchical_database_path": str(
                    self.memory.hierarchical_store.db_path
                ),
                "layer_counts": self.memory.hierarchical_store.layer_counts(),
            },
            "status": "ok",
        }

    def export_trace(self, session_id: str) -> dict[str, object]:
        return self.memory.export_trace(session_id)

    def commit_execution(
        self,
        session_id: str,
        response: dict[str, object],
        *,
        step_id: int | None = None,
        action_success: bool,
        robot_before: dict[str, float] | None = None,
        robot_after: dict[str, float] | None = None,
        environment: dict[str, object] | None = None,
        environment_context: dict[str, object] | None = None,
    ) -> dict[str, object]:
        state = self.memory.commit_execution(
            session_id,
            step_id=step_id,
            executed_action=dict(response["action"]),
            done=bool(response.get("done", False)),
            confidence=float(response.get("confidence", 0.0)),
            planner_source=str(response.get("planner_source", "rule_fallback")),
            skill_call=response.get("skill_call"),
            action_success=action_success,
            robot_before=robot_before,
            robot_after=robot_after,
            environment=environment,
        )
        task_plan_payload = response.get("task_plan") or {}
        task_plan = self.task_semantics.analyze(
            state.instruction,
            mode=str(task_plan_payload.get("mode") or "default"),
            legacy_actions=self.config.allowed_actions,
        )
        observation = response.get("observation") or {}
        verification = self.task_verifier.verify(
            task_plan,
            steps=state.steps,
            target_visible=bool(observation.get("target_visible")),
            confidence=float(response.get("confidence", 0.0)),
            stop_confidence_threshold=self.config.stop_confidence_threshold,
            environment_context=environment_context,
        ).to_dict()
        action_type = str((response.get("action") or {}).get("type") or "")
        done = bool(verification["complete"]) or action_type == "ASK_CLARIFY"
        state = self.memory.finalize_execution(
            state,
            step_id=(
                int(step_id)
                if step_id is not None
                else int(state.steps[-1]["step_id"])
            ),
            completion_status=verification,
            done=done,
            environment_context=environment_context,
        )
        return {
            "memory_summary": self.memory.summarize(state),
            "replay": state.recent_steps(self.config.history_window),
            "recalled_memories": list(state.retrieved_memories),
            "search_map": self.memory.search_map(state),
            "confidence_trace": self.memory.confidence_trace(state),
            "execution_plan": (
                state.execution_plan.to_dict()
                if state.execution_plan is not None
                else {}
            ),
            "completion_status": verification,
            "done": done,
        }

    def _validate_request(self, request: AgentRequest) -> None:
        if not request.session_id.strip():
            raise ValueError("session_id is required")
        if not request.instruction.strip():
            raise ValueError("instruction is required")
        if request.step_id < 0 or request.step_id > self.config.max_steps:
            raise ValueError("step_id is outside configured max_steps")

    @staticmethod
    def _action_finishes_step(
        action: Action,
        *,
        completion_status: dict[str, object],
    ) -> bool:
        if action.type in {"STOP", "Done"}:
            return bool(completion_status.get("complete", False))
        return action.type == "ASK_CLARIFY"

    def _resolve_target_crop(self, observation: Image.Image, request: AgentRequest) -> Image.Image | None:
        if request.target_crop:
            return load_image_from_any(request.target_crop, root=self.config.path.parent.parent)
        if request.clicked_point:
            patch_size = int(self.config.raw["vision"]["candidate_patch_size"])
            return crop_from_point(observation, request.clicked_point, patch_size)
        return None

    def _plan_with_model(
        self,
        analysis,
        confidence: float,
        hints: list[str],
        state,
        request: AgentRequest,
        observation: Image.Image,
        target_crop: Image.Image | None,
        task_plan: TaskPlan,
        execution_plan: TaskExecutionPlan,
        completion_status: dict[str, object],
    ) -> tuple[Action, SkillCall | None, str, str | None, float | None, dict[str, object]]:
        """Plan one action, optionally forbidding every non-VLM fallback.

        Returns:
            action, skill_call, planner_source, fallback_reason,
            planner_confidence, model_info
        """
        if not self.model_adapter.available():
            if self.strict_vlm:
                raise RuntimeError(
                    "strict VLM mode requires an available multimodal model"
                )
            action = self._rule_fallback_planner(confidence, analysis.target_visible, state)
            skill_call = SkillCall(name=action.type, args=action.args, preconditions=[], expected_observation=f"Execute {action.type}")
            return (
                action,
                skill_call,
                "rule_fallback",
                "no_api_key",
                None,
                {
                    "status": "unavailable",
                    "vision_input_used": False,
                },
            )

        # Build payload for model planner
        model_allowed_actions = self._model_allowed_actions(task_plan)
        model_action_specs = [
            spec
            for spec in task_plan.action_specs
            if spec.get("name") in model_allowed_actions
        ]
        payload = {
            "instruction": request.instruction,
            "observation_summary": analysis.scene_summary,
            "candidates": [c.to_dict() for c in analysis.candidates[:3]],
            "confidence": confidence,
            "memory_summary": self.memory.summarize(state),
            "negative_memory": state.negative_memory,
            "explored_regions": state.explored_regions,
            "retrieved_hints": hints,
            "episodic_memories": state.retrieved_memories,
            "layered_memories": state.layered_memories,
            "allowed_actions": model_allowed_actions,
            "action_specs": model_action_specs,
            "terminal_actions": ["STOP", "Done", "ASK_CLARIFY"],
            "task_plan": task_plan.to_dict(),
            "execution_plan": execution_plan.to_dict(),
            "environment_context": self._planner_environment_context(
                request.environment_context
            ),
            "completion_status": completion_status,
            "current_step": request.step_id,
            "max_steps": self.config.max_steps,
            "observation_image": self._model_observation_data_url(observation),
            "target_crop": image_to_data_url(target_crop) if target_crop is not None else None,
            "require_vision": True,
            "strict_vlm": self.strict_vlm,
        }

        result = self.model_adapter.plan_action(payload)

        # Check if model call failed
        if "error" in result:
            if self.strict_vlm:
                error_detail = (
                    result.get("provider_errors")
                    or result.get("errors")
                    or result.get("error")
                )
                raise RuntimeError(
                    "strict VLM model call failed: "
                    f"{result.get('fallback_reason') or result.get('error')}; "
                    f"details={error_detail}"
                )
            action = self._rule_fallback_planner(confidence, analysis.target_visible, state)
            skill_call = SkillCall(name=action.type, args=action.args, preconditions=[], expected_observation=f"Execute {action.type}")
            fallback_reason = result.get("fallback_reason", "model_api_error")
            return (
                action,
                skill_call,
                "rule_fallback",
                fallback_reason,
                None,
                {
                    "status": "error",
                    "vision_input_used": False,
                    "errors": result.get("errors", []),
                },
            )

        # Parse model output
        try:
            action_dict = result.get("action", {})
            action_type = action_dict.get("type")
            action_args = action_dict.get("args", {})

            if not action_type:
                raise ValueError("Missing action type in model response")

            action = Action(action_type, action_args)

            # Parse skill_call if present
            skill_dict = result.get("skill_call")
            if skill_dict:
                skill_call = SkillCall(
                    name=skill_dict.get("name", action_type),
                    args=skill_dict.get("args", action_args),
                    preconditions=skill_dict.get("preconditions", []),
                    expected_observation=skill_dict.get("expected_observation", "")
                )
            else:
                skill_call = SkillCall(name=action_type, args=action_args, preconditions=[], expected_observation=f"Execute {action_type}")

            planner_confidence = float(result.get("confidence", confidence))
            planner_confidence = max(0.0, min(1.0, planner_confidence))
            if self.strict_vlm and (
                not bool(result.get("vision_input_used"))
                or not str(result.get("provider_used") or "").strip()
                or not str(result.get("model_used") or "").strip()
            ):
                raise RuntimeError(
                    "strict VLM mode requires a successful multimodal model audit"
                )
            return (
                action,
                skill_call,
                "model_planner",
                None,
                planner_confidence,
                {
                    "status": "ok",
                    "provider": result.get("provider_used"),
                    "model": result.get("model_used"),
                    "vision_input_used": bool(result.get("vision_input_used", False)),
                    "thought_summary": result.get("thought_summary"),
                    "task_progress": result.get("task_progress"),
                    "target_visible": bool(result.get("target_visible", False)),
                    "target_confidence": float(result.get("target_confidence", 0.0) or 0.0),
                    "validation_repaired": bool(
                        result.get("validation_repaired", False)
                    ),
                },
            )

        except Exception as e:
            # Model output parsing failed
            if self.strict_vlm:
                raise RuntimeError(
                    f"strict VLM response parsing/validation failed: {e}"
                ) from e
            action = self._rule_fallback_planner(confidence, analysis.target_visible, state)
            skill_call = SkillCall(name=action.type, args=action.args, preconditions=[], expected_observation=f"Execute {action.type}")
            return (
                action,
                skill_call,
                "rule_fallback",
                f"parse_error: {str(e)[:50]}",
                None,
                {
                    "status": "parse_error",
                    "provider": result.get("provider_used"),
                    "model": result.get("model_used"),
                    "vision_input_used": bool(result.get("vision_input_used", False)),
                },
            )

    def _ensure_execution_plan(
        self,
        *,
        state,
        task_plan: TaskPlan,
        request: AgentRequest,
        analysis,
        observation: Image.Image,
        target_crop: Image.Image | None,
    ) -> TaskExecutionPlan:
        if state.execution_plan is not None:
            return state.execution_plan

        semantic_subgoals = [dict(subgoal) for subgoal in task_plan.subgoals]
        semantic_ids = [str(subgoal["id"]) for subgoal in semantic_subgoals]
        ordered_ids = list(semantic_ids)
        source = "semantic_fallback"
        task_summary = request.instruction
        failure_policy = (
            "After failed execution, refresh environment evidence and choose a "
            "different valid action; terminate only with an explicit reason."
        )
        vision_input_used = False

        if self.model_adapter.available():
            try:
                result = self.model_adapter.plan_task(
                    {
                        "instruction": request.instruction,
                        "observation_summary": analysis.scene_summary,
                        "task_contract": task_plan.to_dict(),
                        "layered_memories": state.layered_memories,
                        "environment_context": self._planner_environment_context(
                            request.environment_context
                        ),
                        "observation_image": self._model_observation_data_url(
                            observation
                        ),
                        "target_crop": (
                            image_to_data_url(target_crop)
                            if target_crop is not None
                            else None
                        ),
                        "require_vision": True,
                        "strict_vlm": self.strict_vlm,
                    }
                )
            except Exception:
                result = {"error": "task_planner_exception"}
            if self.strict_vlm and "error" in result:
                raise RuntimeError(
                    "strict VLM task planning failed: "
                    f"{result.get('error')}"
                )
            candidate_ids = result.get("ordered_subgoal_ids")
            if (
                "error" not in result
                and isinstance(candidate_ids, list)
                and len(candidate_ids) == len(semantic_ids)
                and len(set(candidate_ids)) == len(candidate_ids)
                and set(candidate_ids) == set(semantic_ids)
            ):
                ordered_ids = [str(item) for item in candidate_ids]
                source = "model_planner"
                task_summary = str(result.get("task_summary") or request.instruction)
                failure_policy = str(
                    result.get("failure_policy") or failure_policy
                )
                vision_input_used = bool(result.get("vision_input_used", False))

        if self.strict_vlm and (
            source != "model_planner"
            or not vision_input_used
        ):
            raise RuntimeError(
                "strict VLM mode requires a multimodal task plan; "
                f"received source={source}, vision_input_used={vision_input_used}"
            )

        subgoals_by_id = {
            str(subgoal["id"]): subgoal for subgoal in semantic_subgoals
        }
        subgoals = [
            ExecutionSubgoal(
                id=subgoal_id,
                description=str(subgoals_by_id[subgoal_id]["description"]),
                success_evidence=str(
                    subgoals_by_id[subgoal_id]["success_evidence"]
                ),
                status="in_progress" if index == 0 else "pending",
            )
            for index, subgoal_id in enumerate(ordered_ids)
        ]
        execution_plan = TaskExecutionPlan(
            plan_id=f"{state.session_id}:v1",
            instruction=request.instruction,
            task_summary=task_summary,
            task_types=list(task_plan.task_types),
            completion_mode=task_plan.completion_mode,
            subgoals=subgoals,
            current_subgoal_id=subgoals[0].id if subgoals else None,
            status="in_progress",
            source=source,
            failure_policy=failure_policy,
            limitations=list(task_plan.limitations),
            vision_input_used=vision_input_used,
            last_updated_step=request.step_id,
        )
        return self.memory.set_execution_plan(state, execution_plan)

    def _rule_fallback_planner(self, confidence: float, target_visible: bool, state) -> Action:
        """Rule-based fallback planner (renamed from _plan_action)."""
        if target_visible and confidence >= self.config.stop_confidence_threshold:
            return Action("STOP", {"reason": "target confidence crossed stop threshold"})
        if len(state.steps) >= self.config.max_steps - 1:
            return Action("ASK_CLARIFY", {"reason": "max steps reached without sufficient evidence"})
        recent_actions = [step.get("action", {}).get("type") for step in state.recent_steps(3)]
        if recent_actions.count("TURN_RIGHT") >= 2:
            return Action("MOVE_FORWARD", {"distance": 1})
        if confidence >= self.config.target_visible_threshold:
            return Action("INSPECT", {"reason": "candidate visible but not confirmed"})
        if len(state.steps) % 4 == 2:
            return Action("TURN_LEFT", {"angle": self.config.raw["agent"]["default_turn_angle_degrees"]})
        return Action("TURN_RIGHT", {"angle": self.config.raw["agent"]["default_turn_angle_degrees"]})

    def _model_observation_data_url(self, observation: Image.Image) -> str:
        target_size = self.config.image_size
        model_image = observation
        if observation.size != target_size:
            model_image = observation.resize(
                target_size,
                resample=Image.Resampling.LANCZOS,
            )
        return image_to_data_url(model_image)

    def _model_allowed_actions(self, task_plan: TaskPlan) -> list[str]:
        candidates = list(task_plan.action_candidates)
        if not self.strict_vlm:
            return candidates
        alias_to_canonical = {
            "Done": "STOP",
            "LookDown": "LOOK_DOWN",
            "LookUp": "LOOK_UP",
            "MoveAhead": "MOVE_FORWARD",
            "Pass": "INSPECT",
            "RotateLeft": "TURN_LEFT",
            "RotateRight": "TURN_RIGHT",
        }
        canonical_present = set(candidates)
        return [
            action
            for action in candidates
            if not (
                action in alias_to_canonical
                and alias_to_canonical[action] in canonical_present
            )
        ]

    @staticmethod
    def _has_successful_real_vlm_step(state) -> bool:
        for step in state.steps:
            model_info = step.get("model_info") or {}
            if str(model_info.get("status") or "") != "ok":
                continue
            if model_info.get("vision_input_used") is not True:
                continue
            provider = str(
                model_info.get("provider")
                or model_info.get("provider_used")
                or ""
            ).lower()
            model = str(
                model_info.get("model")
                or model_info.get("model_used")
                or ""
            ).lower()
            if not provider or not model:
                continue
            blocked_tokens = ("fake", "mock", "test")
            if any(token in provider for token in blocked_tokens):
                continue
            if any(token in model for token in blocked_tokens):
                continue
            return True
        return False

    @staticmethod
    def _verified_approach_action(
        *,
        task_plan: TaskPlan,
        completion_status: dict,
        environment_context: dict | None,
        state,
        allow_yield: bool = True,
    ) -> Action | None:
        if completion_status.get("approach_verified"):
            return None
        if allow_yield and EmbodiedSearchAgent._approach_guidance_should_yield(
            task_plan=task_plan,
            completion_status=completion_status,
            state=state,
        ):
            return None
        context = environment_context or {}
        approach = context.get("approach") or {}
        if EmbodiedSearchAgent._approach_guidance_is_stalled(
            approach=approach,
            state=state,
        ):
            return None
        if (
            not isinstance(approach, dict)
            or approach.get("source") != "ai2thor_interactable_pose"
            or approach.get("path_status")
            not in {"PathComplete", "PoseAlignment"}
        ):
            return None
        object_id = str(approach.get("objectId") or "")
        if (
            not object_id
            or object_id
            not in task_plan.matching_target_object_ids(context)
        ):
            return None
        if not EmbodiedSearchAgent._approach_target_matches_next_action(
            object_id=object_id,
            completion_status=completion_status,
            environment_context=context,
        ):
            return None
        payload = approach.get("recommended_action")
        if not isinstance(payload, dict):
            return None
        action_type = str(payload.get("type") or "")
        allowed_by_status = {
            "PathComplete": {
                "MOVE_FORWARD",
                "TURN_LEFT",
                "TURN_RIGHT",
            },
            "PoseAlignment": {
                "TURN_LEFT",
                "TURN_RIGHT",
                "LOOK_UP",
                "LOOK_DOWN",
            },
        }
        if (
            action_type not in task_plan.action_candidates
            or action_type
            not in allowed_by_status[str(approach["path_status"])]
        ):
            return None
        args = payload.get("args")
        parameter_by_action = {
            "MOVE_FORWARD": "distance",
            "TURN_LEFT": "angle",
            "TURN_RIGHT": "angle",
            "LOOK_UP": "angle",
            "LOOK_DOWN": "angle",
        }
        parameter = parameter_by_action[action_type]
        if not isinstance(args, dict) or set(args) != {parameter}:
            return None
        value = args.get(parameter)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            return None
        action = Action(
            action_type,
            {parameter: float(value)},
        )
        navigation_actions = {
            "MOVE_FORWARD",
            "MOVE_BACK",
            "MOVE_LEFT",
            "MOVE_RIGHT",
            "TURN_LEFT",
            "TURN_RIGHT",
            "LOOK_UP",
            "LOOK_DOWN",
        }
        for step in reversed(state.steps[-4:]):
            previous_action = (
                step.get("executed_action")
                or step.get("action")
                or {}
            )
            if (
                step.get("action_success") is True
                and previous_action.get("type") in navigation_actions
            ):
                break
            if (
                step.get("action_success") is False
                and previous_action.get("type") == action.type
                and previous_action.get("args") == action.args
            ):
                return None
        return action

    @staticmethod
    def _approach_guidance_should_yield(
        *,
        task_plan: TaskPlan,
        completion_status: dict,
        state,
    ) -> bool:
        missing_actions = list(completion_status.get("missing_actions") or [])
        if not any(action in missing_actions for action in ("PickupObject", "PutObject", "OpenObject")):
            return False
        recent_approach_steps = 0
        for step in reversed(state.steps[-4:]):
            if step.get("planner_source") != "simulator_oracle":
                break
            if step.get("fallback_reason") != "verified_approach_navigation":
                break
            previous_action = step.get("executed_action") or step.get("action") or {}
            if previous_action.get("type") not in {
                "MOVE_FORWARD",
                "TURN_LEFT",
                "TURN_RIGHT",
                "LOOK_UP",
                "LOOK_DOWN",
            }:
                break
            if step.get("action_success") is not True:
                break
            recent_approach_steps += 1
        if "PutObject" in missing_actions and "PickupObject" not in missing_actions:
            return recent_approach_steps >= 2
        return recent_approach_steps >= 3

    @staticmethod
    def _approach_guidance_is_stalled(*, approach: dict, state) -> bool:
        if not isinstance(approach, dict):
            return False
        payload = approach.get("recommended_action")
        if not isinstance(payload, dict):
            return False
        action_type = str(payload.get("type") or "")
        if action_type not in {"TURN_LEFT", "TURN_RIGHT", "LOOK_UP", "LOOK_DOWN"}:
            return False
        args = payload.get("args")
        if not isinstance(args, dict):
            return False
        value = args.get("angle")
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or abs(float(value)) > 5.0
        ):
            return False
        recent_alignment_steps = 0
        recent_directions: list[str] = []
        for step in reversed(state.steps[-6:]):
            if step.get("planner_source") != "simulator_oracle":
                break
            if step.get("fallback_reason") != "verified_approach_navigation":
                break
            previous_action = step.get("executed_action") or step.get("action") or {}
            previous_type = str(previous_action.get("type") or "")
            if previous_type not in {"TURN_LEFT", "TURN_RIGHT", "LOOK_UP", "LOOK_DOWN"}:
                break
            previous_args = previous_action.get("args") or {}
            previous_value = previous_args.get("angle")
            if (
                isinstance(previous_value, bool)
                or not isinstance(previous_value, (int, float))
                or not math.isfinite(float(previous_value))
                or abs(float(previous_value)) > 5.0
            ):
                break
            if step.get("action_success") is not True:
                break
            recent_alignment_steps += 1
            recent_directions.append(previous_type)
        return recent_alignment_steps >= 4 and len(set(recent_directions)) > 1

    @classmethod
    def _interaction_continuation_ready(
        cls,
        *,
        task_plan: TaskPlan,
        completion_status: dict,
        environment_context: dict | None,
    ) -> bool:
        missing_actions = list(completion_status.get("missing_actions") or [])
        if "PickupObject" in missing_actions:
            target = cls._select_context_object(
                task_plan=task_plan,
                environment_context=environment_context,
                require_flag="pickupable",
                prefer_visible=True,
            )
            return bool(target and target.get("visible") is True)
        if "PutObject" in missing_actions:
            held = cls._held_inventory_object(environment_context)
            receptacle = cls._select_context_object(
                task_plan=task_plan,
                environment_context=environment_context,
                require_flag="receptacle",
                prefer_visible=True,
                exclude_inventory=True,
            )
            return bool(
                held
                and receptacle
                and receptacle.get("visible") is True
            )
        if "OpenObject" in missing_actions:
            target = cls._select_context_object(
                task_plan=task_plan,
                environment_context=environment_context,
                require_flag="openable",
                exclude_open=True,
                prefer_visible=True,
            )
            return bool(target and target.get("visible") is True)
        return True

    @staticmethod
    def _approach_target_matches_next_action(
        *,
        object_id: str,
        completion_status: dict,
        environment_context: dict,
    ) -> bool:
        missing_actions = list(completion_status.get("missing_actions") or [])
        if not missing_actions:
            return True
        target = next(
            (
                item
                for item in environment_context.get("objects", [])
                if str(item.get("objectId") or "") == object_id
            ),
            None,
        )
        if not isinstance(target, dict):
            return False
        if "PickupObject" in missing_actions:
            return bool(target.get("pickupable"))
        if "PutObject" in missing_actions:
            return bool(target.get("receptacle"))
        if "OpenObject" in missing_actions:
            return bool(target.get("openable"))
        return True

    @staticmethod
    def _model_environment_context(
        environment_context: dict | None,
    ) -> dict:
        context = dict(environment_context or {})
        approach = context.get("approach")
        if isinstance(approach, dict):
            sanitized_approach = dict(approach)
            for field in (
                "matched_pose",
                "target_pose",
                "recommended_action",
            ):
                sanitized_approach.pop(field, None)
            context["approach"] = sanitized_approach
        return context

    def _planner_environment_context(
        self,
        environment_context: dict | None,
    ) -> dict:
        context = self._model_environment_context(environment_context)
        if not self.strict_vlm:
            return context
        objects = context.get("objects")
        if isinstance(objects, list):
            context["objects"] = [
                item
                for item in objects
                if isinstance(item, dict) and bool(item.get("visible"))
            ]
        return context

    def _continue_approximate_sit(
        self,
        *,
        completion_status: dict,
        confidence: float,
        target_visible: bool,
        state,
    ) -> Action:
        if not completion_status.get("approach_verified"):
            if target_visible:
                return Action(
                    "MOVE_FORWARD",
                    {"distance": 1, "reason": "approach target before crouching"},
                )
            return self._rule_fallback_planner(confidence, target_visible, state)
        if "Crouch" in completion_status.get("missing_actions", []):
            return Action(
                "Crouch",
                {"reason": "execute the documented sit approximation near the target"},
            )
        return Action(
            "INSPECT",
            {"reason": "refresh simulator evidence for the crouched posture"},
        )

    def _continue_supported_task(
        self,
        *,
        task_plan: TaskPlan,
        completion_status: dict,
        confidence: float,
        target_visible: bool,
        environment_context: dict | None,
        state,
    ) -> Action:
        missing_actions = list(completion_status.get("missing_actions") or [])
        if "OpenObject" in missing_actions:
            target = self._select_context_object(
                task_plan=task_plan,
                environment_context=environment_context,
                require_flag="openable",
                exclude_open=True,
                prefer_visible=True,
            )
            object_type = self._object_type_for_action(target, default="Door")
            object_id = self._object_id_for_action(target)
            args = {
                "objectType": object_type,
                "reason": "open the requested doorway before crossing the threshold",
            }
            if object_id:
                args["objectId"] = object_id
            return Action(
                "OpenObject",
                args,
            )
        if "PickupObject" in missing_actions:
            target = self._select_context_object(
                task_plan=task_plan,
                environment_context=environment_context,
                require_flag="pickupable",
                prefer_visible=True,
            )
            object_type = self._object_type_for_action(target, default="object")
            object_id = self._object_id_for_action(target)
            args = {
                "objectType": object_type,
                "reason": "pickup the required object before placement",
            }
            if object_id:
                args["objectId"] = object_id
            return Action(
                "PickupObject",
                args,
            )
        if "PutObject" in missing_actions:
            held = self._held_inventory_object(environment_context)
            receptacle = self._select_context_object(
                task_plan=task_plan,
                environment_context=environment_context,
                require_flag="receptacle",
                prefer_visible=True,
                exclude_inventory=True,
            )
            held_type = self._object_type_for_action(held, default="held object")
            held_id = self._object_id_for_action(held)
            receptacle_type = self._object_type_for_action(receptacle, default="receptacle")
            receptacle_id = self._object_id_for_action(receptacle)
            args = {
                "object": held_type,
                "heldObjectType": held_type,
                "receptacleType": receptacle_type,
                "reason": "place the held object into the requested receptacle",
            }
            if held_id:
                args["heldObjectId"] = held_id
            if receptacle_id:
                args["receptacleObjectId"] = receptacle_id
                args["objectId"] = receptacle_id
            return Action(
                "PutObject",
                args,
            )
        if "exit_room" in task_plan.task_types:
            return Action(
                "MOVE_FORWARD",
                {"distance": 1, "reason": "cross the verified doorway threshold"},
            )
        if "navigate_to" in task_plan.task_types:
            if target_visible:
                return Action(
                    "MOVE_FORWARD",
                    {"distance": 1, "reason": "approach target before completion"},
                )
            return self._rule_fallback_planner(confidence, target_visible, state)
        return Action(
            "INSPECT",
            {"reason": "refresh simulator evidence before termination"},
        )

    @staticmethod
    def _held_inventory_object(
        environment_context: dict | None,
    ) -> dict | None:
        inventory = (environment_context or {}).get("inventoryObjects") or []
        if not isinstance(inventory, list) or not inventory:
            return None
        first = inventory[0]
        return first if isinstance(first, dict) else None

    @staticmethod
    def _object_id_for_action(item: dict | None) -> str:
        if not isinstance(item, dict):
            return ""
        return str(item.get("objectId") or item.get("name") or "").strip()

    @staticmethod
    def _object_type_for_action(item: dict | None, *, default: str) -> str:
        if not isinstance(item, dict):
            return default
        return str(item.get("objectType") or item.get("name") or default).strip() or default

    @staticmethod
    def _object_distance_sort_key(item: dict) -> float:
        try:
            distance = float(item.get("distance"))
        except (TypeError, ValueError):
            return float("inf")
        return distance if math.isfinite(distance) and distance >= 0.0 else float("inf")

    @staticmethod
    def _instruction_mentions_object_type(
        task_plan: TaskPlan,
        object_type: str,
    ) -> bool:
        normalized_instruction = task_plan.instruction.lower()
        normalized_type = "".join(
            ch for ch in object_type.lower() if ch.isalnum()
        )
        if not normalized_type:
            return False
        if object_type.lower() in normalized_instruction:
            return True
        aliases = {
            "cardboardbox": ("cardboard box", "box", "纸箱", "箱子", "盒子"),
            "box": ("box", "纸箱", "箱子", "盒子"),
            "vase": ("vase", "花瓶"),
            "mug": ("mug", "马克杯", "杯子"),
            "cup": ("cup", "杯子"),
            "bowl": ("bowl", "碗"),
            "door": ("door", "门", "房门"),
            "sofa": ("sofa", "couch", "沙发"),
            "television": ("television", "tv", "电视"),
        }
        return any(
            alias.lower() in normalized_instruction
            for alias in aliases.get(normalized_type, (normalized_type,))
        )

    @classmethod
    def _select_context_object(
        cls,
        *,
        task_plan: TaskPlan,
        environment_context: dict | None,
        require_flag: str,
        prefer_visible: bool,
        exclude_open: bool = False,
        exclude_inventory: bool = False,
    ) -> dict | None:
        context = environment_context or {}
        inventory_ids = {
            str(item.get("objectId") or "")
            for item in context.get("inventoryObjects", []) or []
            if isinstance(item, dict)
        }
        candidates = []
        for item in context.get("objects", []) or []:
            if not isinstance(item, dict):
                continue
            if require_flag and item.get(require_flag) is not True:
                continue
            if exclude_open and item.get("isOpen") is True:
                continue
            if exclude_inventory and str(item.get("objectId") or "") in inventory_ids:
                continue
            candidates.append(item)
        if not candidates:
            return None

        matching_ids = task_plan.matching_target_object_ids(context)

        def score(item: dict) -> tuple[int, int, int, float, str]:
            object_id = str(item.get("objectId") or "")
            object_type = str(item.get("objectType") or item.get("name") or "")
            return (
                0 if object_id in matching_ids else 1,
                0 if cls._instruction_mentions_object_type(task_plan, object_type) else 1,
                0 if (not prefer_visible or item.get("visible") is True) else 1,
                cls._object_distance_sort_key(item),
                object_id or object_type,
            )

        return sorted(candidates, key=score)[0]

    def _build_thought(
        self,
        scene_summary: str,
        action: Action,
        confidence: float,
        hints: list[str],
        done: bool,
        *,
        completion_status: dict[str, object],
    ) -> str:
        hint_text = "; ".join(hints) if hints else "暂无相关记忆"
        if action.type == "ASK_CLARIFY":
            reason = str(
                action.args.get("reason")
                or completion_status.get("reason")
                or "任务需要进一步澄清"
            )
            return f"{scene_summary} 任务尚未完成：{reason}"
        if done and completion_status.get("complete"):
            return f"{scene_summary} 置信度 {confidence:.2f} 已足够高，智能体停止并报告目标。"
        if done:
            reason = str(completion_status.get("reason") or "完成条件尚未验证")
            return f"{scene_summary} 本轮已终止，但任务未验证完成：{reason}"
        if (
            completion_status.get("completion_mode") == "approximate_sit"
            and not completion_status.get("complete")
        ):
            reason = str(completion_status.get("reason") or "近似坐下条件尚未验证")
            return (
                f"{scene_summary} 任务尚未完成：{reason}。"
                f"下一步动作是 {action.type}。"
            )
        return f"{scene_summary} 检索提示: {hint_text}。下一步动作是 {action.type}，因为置信度为 {confidence:.2f}。"

    def _build_structured_thought(
        self,
        analysis,
        action: Action,
        confidence: float,
        hints: list[str],
        done: bool,
        *,
        completion_status: dict[str, object],
    ) -> dict[str, str]:
        """构建结构化的中文思考输出"""
        # 视觉观察
        best = analysis.best_candidate
        if best:
            observation = f"当前画面{best.region}有一个{best.color_name}的区域，可能是目标物体。置信度：{confidence:.2f}"
        else:
            observation = "当前画面中未检测到明显的目标物体特征，需要继续探索。"

        # 推理过程
        if action.type == "ASK_CLARIFY":
            reason = str(
                action.args.get("reason")
                or completion_status.get("reason")
                or "任务需要进一步澄清"
            )
            reasoning = f"任务未完成，当前仿真能力无法验证所请求的行为：{reason}"
        elif done and completion_status.get("complete"):
            reasoning = f"目标已确认！置信度 {confidence:.2f} 超过阈值 {self.config.stop_confidence_threshold:.2f}，可以停止搜索。"
        elif done:
            reasoning = f"本轮已终止，但完成条件未通过验证：{completion_status.get('reason', '未提供原因')}"
        elif (
            completion_status.get("completion_mode") == "approximate_sit"
            and not completion_status.get("complete")
        ):
            reasoning = (
                "任务未完成，必须继续执行并验证近似坐下子目标："
                f"{completion_status.get('reason', '完成条件尚未满足')}"
            )
        elif confidence >= self.config.target_visible_threshold:
            reasoning = f"发现疑似目标，但置信度 {confidence:.2f} 还不够高，需要更近距离观察或换个角度。"
        else:
            reasoning = f"当前区域置信度较低 ({confidence:.2f})，建议继续探索其他区域。"
            if hints:
                reasoning += f" 记忆提示：{hints[0]}"

        # 动作映射
        action_map = {
            "MOVE_FORWARD": "向前移动",
            "TURN_LEFT": "向左转",
            "TURN_RIGHT": "向右转",
            "LOOK_UP": "向上看",
            "LOOK_DOWN": "向下看",
            "Crouch": "蹲下",
            "Stand": "站起",
            "INSPECT": "仔细检查",
            "STOP": "停止",
            "ASK_CLARIFY": "请求澄清"
        }
        action_text = action_map.get(action.type, action.type)

        return {
            "observation": observation,
            "reasoning": reasoning,
            "action": action_text,
            "confidence": f"{confidence:.3f}"
        }

    @staticmethod
    def _build_decision_trace(
        *,
        planner_source: str,
        model_info: dict[str, object],
        action: Action,
        completion_status: dict[str, object],
    ) -> str:
        """Build a user-visible decision audit without exposing hidden reasoning."""
        lines = [
            f"planner_source={planner_source}",
            f"vision_input_used={bool(model_info.get('vision_input_used', False))}",
        ]
        provider = model_info.get("provider") or model_info.get("provider_used")
        model = model_info.get("model") or model_info.get("model_used")
        if provider or model:
            lines.append(f"model={provider or '-'}:{model or '-'}")
        task_progress = model_info.get("task_progress")
        if task_progress:
            lines.append(f"task_progress={task_progress}")
        if model_info.get("thought_summary"):
            lines.append("model_summary_present=True")
        lines.extend(
            [
                f"selected_action={action.type}",
                f"action_args={action.args}",
                f"completion_complete={bool(completion_status.get('complete', False))}",
                f"completion_reason={completion_status.get('reason', '-')}",
            ]
        )
        return "\n".join(str(line) for line in lines)
