from __future__ import annotations

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
    ):
        self.config = config or load_config()
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
        recalled_memories = self.memory.retrieve_relevant(state)
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

        # Try model planner first
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
        confidence = (
            min(vision_confidence, planner_confidence)
            if planner_source == "model_planner" and planner_confidence is not None
            else vision_confidence
        )

        # Validate action
        if not task_plan.supported:
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
        done = action.type in self.config.terminal_actions or action.type == "Done"
        premature_sit_action = (
            task_plan.completion_mode == "approximate_sit"
            and action.type == "Crouch"
            and not completion_status.get("approach_verified")
        )
        if (
            action.type in {"STOP", "Done"}
            and not completion_status["complete"]
        ) or premature_sit_action:
            if task_plan.is_visual_search:
                action = self._rule_fallback_planner(confidence, analysis.target_visible, state)
                skill_call = SkillCall(
                    name=action.type,
                    args=action.args,
                    preconditions=[],
                    expected_observation=f"Execute {action.type}",
                )
                fallback_reason = "stop_confidence_too_low"
            elif task_plan.completion_mode == "approximate_sit":
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
            else:
                action = Action(
                    "ASK_CLARIFY",
                    {"reason": completion_status["reason"]},
                )
                skill_call = SkillCall(
                    name="ASK_CLARIFY",
                    args=action.args,
                    preconditions=[],
                    expected_observation="task completion can be verified",
                )
                fallback_reason = "task_completion_not_verified"
            planner_source = "rule_fallback"
            model_info["decision_status"] = "rejected_premature_termination"
            done = action.type in self.config.terminal_actions or action.type == "Done"

        model_summary = str(model_info.get("thought_summary") or "").strip()
        thought = (
            model_summary
            if planner_source == "model_planner" and model_summary
            else self._build_thought(
                analysis.scene_summary,
                action,
                confidence,
                hints,
                done,
                completion_status=completion_status,
            )
        )
        structured_thought = self._build_structured_thought(
            analysis,
            action,
            confidence,
            hints,
            done,
            completion_status=completion_status,
        )
        if planner_source == "model_planner" and model_summary:
            structured_thought["reasoning"] = model_summary

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
        action_type = str(
            (response.get("action") or {}).get("type") or ""
        )
        done = bool(verification["complete"]) or action_type in {
            "STOP",
            "Done",
            "ASK_CLARIFY",
        }
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
        """Try to plan with model, fallback to rules if needed.

        Returns:
            action, skill_call, planner_source, fallback_reason,
            planner_confidence, model_info
        """
        if not self.model_adapter.available():
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
            "allowed_actions": list(task_plan.action_candidates),
            "action_specs": list(task_plan.action_specs),
            "terminal_actions": ["STOP", "Done", "ASK_CLARIFY"],
            "task_plan": task_plan.to_dict(),
            "execution_plan": execution_plan.to_dict(),
            "environment_context": request.environment_context or {},
            "completion_status": completion_status,
            "current_step": request.step_id,
            "max_steps": self.config.max_steps,
            "observation_image": image_to_data_url(observation),
            "target_crop": image_to_data_url(target_crop) if target_crop is not None else None,
            "require_vision": True,
        }

        result = self.model_adapter.plan_action(payload)

        # Check if model call failed
        if "error" in result:
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
                },
            )

        except Exception as e:
            # Model output parsing failed
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
                        "environment_context": request.environment_context or {},
                        "observation_image": image_to_data_url(observation),
                        "target_crop": (
                            image_to_data_url(target_crop)
                            if target_crop is not None
                            else None
                        ),
                        "require_vision": True,
                    }
                )
            except Exception:
                result = {"error": "task_planner_exception"}
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
