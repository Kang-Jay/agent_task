from __future__ import annotations

from PIL import Image

from src.memory.session_memory import SessionMemory
from src.rag.retriever import HintRetriever
from src.task.config import AgentConfig, load_config
from src.types.schema import Action, AgentRequest, AgentResponse
from src.vision.heuristic_vision import HeuristicVision
from src.vision.image_io import crop_from_point, load_image_from_any


class EmbodiedSearchAgent:
    def __init__(self, config: AgentConfig | None = None):
        self.config = config or load_config()
        self.vision = HeuristicVision(self.config)
        self.memory = SessionMemory(self.config)
        self.retriever = HintRetriever(self.config)

    def reset(self, session_id: str) -> dict[str, str]:
        self.memory.reset(session_id)
        return {"status": "reset", "session_id": session_id}

    def step(self, request: AgentRequest) -> AgentResponse:
        self._validate_request(request)
        observation = load_image_from_any(request.observation_image, root=self.config.path.parent.parent)
        target_crop = self._resolve_target_crop(observation, request)
        state = self.memory.get_or_create(request.session_id, request.instruction)
        analysis = self.vision.analyze(observation, request.instruction, target_crop)
        hints = self.retriever.retrieve(request.instruction, state)
        action = self._plan_action(analysis.best_candidate.confidence if analysis.best_candidate else 0.0, analysis.target_visible, state)
        confidence = analysis.best_candidate.confidence if analysis.best_candidate else 0.0
        done = action.type in self.config.terminal_actions
        thought = self._build_thought(analysis.scene_summary, action, confidence, hints, done)
        step_record = {
            "step_id": request.step_id,
            "thought": thought,
            "action": action.to_dict(),
            "confidence": confidence,
            "done": done,
            "best_candidate": analysis.best_candidate.to_dict() if analysis.best_candidate else None,
            "candidate_count": len(analysis.candidates),
            "retrieved_hints": hints,
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
            search_map=self.memory.search_map(state),
            confidence_trace=self.memory.confidence_trace(state),
            target_binding={
                "language": bool(request.instruction.strip()),
                "clicked_point": request.clicked_point,
                "target_crop": bool(request.target_crop or request.clicked_point),
                "mode": "multimodal" if request.target_crop or request.clicked_point else "language_only",
            },
        )

    def audit(self) -> dict[str, object]:
        return {
            "config_path": str(self.config.path),
            "pipeline": self.config.raw["pipeline"]["stages"],
            "allowed_actions": self.config.allowed_actions,
            "max_steps": self.config.max_steps,
            "stop_confidence_threshold": self.config.stop_confidence_threshold,
            "image_size": list(self.config.image_size),
            "status": "ok",
        }

    def export_trace(self, session_id: str) -> dict[str, object]:
        return self.memory.export_trace(session_id)

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

    def _plan_action(self, confidence: float, target_visible: bool, state) -> Action:
        if target_visible and confidence >= self.config.stop_confidence_threshold:
            return self._validated_action(Action("STOP", {"reason": "target confidence crossed stop threshold"}))
        if len(state.steps) >= self.config.max_steps - 1:
            return self._validated_action(Action("ASK_CLARIFY", {"reason": "max steps reached without sufficient evidence"}))
        recent_actions = [step.get("action", {}).get("type") for step in state.recent_steps(3)]
        if recent_actions.count("TURN_RIGHT") >= 2:
            return self._validated_action(Action("MOVE_FORWARD", {"distance": 1}))
        if confidence >= self.config.target_visible_threshold:
            return self._validated_action(Action("INSPECT", {"reason": "candidate visible but not confirmed"}))
        if len(state.steps) % 4 == 2:
            return self._validated_action(Action("TURN_LEFT", {"angle": self.config.raw["agent"]["default_turn_angle_degrees"]}))
        return self._validated_action(Action("TURN_RIGHT", {"angle": self.config.raw["agent"]["default_turn_angle_degrees"]}))

    def _validated_action(self, action: Action) -> Action:
        if action.type not in self.config.allowed_actions:
            return Action("ASK_CLARIFY", {"reason": f"illegal action blocked: {action.type}"})
        return action

    def _build_thought(self, scene_summary: str, action: Action, confidence: float, hints: list[str], done: bool) -> str:
        hint_text = "; ".join(hints) if hints else "no retrieved prior is needed"
        if done:
            return f"{scene_summary} Confidence {confidence:.2f} is high enough, so the agent stops and reports the target."
        return f"{scene_summary} Retrieved hint: {hint_text}. Next action is {action.type} because confidence is {confidence:.2f}."
