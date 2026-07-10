from __future__ import annotations

import json
import math
import os
import platform
import re
import shutil
import subprocess
import threading
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageDraw

from src.agent.controller import EmbodiedSearchAgent
from src.simulation.ai2thor_actions import AI2ThorActionCatalog, AI2ThorActionExecutor
from src.simulation.ai2thor_approach import AI2ThorApproachVerifier
from src.simulation.ai2thor_interactions import (
    AI2ThorInteractionResolver,
    OBJECT_ID_ACTIONS,
)
from src.simulation.ai2thor_postconditions import AI2ThorPostconditionVerifier
from src.simulation.ai2thor_runtime import (
    DEFAULT_GRID_SIZE_METERS,
    create_controller_safely,
    should_snap_to_grid,
)
from src.simulation.object_closeup import render_closeup, resolve_clicked_object
from src.simulation.room_simulator import DemoResult, DemoStep
from src.simulation.stream_protocol import StreamCancelled, StreamEventEmitter
from src.simulation.video_encoding import write_browser_compatible_mp4
from src.task.config import ROOT, load_config
from src.types.schema import AgentRequest
from src.vision.image_io import image_to_data_url


AI2THOR_OUTPUT_DIR = ROOT / "docs" / "ai2thor_outputs"


TARGET_ALIASES: dict[str, list[str]] = {
    "pillow": ["pillow", "枕头"],
    "sofa": ["sofa", "couch", "沙发"],
    "television": ["television", "tv", "电视", "电视机"],
    "floorlamp": ["floorlamp", "floor lamp", "lamp", "灯", "落地灯"],
    "remotecontrol": ["remotecontrol", "remote control", "remote", "遥控器"],
    "sidetable": ["sidetable", "side table"],
    "diningtable": ["diningtable", "dining table", "table", "桌子"],
    "coffeetable": ["coffeetable", "coffee table", "table", "桌子"],
    "armchair": ["armchair", "arm chair", "chair", "椅子"],
    "garbagecan": ["garbagecan", "garbage can", "trash can", "bin", "垃圾桶"],
}

STRUCTURAL_OBJECTS = {"floor", "wall", "ceiling", "window", "painting"}


@dataclass(frozen=True)
class SimulatorStatus:
    available: bool
    backend: str
    scene: str
    message: str
    diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "backend": self.backend,
            "scene": self.scene,
            "message": self.message,
            "diagnostics": self.diagnostics,
        }


class AI2ThorVisualSearchDemo:
    """Runs the embodied visual-search loop in AI2-THOR when the runtime is available."""

    def __init__(
        self,
        scene: str = "FloorPlan211",
        agent: EmbodiedSearchAgent | None = None,
        agent_mode: str = "default",
    ):
        self.config = load_config()
        self.agent = agent or EmbodiedSearchAgent(self.config)
        self.scene = scene
        self.agent_mode = agent_mode
        self.action_catalog = AI2ThorActionCatalog()
        if agent_mode not in self.action_catalog.summary()["mode_controllers"]:
            raise ValueError(f"Unsupported AI2-THOR agent mode: {agent_mode}")
        self.action_executor = AI2ThorActionExecutor(self.action_catalog)
        self.approach_verifier = AI2ThorApproachVerifier(
            self.action_executor
        )
        self.interaction_resolver = AI2ThorInteractionResolver()
        self.postconditions = AI2ThorPostconditionVerifier()

    @staticmethod
    def status(scene: str = "FloorPlan211") -> SimulatorStatus:
        diagnostics: dict[str, Any] = {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "ai2thor_import": False,
            "vulkaninfo": shutil.which("vulkaninfo"),
            "nvidia_smi": shutil.which("nvidia-smi"),
            "wsl": bool(os.environ.get("WSL_DISTRO_NAME")),
        }
        try:
            import ai2thor  # type: ignore
            from ai2thor import build  # type: ignore

            diagnostics["ai2thor_import"] = True
            diagnostics["ai2thor_version"] = getattr(ai2thor, "__version__", "unknown")
            diagnostics["ai2thor_build_commit"] = getattr(build, "COMMIT_ID", "unknown")
            try:
                diagnostics["catalog_match"] = AI2ThorActionCatalog().verify_runtime(
                    ai2thor_version=str(diagnostics["ai2thor_version"]),
                    build_commit=str(diagnostics["ai2thor_build_commit"]),
                )
            except RuntimeError as exc:
                diagnostics["catalog_error"] = str(exc)
        except Exception as exc:
            diagnostics["ai2thor_error"] = repr(exc)
            return SimulatorStatus(
                available=False,
                backend="ai2thor",
                scene=scene,
                message="AI2-THOR is not importable in this Python environment.",
                diagnostics=diagnostics,
            )

        if platform.system().lower() == "windows":
            return SimulatorStatus(
                available=False,
                backend="ai2thor",
                scene=scene,
                message=(
                    "AI2-THOR PyPI builds do not provide a working native Windows Unity build here; "
                    "run through WSL2/Linux with Vulkan configured."
                ),
                diagnostics=diagnostics,
            )

        return SimulatorStatus(
            available=True,
            backend="ai2thor",
            scene=scene,
            message="AI2-THOR import succeeded; runtime launch will be tested when the demo starts.",
            diagnostics=diagnostics,
        )

    def run_demo(
        self,
        instruction: str,
        max_steps: int = 8,
        clicked_point: list[int] | None = None,
        clicked_object_id: str | None = None,
        session_id: str = "ai2thor-demo",
        episode_id: str | None = None,
        emit: Callable[[dict[str, Any]], None] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> DemoResult:
        from ai2thor.controller import Controller  # type: ignore
        from ai2thor.platform import CloudRendering  # type: ignore

        self.action_catalog.verify_installed_runtime()
        max_steps = min(int(max_steps), self.config.max_steps)
        safe_session_id = re.sub(r"[^A-Za-z0-9._-]", "_", session_id)
        episode_id = episode_id or uuid.uuid4().hex
        emitter = StreamEventEmitter(episode_id, emit)
        run_output_dir = AI2THOR_OUTPUT_DIR / safe_session_id / episode_id
        frame_dir = run_output_dir / "frames"
        frame_dir.mkdir(parents=True, exist_ok=False)

        controller = None
        result = DemoResult(
            episode_id=episode_id,
            output_dir=str(run_output_dir.relative_to(ROOT)),
        )
        self.agent.reset(session_id)
        task_plan = self.agent.task_semantics.analyze(
            instruction,
            mode=self.agent_mode,
            legacy_actions=self.config.allowed_actions,
        )
        emitter.emit(
            "task_parsed",
            session_id=session_id,
            instruction=instruction,
            scene=self.scene,
            agent_mode=self.agent_mode,
            max_steps=max_steps,
            task_plan=task_plan.to_dict(),
        )
        try:
            StreamEventEmitter.raise_if_cancelled(cancel_event)
            emitter.emit(
                "simulator_starting",
                scene=self.scene,
                agent_mode=self.agent_mode,
            )
            rotate_step_degrees = self.config.raw["agent"][
                "default_turn_angle_degrees"
            ]
            controller = create_controller_safely(
                Controller,
                scene=self.scene,
                platform=CloudRendering,
                agentMode=self.agent_mode,
                width=960,
                height=540,
                quality="Low",
                gridSize=DEFAULT_GRID_SIZE_METERS,
                rotateStepDegrees=rotate_step_degrees,
                snapToGrid=should_snap_to_grid(
                    mode=self.agent_mode,
                    rotate_step_degrees=rotate_step_degrees,
                ),
                renderInstanceSegmentation=True,
            )
            event = controller.last_event
            emitter.emit(
                "simulator_ready",
                scene=self.scene,
                agent_mode=self.agent_mode,
                robot=self._robot_state(event.metadata),
            )
            event, map_camera_id, map_camera_properties = (
                self._initialize_map_camera(
                    controller=controller,
                    event=event,
                    emitter=emitter,
                )
            )
            reachable_execution = self.action_executor.execute(
                controller,
                mode=self.agent_mode,
                action="GetReachablePositions",
                actor="manual",
            )
            reachable_event = reachable_execution.event
            reachable_positions = reachable_event.metadata.get("actionReturn") or []
            event = reachable_event
            agent_path: list[dict[str, float]] = []
            confirmed_target_steps = 0
            bound_target_object_id: str | None = None
            for step_id in range(max_steps):
                StreamEventEmitter.raise_if_cancelled(cancel_event)
                grounded_target = self._ground_target_from_segmentation(
                    event,
                    instruction,
                )
                if grounded_target:
                    bound_target_object_id = str(
                        grounded_target["object_id"]
                    )
                environment_context = self.interaction_resolver.build_context(
                    event.metadata
                )
                if (
                    "navigate_to" in task_plan.task_types
                    and bound_target_object_id
                ):
                    environment_context["approach"] = (
                        self.approach_verifier.verify(
                            controller,
                            mode=self.agent_mode,
                            metadata=event.metadata,
                            object_id=bound_target_object_id,
                        ).to_context()
                    )
                obs = Image.fromarray(event.frame).convert("RGB")
                obs_path = frame_dir / f"ai2thor_obs_{step_id:02d}.png"
                obs.save(obs_path)
                emitter.emit(
                    "observation_ready",
                    step_id=step_id,
                    observation_path=str(obs_path.relative_to(ROOT)),
                    robot=self._robot_state(event.metadata),
                    visible_objects=self._visible_objects(event.metadata)[:20],
                )

                # 在第一步传递 clicked_point 或 clicked_object，并尝试渲染特写
                StreamEventEmitter.raise_if_cancelled(cancel_event)

                # Prepare click-based target reference (step 0 only)
                target_crop_url = None
                resolved_object_id = None
                clicked_binding = None
                if step_id == 0 and (clicked_point or clicked_object_id):
                    target_crop_url, resolved_object_id, clicked_binding = (
                        self._prepare_click_target(
                            controller,
                            event,
                            clicked_point=clicked_point,
                            clicked_object_id=clicked_object_id,
                        )
                    )
                    if clicked_binding:
                        emitter.emit(
                            "closeup_ready",
                            step_id=step_id,
                            object_id=clicked_binding["object_id"],
                            object_type=clicked_binding["object_type"],
                            affordances=clicked_binding["affordances"],
                            closeup_source=clicked_binding["closeup_source"],
                            world_position=clicked_binding["world_position"],
                        )

                emitter.emit(
                    "model_request_started",
                    step_id=step_id,
                    planner=self.agent.model_adapter.audit(),
                )
                response = self.agent.step(
                    AgentRequest(
                        session_id=session_id,
                        instruction=instruction,
                        observation_image=image_to_data_url(obs),
                        step_id=step_id,
                        clicked_point=clicked_point if step_id == 0 else None,
                        clicked_object_id=resolved_object_id if step_id == 0 else None,
                        target_crop=target_crop_url if step_id == 0 else None,
                        agent_mode=self.agent_mode,
                        environment_context=environment_context,
                    )
                )
                response_dict = response.to_dict()
                visual_search_task = self._should_use_visual_search_oracle(
                    response_dict.get("task_plan"),
                    instruction,
                )
                emitter.emit(
                    "model_decision",
                    step_id=step_id,
                    thought=response_dict.get("thought", ""),
                    structured_thought=response_dict.get("structured_thought"),
                    proposed_action=response_dict.get("action"),
                    confidence=response_dict.get("confidence", 0.0),
                    planner_source=response_dict.get("planner_source"),
                    model_info=response_dict.get("model_info"),
                    task_plan=response_dict.get("task_plan"),
                    completion_status=response_dict.get("completion_status"),
                )
                if grounded_target and visual_search_task:
                    if grounded_target["confidence"] >= self.config.stop_confidence_threshold:
                        confirmed_target_steps += 1
                    else:
                        confirmed_target_steps = 0
                    action_type = "STOP" if confirmed_target_steps >= 2 else "INSPECT"
                    self._apply_grounded_target(response_dict, grounded_target, action_type, confirmed_target_steps)
                else:
                    confirmed_target_steps = 0
                    action_type = response_dict["action"]["type"]
                    if self._should_force_search(
                        visual_search_task=visual_search_task,
                        action_type=action_type,
                    ):
                        action_type = self._search_action(step_id)
                        self._apply_search_response(response_dict, action_type)
                interaction_binding: dict[str, Any] | None = None
                if action_type in OBJECT_ID_ACTIONS:
                    binding = self.interaction_resolver.resolve(
                        action=action_type,
                        args=response_dict["action"].get("args", {}),
                        instruction=instruction,
                        metadata=event.metadata,
                    )
                    interaction_binding = binding.to_dict()
                    if binding.valid:
                        response_dict["action"]["args"] = binding.args
                        if response_dict.get("skill_call"):
                            response_dict["skill_call"]["args"] = binding.args
                    else:
                        failed_action = action_type
                        action_type = self._search_action(step_id)
                        reason = "; ".join(binding.errors)
                        response_dict["action"] = {
                            "type": action_type,
                            "args": {
                                "reason": (
                                    f"{failed_action} preconditions not met: {reason}"
                                )
                            },
                        }
                        response_dict["done"] = False
                        response_dict["planner_source"] = "rule_fallback"
                        response_dict["fallback_reason"] = "interaction_binding_failed"
                        response_dict["thought"] = (
                            f"Cannot safely execute {failed_action}: {reason}. "
                            f"The agent performs {action_type} to gather a better observation."
                        )
                        response_dict["skill_call"] = {
                            "name": action_type,
                            "args": response_dict["action"]["args"],
                            "preconditions": [
                                "interaction target must be visible and affordance-valid"
                            ],
                            "expected_observation": (
                                "a new view with a uniquely groundable interaction target"
                            ),
                        }
                response_dict["interaction_binding"] = interaction_binding
                done = action_type in {"STOP", "Done", "ASK_CLARIFY"}
                response_dict["done"] = done
                action_args = {
                    key: value
                    for key, value in response_dict["action"].get("args", {}).items()
                    if key != "reason"
                }
                emitter.emit(
                    "action_validated",
                    step_id=step_id,
                    action=action_type,
                    args=action_args,
                    done=done,
                    planner_source=response_dict.get("planner_source"),
                    grounded_target=grounded_target,
                    interaction_binding=interaction_binding,
                )
                robot_before = self._robot_state(event.metadata)
                agent_path.append(robot_before)
                next_event = event
                action_success = action_type != "ASK_CLARIFY"
                execution_record: dict[str, Any] | None = None
                if not done:
                    StreamEventEmitter.raise_if_cancelled(cancel_event)
                    try:
                        execution = self.action_executor.execute(
                            controller,
                            mode=self.agent_mode,
                            action=action_type,
                            args=action_args,
                            actor="agent",
                        )
                        next_event = execution.event
                        postcondition = self.postconditions.verify(
                            action=execution.action,
                            args=execution.args,
                            before=event.metadata,
                            after=next_event.metadata,
                            runtime_success=execution.success,
                        )
                        action_success = execution.success and postcondition.passed
                        execution_record = execution.to_dict()
                        execution_record["postcondition"] = postcondition.to_dict()
                    except ValueError as exc:
                        action_success = False
                        execution_record = {
                            "action": action_type,
                            "mode": self.agent_mode,
                            "args": action_args,
                            "success": False,
                            "error_message": str(exc),
                        }
                emitter.emit(
                    "action_executed",
                    step_id=step_id,
                    action=action_type,
                    action_success=action_success,
                    execution=execution_record,
                )
                post_grounded_target = (
                    self._ground_target_from_segmentation(
                        next_event,
                        instruction,
                    )
                )
                if post_grounded_target:
                    bound_target_object_id = str(
                        post_grounded_target["object_id"]
                    )
                post_environment_context = (
                    self.interaction_resolver.build_context(
                        next_event.metadata
                    )
                )
                if (
                    "navigate_to" in task_plan.task_types
                    and bound_target_object_id
                ):
                    post_approach = self.approach_verifier.verify(
                        controller,
                        mode=self.agent_mode,
                        metadata=next_event.metadata,
                        object_id=bound_target_object_id,
                    ).to_context()
                    pre_approach = environment_context.get("approach") or {}
                    if (
                        action_type == "Crouch"
                        and not post_approach.get("verified")
                        and pre_approach.get("verified")
                        and pre_approach.get("objectId")
                        == bound_target_object_id
                    ):
                        post_approach = {
                            **pre_approach,
                            "verifiedAt": "pre_action",
                            "verifiedStepId": step_id,
                            "reason": (
                                "Crouch executed from the verified target "
                                "interaction pose"
                            ),
                        }
                    post_environment_context["approach"] = post_approach
                robot_after = self._robot_state(next_event.metadata)
                committed = self.agent.commit_execution(
                    session_id,
                    response_dict,
                    step_id=step_id,
                    action_success=action_success,
                    robot_before=robot_before,
                    robot_after=robot_after,
                    environment={"backend": "ai2thor", "scene": self.scene},
                    environment_context=post_environment_context,
                )
                response_dict.update(committed)
                done = bool(response_dict["done"])
                response_dict["execution"] = execution_record
                visible_objects = self._visible_objects(next_event.metadata)
                display_grounded_target = (
                    post_grounded_target or grounded_target
                )
                if display_grounded_target:
                    visible_objects = sorted(
                        set(
                            visible_objects
                            + [
                                f"{display_grounded_target['object_type']} "
                                "(segmented)"
                            ]
                        )
                    )
                topdown = self._render_unity_map_view(
                    next_event,
                    map_camera_id=map_camera_id,
                    map_camera_properties=map_camera_properties,
                    best_candidate=response_dict["observation"].get(
                        "best_candidate"
                    ),
                    instruction=instruction,
                    agent_path=agent_path,
                    planned_action=action_type,
                )
                map_view_source = "unity_third_party_camera"
                if topdown is None:
                    map_view_source = "procedural_2d_fallback"
                    topdown = self._render_topdown(
                        next_event.metadata,
                        response_dict["observation"].get("best_candidate"),
                        instruction=instruction,
                        reachable_positions=reachable_positions,
                        agent_path=agent_path,
                        planned_action=action_type,
                    )
                topdown_path = frame_dir / f"ai2thor_topdown_{step_id:02d}.png"
                topdown.save(topdown_path)
                frame = self._compose_frame(obs, topdown, response_dict, instruction, step_id, visible_objects)
                frame_path = frame_dir / f"ai2thor_frame_{step_id:02d}.png"
                frame.save(frame_path)
                step_record = DemoStep(
                    frame_path=str(frame_path.relative_to(ROOT)),
                    observation_path=str(obs_path.relative_to(ROOT)),
                    topdown_path=str(topdown_path.relative_to(ROOT)),
                    thought=response_dict["thought"],
                    action=action_type,
                    confidence=response_dict["confidence"],
                    done=done,
                    robot=robot_before,
                    best_candidate=response_dict["observation"].get("best_candidate"),
                    visible_objects=visible_objects[:10],
                    backend="ai2thor",
                    scene=self.scene,
                    structured_thought=response_dict.get("structured_thought"),
                    target_binding=response_dict.get("target_binding"),
                    skill_call=response_dict.get("skill_call"),
                    planner_source=response_dict.get("planner_source", "rule_fallback"),
                    memory_summary=response_dict.get("memory_summary", ""),
                    recalled_memories=response_dict.get("recalled_memories", []),
                    search_map=response_dict.get("search_map"),
                    model_info=response_dict.get("model_info"),
                    fallback_reason=response_dict.get("fallback_reason"),
                    task_plan=response_dict.get("task_plan"),
                    completion_status=response_dict.get("completion_status"),
                    execution=execution_record,
                    interaction_binding=interaction_binding,
                    map_view_source=map_view_source,
                )
                result.steps.append(step_record)
                emitter.emit(
                    "environment_feedback",
                    step_id=step_id,
                    robot_before=robot_before,
                    robot_after=robot_after,
                    visible_objects=visible_objects[:20],
                    action_success=action_success,
                    memory_summary=response_dict.get("memory_summary", ""),
                    completion_status=response_dict.get("completion_status"),
                    map_view_source=map_view_source,
                )
                emitter.emit(
                    "step_completed",
                    step_id=step_id,
                    step=step_record.__dict__,
                )
                if done:
                    break
                event = next_event

            video_path = run_output_dir / "ai2thor_visual_search_demo.mp4"
            self._write_video([ROOT / step.frame_path for step in result.steps], video_path)
            summary_path = run_output_dir / "ai2thor_demo_summary.json"
            result.video_path = str(video_path.relative_to(ROOT))
            result.summary_path = str(summary_path.relative_to(ROOT))
            summary_path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
            emitter.emit("episode_completed", result=result.to_dict())
            return result
        except StreamCancelled as exc:
            emitter.emit("episode_cancelled", message=str(exc))
            raise
        except Exception as exc:
            emitter.emit(
                "error",
                error_type=type(exc).__name__,
                message=str(exc),
            )
            raise
        finally:
            if controller is not None:
                controller.stop()

    def _map_action(self, action_type: str) -> str:
        return self.action_catalog.resolve_name(action_type)

    def _is_visual_search_task(self, instruction: str) -> bool:
        normalized = instruction.lower()
        search_markers = (
            "find",
            "locate",
            "search for",
            "look for",
            "找到",
            "寻找",
            "查找",
            "搜索",
        )
        manipulation_markers = (
            "pick up",
            "put",
            "open",
            "close",
            "toggle",
            "turn on",
            "turn off",
            "sit",
            "drop",
            "throw",
            "拿起",
            "放到",
            "打开",
            "关闭",
            "坐",
            "走到",
        )
        return any(marker in normalized for marker in search_markers) and not any(
            marker in normalized for marker in manipulation_markers
        )

    def _should_use_visual_search_oracle(
        self,
        task_plan: dict[str, Any] | None,
        instruction: str,
    ) -> bool:
        if task_plan:
            return (
                task_plan.get("supported") is not False
                and bool(task_plan.get("is_visual_search"))
            )
        return self._is_visual_search_task(instruction)

    def _search_action(self, step_id: int) -> str:
        if step_id > 0 and step_id % 4 == 3:
            return "MOVE_FORWARD"
        return "TURN_RIGHT"

    @staticmethod
    def _should_force_search(
        *,
        visual_search_task: bool,
        action_type: str,
    ) -> bool:
        return visual_search_task and action_type in {"STOP", "Done", "ASK_CLARIFY"}

    def _prepare_click_target(
        self,
        controller: Any,
        event: Any,
        *,
        clicked_point: list[int] | None,
        clicked_object_id: str | None,
    ) -> tuple[str | None, str | None, dict[str, Any] | None]:
        """Resolve a first-step click into (target_crop_data_url, object_id, binding_dict).

        Grounds the click to an AI2-THOR object and renders a close-up
        reference image near it. On any failure returns (None, resolved_id, ...)
        so the caller falls back to the legacy clicked_point crop path.
        """
        x = y = None
        if clicked_point and len(clicked_point) == 2:
            x, y = int(clicked_point[0]), int(clicked_point[1])
        binding = resolve_clicked_object(
            event,
            x=x,
            y=y,
            object_id=clicked_object_id,
            structural_types=STRUCTURAL_OBJECTS,
            min_mask_pixels=int(self.config.raw["closeup"]["min_mask_pixels"]),
        )
        if binding is None or binding.world_position is None:
            return None, clicked_object_id, None

        image, source, bbox = render_closeup(
            self.action_executor,
            controller,
            mode=self.agent_mode,
            target_position=binding.world_position,
            config=self.config,
        )
        binding = replace(binding, closeup_source=source, closeup_bbox=bbox)
        if image is None:
            return None, binding.object_id, binding.to_dict()
        return image_to_data_url(image), binding.object_id, binding.to_dict()

    def _ground_target_from_segmentation(self, event: Any, instruction: str) -> dict[str, Any] | None:
        masks = getattr(event, "instance_masks", None) or {}
        if not masks:
            return None
        terms = self._target_terms(instruction)
        if not terms:
            return None
        metadata_by_id = {
            str(item.get("objectId") or item.get("name") or ""): item
            for item in event.metadata.get("objects", [])
        }
        best: dict[str, Any] | None = None
        for object_id, raw_mask in masks.items():
            object_type = self._object_type(object_id, metadata_by_id.get(str(object_id)))
            normalized_type = self._normalize(object_type)
            if normalized_type in STRUCTURAL_OBJECTS or not self._matches_target(normalized_type, str(object_id), terms):
                continue
            mask = np.asarray(raw_mask).astype(bool)
            ys, xs = np.nonzero(mask)
            if len(xs) == 0:
                continue
            height, width = mask.shape[:2]
            bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
            area_ratio = float(len(xs)) / float(max(width * height, 1))
            center_x = (bbox[0] + bbox[2]) / 2.0 / max(width, 1)
            center_y = (bbox[1] + bbox[3]) / 2.0 / max(height, 1)
            center_distance = min(((center_x - 0.5) ** 2 + (center_y - 0.5) ** 2) ** 0.5 / 0.707, 1.0)
            confidence = min(0.98, 0.74 + min(area_ratio * 9.0, 0.18) + (1.0 - center_distance) * 0.08)
            candidate = {
                "label": object_type,
                "object_type": object_type,
                "object_id": str(object_id),
                "bbox": bbox,
                "confidence": round(confidence, 3),
                "color_name": "segmentation",
                "region": self._screen_region(center_x, center_y),
                "reason": "AI2-THOR instance segmentation matched the requested target object",
                "image_size": [width, height],
                "area_ratio": round(area_ratio, 5),
            }
            if best is None or candidate["confidence"] > best["confidence"]:
                best = candidate
        return best

    def _apply_grounded_target(self, response: dict[str, Any], target: dict[str, Any], action_type: str, confirmed_steps: int) -> None:
        done = action_type == "STOP"
        response["action"]["type"] = action_type
        response["action"]["args"] = {"reason": "confirmed by AI2-THOR instance segmentation"}
        response["confidence"] = target["confidence"]
        response["done"] = done
        response["planner_source"] = "simulator_oracle"
        response["skill_call"] = {
            "name": action_type,
            "args": response["action"]["args"],
            "preconditions": ["target confirmed by AI2-THOR instance segmentation"],
            "expected_observation": (
                "episode terminates with a confirmed target"
                if done
                else "target remains visible for one more confirmation step"
            ),
        }
        response["thought"] = (
            f"AI2-THOR segmentation grounds the requested target as {target['object_type']} "
            f"at {target['region']} with confidence {target['confidence']:.2f}. "
            f"{'The agent stops after confirmation.' if done else 'The agent inspects once before final confirmation.'}"
        )
        # Sync structured_thought when overriding action
        response["structured_thought"] = {
            "observation": f"AI2-THOR 分割确认目标为 {target['object_type']}，位于 {target['region']}，置信度 {target['confidence']:.2f}",
            "reasoning": f"模拟器实例分割已确认目标物体。{'已完成确认，停止搜索。' if done else '需要再次检查确认。'}",
            "action": "停止" if done else "仔细检查",
            "confidence": f"{target['confidence']:.3f}"
        }
        response["observation"]["image_size"] = target["image_size"]
        response["observation"]["target_visible"] = True
        response["observation"]["scene_summary"] = (
            f"Target object {target['object_type']} is grounded by simulator instance segmentation."
        )
        candidate = {
            "label": target["label"],
            "object_type": target.get("object_type", target["label"]),
            "object_id": target.get("object_id"),
            "bbox": target["bbox"],
            "confidence": target["confidence"],
            "color_name": target["color_name"],
            "region": target["region"],
            "reason": target["reason"],
        }
        response["observation"]["best_candidate"] = candidate
        response["observation"]["candidates"] = [candidate]

    def _apply_search_response(self, response: dict[str, Any], action_type: str) -> None:
        response["action"]["type"] = action_type
        response["action"]["args"] = {"reason": "target not grounded in AI2-THOR segmentation yet"}
        response["confidence"] = min(float(response.get("confidence", 0.0)), self.config.target_visible_threshold - 0.01)
        response["done"] = False
        response["planner_source"] = "simulator_oracle"
        response["skill_call"] = {
            "name": action_type,
            "args": response["action"]["args"],
            "preconditions": ["target not confirmed by AI2-THOR segmentation"],
            "expected_observation": "new scene coverage or a segmented target",
        }
        response["thought"] = (
            "The target is not confirmed by AI2-THOR instance segmentation yet, "
            f"so the agent executes {action_type} to continue the embodied search."
        )
        # Sync structured_thought when overriding action
        action_name_cn = {
            "TURN_LEFT": "向左转",
            "TURN_RIGHT": "向右转",
            "MOVE_FORWARD": "向前移动",
            "LOOK_UP": "向上看",
            "LOOK_DOWN": "向下看",
            "INSPECT": "仔细检查"
        }.get(action_type, action_type)
        response["structured_thought"] = {
            "observation": "AI2-THOR 实例分割尚未确认目标物体",
            "reasoning": f"模拟器分割未检测到目标，继续搜索。当前置信度 {response['confidence']:.2f}",
            "action": action_name_cn,
            "confidence": f"{response['confidence']:.3f}"
        }
        response["observation"]["target_visible"] = False
        response["observation"]["scene_summary"] = "No simulator-grounded target object is visible in the current robot view."
        response["observation"]["best_candidate"] = None
        response["observation"]["candidates"] = []

    def _target_terms(self, instruction: str) -> list[str]:
        lower = instruction.lower()
        matches: list[tuple[int, str]] = []
        for normalized, aliases in TARGET_ALIASES.items():
            for alias in aliases:
                index = lower.find(alias.lower())
                if index >= 0:
                    matches.append((index, normalized))
                    break
        return [item[1] for item in sorted(matches)]

    def _matches_target(self, object_type: str, object_id: str, terms: list[str]) -> bool:
        normalized_id = self._normalize(object_id)
        return any(term in object_type or term in normalized_id or object_type in term for term in terms)

    def _object_type(self, object_id: str, metadata: dict[str, Any] | None) -> str:
        if metadata and metadata.get("objectType"):
            return str(metadata["objectType"])
        return object_id.split("|", 1)[0]

    def _normalize(self, text: str) -> str:
        return "".join(ch for ch in text.lower() if ch.isalnum())

    def _screen_region(self, center_x: float, center_y: float) -> str:
        col = "left" if center_x < 1 / 3 else "right" if center_x > 2 / 3 else "center"
        row = "top" if center_y < 1 / 3 else "bottom" if center_y > 2 / 3 else "middle"
        return f"{row} {col}"

    def _visible_objects(self, metadata: dict[str, Any]) -> list[str]:
        names = []
        for item in metadata.get("objects", []):
            if item.get("visible"):
                names.append(str(item.get("objectType") or item.get("name") or "object"))
        return sorted(set(names))

    def _robot_state(self, metadata: dict[str, Any]) -> dict[str, float]:
        agent = metadata.get("agent", {})
        position = agent.get("position", {})
        rotation = agent.get("rotation", {})
        return {
            "x": float(position.get("x", 0.0)),
            "y": float(position.get("z", 0.0)),
            "heading": float(rotation.get("y", 0.0)),
        }

    def _initialize_map_camera(
        self,
        *,
        controller: Any,
        event: Any,
        emitter: StreamEventEmitter,
    ) -> tuple[Any, int | None, dict[str, Any] | None]:
        current_event = event
        try:
            properties_execution = self.action_executor.execute(
                controller,
                mode=self.agent_mode,
                action="GetMapViewCameraProperties",
                actor="manual",
            )
            current_event = properties_execution.event
            if not properties_execution.success:
                raise RuntimeError(properties_execution.error_message)
            properties = current_event.metadata.get("actionReturn")
            if not isinstance(properties, dict):
                raise RuntimeError(
                    "GetMapViewCameraProperties returned no camera properties"
                )

            camera_args = dict(properties)
            camera_args["antiAliasing"] = "fxaa"
            camera_execution = self.action_executor.execute(
                controller,
                mode=self.agent_mode,
                action="AddThirdPartyCamera",
                args=camera_args,
                actor="system",
            )
            current_event = camera_execution.event
            if not camera_execution.success:
                raise RuntimeError(camera_execution.error_message)
            frames = getattr(current_event, "third_party_camera_frames", [])
            if not frames:
                raise RuntimeError(
                    "AddThirdPartyCamera returned no camera frame"
                )
            camera_id = len(frames) - 1
            emitter.emit(
                "map_camera_ready",
                source="unity_third_party_camera",
                camera_id=camera_id,
                properties=properties,
            )
            return current_event, camera_id, properties
        except Exception as exc:
            emitter.emit(
                "map_camera_fallback",
                source="procedural_2d_fallback",
                reason=str(exc),
            )
            return current_event, None, None

    @staticmethod
    def _heading_triangle(
        center_x: int,
        center_y: int,
        heading_degrees: float,
    ) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
        """Return a screen-space marker using AI2-THOR's yaw convention.

        AI2-THOR heading 0 faces +Z. The map renders +Z upward, so positive
        yaw rotates clockwise: 0 up, 90 right, 180 down, and 270 left.
        """
        rotation = math.radians(heading_degrees)
        nose = (
            center_x + int(math.sin(rotation) * 28),
            center_y - int(math.cos(rotation) * 28),
        )
        left = (
            center_x + int(math.sin(rotation + 2.5) * 18),
            center_y - int(math.cos(rotation + 2.5) * 18),
        )
        right = (
            center_x + int(math.sin(rotation - 2.5) * 18),
            center_y - int(math.cos(rotation - 2.5) * 18),
        )
        return nose, left, right

    @staticmethod
    def _project_unity_map_point(
        *,
        x: float,
        z: float,
        camera_properties: dict[str, Any],
        size: int,
    ) -> tuple[int, int]:
        position = camera_properties.get("position") or {}
        center_x = float(position.get("x", 0.0))
        center_z = float(position.get("z", 0.0))
        half_extent = float(
            camera_properties.get("orthographicSize") or 1.0
        )
        if half_extent <= 0:
            raise ValueError("orthographicSize must be positive")
        px = int(
            round(
                (x - center_x + half_extent)
                / (2 * half_extent)
                * size
            )
        )
        py = int(
            round(
                (center_z + half_extent - z)
                / (2 * half_extent)
                * size
            )
        )
        return px, py

    def _render_unity_map_view(
        self,
        event: Any,
        *,
        map_camera_id: int | None,
        map_camera_properties: dict[str, Any] | None,
        best_candidate: dict[str, Any] | None,
        instruction: str,
        agent_path: list[dict[str, float]] | None,
        planned_action: str | None,
    ) -> Image.Image | None:
        if map_camera_id is None or map_camera_properties is None:
            return None
        frames = getattr(event, "third_party_camera_frames", [])
        if map_camera_id >= len(frames):
            return None

        frame = Image.fromarray(frames[map_camera_id]).convert("RGB")
        side = min(frame.size)
        left = (frame.width - side) // 2
        top = (frame.height - side) // 2
        image = frame.crop((left, top, left + side, top + side)).resize(
            (520, 520)
        )
        draw = ImageDraw.Draw(image)

        def project(x: float, z: float) -> tuple[int, int]:
            return self._project_unity_map_point(
                x=x,
                z=z,
                camera_properties=map_camera_properties,
                size=520,
            )

        path = [
            project(float(point["x"]), float(point["y"]))
            for point in (agent_path or [])
            if "x" in point and "y" in point
        ]
        if len(path) >= 2:
            draw.line(path, fill=(0, 255, 215), width=5)
        for px, py in path:
            draw.ellipse(
                [px - 3, py - 3, px + 3, py + 3],
                fill=(0, 190, 160),
            )

        target_terms = self._target_terms(instruction)
        if not target_terms and best_candidate:
            target_label = self._normalize(
                str(best_candidate.get("label") or "")
            )
            if target_label:
                target_terms = [target_label]
        target_object_id = (
            str(best_candidate.get("object_id") or "")
            if best_candidate
            else ""
        )
        target_index = 0
        for item in event.metadata.get("objects", []):
            position = item.get("position") or {}
            object_type = str(
                item.get("objectType") or item.get("name") or "object"
            )
            normalized_type = self._normalize(object_type)
            object_id = str(
                item.get("objectId") or item.get("name") or object_type
            )
            is_target = (
                best_candidate is not None
                and "x" in position
                and "z" in position
                and bool(item.get("visible"))
                and normalized_type not in STRUCTURAL_OBJECTS
                and target_terms
                and self._matches_target(
                    normalized_type,
                    object_id,
                    target_terms,
                )
                and (not target_object_id or object_id == target_object_id)
            )
            if not is_target:
                continue
            px, py = project(
                float(position["x"]),
                float(position["z"]),
            )
            draw.ellipse(
                [px - 9, py - 9, px + 9, py + 9],
                fill=(255, 70, 55),
                outline=(110, 20, 15),
                width=2,
            )
            if target_index == 0:
                draw.rectangle(
                    [px + 12, py - 12, px + 150, py + 12],
                    fill=(8, 13, 23),
                )
                draw.text(
                    (px + 17, py - 8),
                    f"Target: {object_type}"[:24],
                    fill=(255, 245, 240),
                )
            target_index += 1

        agent = event.metadata.get("agent", {})
        agent_position = agent.get("position") or {}
        ax, ay = project(
            float(agent_position.get("x", 0.0)),
            float(agent_position.get("z", 0.0)),
        )
        heading = float(
            (agent.get("rotation") or {}).get("y", 0.0)
        ) % 360.0
        nose, marker_left, marker_right = self._heading_triangle(
            ax,
            ay,
            heading,
        )
        draw.polygon(
            [nose, marker_left, marker_right],
            fill=(0, 210, 255),
            outline=(0, 45, 70),
        )
        draw.rectangle([10, 10, 510, 58], fill=(8, 13, 23))
        draw.text(
            (22, 18),
            f"Unity 3D global map | {self.scene}",
            fill=(245, 248, 252),
        )
        action_text = (
            f" | next {planned_action}" if planned_action else ""
        )
        draw.text(
            (22, 38),
            f"heading {heading:.0f} deg{action_text}",
            fill=(100, 240, 220),
        )
        return image

    def _render_topdown(
        self,
        metadata: dict[str, Any],
        best_candidate: dict[str, Any] | None,
        *,
        instruction: str = "",
        reachable_positions: list[dict[str, Any]] | None = None,
        agent_path: list[dict[str, float]] | None = None,
        planned_action: str | None = None,
    ) -> Image.Image:
        image = Image.new("RGB", (520, 520), (245, 247, 250))
        draw = ImageDraw.Draw(image)
        draw.rectangle([16, 16, 504, 504], outline=(20, 33, 48), width=4)
        reachable_positions = reachable_positions or []
        agent_path = agent_path or []
        target_terms = self._target_terms(instruction)
        if not target_terms and best_candidate:
            target_label = self._normalize(str(best_candidate.get("label") or ""))
            if target_label:
                target_terms = [target_label]

        target_positions: list[tuple[float, float, str, bool]] = []
        target_object_id = str(best_candidate.get("object_id") or "") if best_candidate else ""
        for item in metadata.get("objects", []):
            pos = item.get("position") or {}
            object_type = str(item.get("objectType") or item.get("name") or "object")
            normalized_type = self._normalize(object_type)
            object_id = str(item.get("objectId") or item.get("name") or object_type)
            if (
                best_candidate is not None
                and
                "x" in pos
                and "z" in pos
                and bool(item.get("visible"))
                and normalized_type not in STRUCTURAL_OBJECTS
                and target_terms
                and self._matches_target(normalized_type, object_id, target_terms)
                and (not target_object_id or object_id == target_object_id)
            ):
                target_positions.append(
                    (
                        float(pos["x"]),
                        float(pos["z"]),
                        object_type,
                        bool(item.get("visible")),
                    )
                )

        agent = metadata.get("agent", {})
        agent_position = agent.get("position", {})
        all_points: list[tuple[float, float]] = [
            (float(point["x"]), float(point["z"]))
            for point in reachable_positions
            if "x" in point and "z" in point
        ]
        all_points.extend(
            (float(point["x"]), float(point["y"]))
            for point in agent_path
            if "x" in point and "y" in point
        )
        all_points.extend((x, z) for x, z, _, _ in target_positions)
        all_points.append(
            (
                float(agent_position.get("x", 0.0)),
                float(agent_position.get("z", 0.0)),
            )
        )
        xs = [point[0] for point in all_points] or [0.0]
        zs = [point[1] for point in all_points] or [0.0]
        min_x, max_x = min(xs) - 0.5, max(xs) + 0.5
        min_z, max_z = min(zs) - 0.5, max(zs) + 0.5

        def project(x: float, z: float) -> tuple[int, int]:
            px = 28 + int((x - min_x) / max(max_x - min_x, 0.1) * 464)
            py = 28 + int((max_z - z) / max(max_z - min_z, 0.1) * 464)
            return px, py

        for point in reachable_positions:
            if "x" not in point or "z" not in point:
                continue
            px, py = project(float(point["x"]), float(point["z"]))
            draw.ellipse([px - 2, py - 2, px + 2, py + 2], fill=(176, 186, 197))

        projected_path = [
            project(float(point["x"]), float(point["y"]))
            for point in agent_path
            if "x" in point and "y" in point
        ]
        if len(projected_path) >= 2:
            draw.line(projected_path, fill=(25, 170, 156), width=5)
        for px, py in projected_path:
            draw.ellipse([px - 3, py - 3, px + 3, py + 3], fill=(15, 118, 110))

        labeled_target = target_positions[0] if target_positions else None
        for x, z, label, visible in target_positions:
            px, py = project(x, z)
            color = (226, 60, 46) if visible else (244, 142, 38)
            draw.ellipse([px - 8, py - 8, px + 8, py + 8], fill=color, outline=(135, 35, 25), width=2)
            if labeled_target == (x, z, label, visible):
                draw.text((px + 11, py - 8), f"Target: {label}"[:28], fill=(135, 35, 25))

        ax, ay = project(
            float(agent_position.get("x", 0.0)),
            float(agent_position.get("z", 0.0)),
        )
        heading = float(agent.get("rotation", {}).get("y", 0.0)) % 360.0
        nose, left, right = self._heading_triangle(ax, ay, heading)
        draw.polygon([nose, left, right], fill=(0, 126, 167), outline=(0, 70, 96))
        draw.text((24, 24), f"AI2-THOR {self.scene} reachable map", fill=(15, 23, 42))
        action_suffix = f" before {planned_action}" if planned_action else ""
        draw.text(
            (24, 44),
            f"Heading {heading:.0f} deg{action_suffix} (0 deg = +Z / up)",
            fill=(55, 65, 78),
        )
        draw.text((24, 64), "gray reachable | teal path | blue robot | red target", fill=(55, 65, 78))
        if best_candidate:
            draw.text(
                (24, 486),
                f"Best candidate: {best_candidate.get('label')} {best_candidate.get('confidence')}",
                fill=(170, 35, 24),
            )
        return image

    def _compose_frame(
        self,
        obs: Image.Image,
        topdown: Image.Image,
        response: dict[str, Any],
        instruction: str,
        step_id: int,
        visible_objects: list[str],
    ) -> Image.Image:
        canvas = Image.new("RGB", (1600, 900), (8, 13, 23))
        draw = ImageDraw.Draw(canvas)
        draw.text((34, 28), f"AI2-THOR x Agent | {self.scene}", fill=(245, 248, 252))
        draw.text((34, 62), f"Instruction: {instruction}", fill=(180, 195, 210))
        canvas.paste(obs.resize((900, 506)), (34, 116))
        canvas.paste(topdown.resize((420, 420)), (980, 116))
        draw.rectangle([34, 116, 934, 622], outline=(57, 217, 198), width=3)
        draw.rectangle([980, 116, 1400, 536], outline=(88, 166, 255), width=3)
        panel_x = 980
        panel_y = 570
        draw.rounded_rectangle([panel_x, panel_y, 1560, 850], radius=8, fill=(242, 246, 250))
        draw.text((panel_x + 20, panel_y + 18), f"Step {step_id}", fill=(15, 23, 42))
        draw.text((panel_x + 20, panel_y + 52), f"Planned action: {response['action']['type']}", fill=(0, 126, 120))
        draw.text((panel_x + 20, panel_y + 86), f"Confidence: {response['confidence']:.3f}", fill=(190, 45, 35))
        self._wrapped_text(draw, "Thought: " + response["thought"], panel_x + 20, panel_y + 126, 74, fill=(22, 31, 43))
        visible = ", ".join(visible_objects[:8]) or "None"
        self._wrapped_text(draw, "Visible: " + visible, 34, 656, 110, fill=(220, 230, 242))
        return canvas

    def _wrapped_text(self, draw: ImageDraw.ImageDraw, text: str, x: int, y: int, width: int, fill: tuple[int, int, int]) -> None:
        line = ""
        for word in text.split():
            probe = f"{line} {word}".strip()
            if len(probe) > width:
                draw.text((x, y), line, fill=fill)
                y += 23
                line = word
            else:
                line = probe
        if line:
            draw.text((x, y), line, fill=fill)

    def _write_video(self, frames: list[Path], path: Path) -> None:
        write_browser_compatible_mp4(
            frames,
            path,
            fps=2.0,
            hold_frames=2,
        )


def ai2thor_environment_report() -> dict[str, Any]:
    status = AI2ThorVisualSearchDemo.status().to_dict()
    player_log = Path.home() / ".config" / "unity3d" / "Allen Institute for Artificial Intelligence" / "AI2-THOR" / "Player.log"
    if player_log.exists():
        try:
            status["player_log_tail"] = player_log.read_text(encoding="utf-8", errors="replace").splitlines()[-40:]
        except OSError as exc:
            status["player_log_error"] = repr(exc)
    if shutil.which("vulkaninfo"):
        try:
            probe = subprocess.run(["vulkaninfo", "--summary"], capture_output=True, text=True, timeout=10, check=False)
            status["vulkan_summary"] = (probe.stdout + probe.stderr).splitlines()[:80]
        except Exception as exc:
            status["vulkan_summary_error"] = repr(exc)
    return status
