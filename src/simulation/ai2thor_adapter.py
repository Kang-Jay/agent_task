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
from PIL import Image, ImageDraw, ImageFont

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
    ai2thor_platform_kwargs,
    create_controller_safely,
    should_snap_to_grid,
)
from src.simulation.object_closeup import render_closeup, resolve_clicked_object
from src.simulation.room_simulator import DemoResult, DemoStep, load_render_font
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
    "door": ["door", "doorway", "门", "房门", "门口", "出口", "右边的门"],
    "vase": ["vase", "花瓶"],
    "cup": ["cup", "杯子"],
    "mug": ["mug", "马克杯"],
    "bowl": ["bowl", "碗"],
    "box": ["box", "cardboardbox", "cardboard box", "纸箱", "箱子", "盒子"],
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


@dataclass(frozen=True)
class DoorThresholdGeometry:
    source: str
    center: tuple[float, float]
    tangent: tuple[float, float]
    normal: tuple[float, float]
    half_length: float | None


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
        initial_pose: dict[str, Any] | None = None,
        emit: Callable[[dict[str, Any]], None] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> DemoResult:
        from ai2thor.controller import Controller  # type: ignore

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
                **ai2thor_platform_kwargs(),
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
            if initial_pose:
                event = self._teleport_initial_pose(controller, initial_pose)
            episode_start_metadata = event.metadata
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
                pre_action_obs_path = (
                    frame_dir / f"ai2thor_obs_{step_id:02d}.png"
                )
                obs.save(pre_action_obs_path)
                emitter.emit(
                    "observation_ready",
                    step_id=step_id,
                    observation_path=str(
                        pre_action_obs_path.relative_to(ROOT)
                    ),
                    observation_phase="before_action",
                    purpose="model_input_audit",
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
                vlm_confirmation = self._vlm_target_confirmation(response_dict)
                vlm_authoritative = self.config.visual_search_authority == "vlm"
                if visual_search_task and vlm_authoritative:
                    # The VLM's own visual judgment is authoritative. Instance
                    # segmentation (grounded_target) is retained only as
                    # cross-validation evidence, never as the decider.
                    if (
                        vlm_confirmation["target_visible"]
                        and vlm_confirmation["target_confidence"]
                        >= self.config.stop_confidence_threshold
                    ):
                        confirmed_target_steps += 1
                    else:
                        confirmed_target_steps = 0
                    if confirmed_target_steps >= 1:
                        action_type = (
                            "STOP" if confirmed_target_steps >= 2 else "INSPECT"
                        )
                        self._apply_vlm_target_confirmation(
                            response_dict,
                            confirmation=vlm_confirmation,
                            grounded_target=grounded_target,
                            action_type=action_type,
                            confirmed_steps=confirmed_target_steps,
                        )
                    else:
                        action_type = response_dict["action"]["type"]
                        if self._should_force_search(
                            visual_search_task=visual_search_task,
                            action_type=action_type,
                        ):
                            action_type = self._search_action(step_id)
                            self._apply_search_response(response_dict, action_type)
                elif grounded_target and visual_search_task:
                    # Legacy: simulator instance segmentation is authoritative.
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
                if (
                    not agent_path
                    or robot_before["x"] != agent_path[-1]["x"]
                    or robot_before["y"] != agent_path[-1]["y"]
                ):
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
                post_action_obs = Image.fromarray(
                    next_event.frame
                ).convert("RGB")
                post_action_obs_path = frame_dir / (
                    f"ai2thor_obs_after_{step_id:02d}.png"
                )
                post_action_obs.save(post_action_obs_path)
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
                final_state = self._interaction_final_state(
                    instruction=instruction,
                    metadata=next_event.metadata,
                )
                if final_state:
                    post_environment_context["final_state"] = final_state
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
                if "exit_room" in task_plan.task_types:
                    selected_door_object_id = self._selected_exit_door_object_id(
                        response=response_dict,
                        bound_target_object_id=bound_target_object_id,
                        before_metadata=event.metadata,
                        after_metadata=next_event.metadata,
                    )
                    door_crossing = self._door_crossing_context(
                        instruction=instruction,
                        start_metadata=episode_start_metadata,
                        before_metadata=event.metadata,
                        after_metadata=next_event.metadata,
                        selected_door_object_id=selected_door_object_id,
                    )
                    if door_crossing:
                        post_environment_context["door_crossing"] = door_crossing
                        post_environment_context["exit"] = door_crossing
                robot_after = self._robot_state(next_event.metadata)
                if (
                    not agent_path
                    or robot_after["x"] != agent_path[-1]["x"]
                    or robot_after["y"] != agent_path[-1]["y"]
                ):
                    agent_path.append(robot_after)
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
                frame = self._compose_frame(
                    post_action_obs,
                    topdown,
                    response_dict,
                    instruction,
                    step_id,
                    visible_objects,
                )
                frame_path = frame_dir / f"ai2thor_frame_{step_id:02d}.png"
                frame.save(frame_path)
                step_record = DemoStep(
                    frame_path=str(frame_path.relative_to(ROOT)),
                    observation_path=str(
                        post_action_obs_path.relative_to(ROOT)
                    ),
                    topdown_path=str(topdown_path.relative_to(ROOT)),
                    thought=response_dict["thought"],
                    action=action_type,
                    confidence=response_dict["confidence"],
                    done=done,
                    robot=robot_after,
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
                    environment_context=environment_context,
                    post_environment_context=post_environment_context,
                    map_view_source=map_view_source,
                )
                result.steps.append(step_record)
                emitter.emit(
                    "environment_feedback",
                    step_id=step_id,
                    robot_before=robot_before,
                    robot_after=robot_after,
                    observation_path=str(
                        post_action_obs_path.relative_to(ROOT)
                    ),
                    observation_phase="after_action",
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

    @staticmethod
    def _vlm_target_confirmation(response: dict[str, Any]) -> dict[str, Any]:
        """Extract the VLM's own visual target-confirmation signal.

        The model reports ``target_visible`` and ``target_confidence`` in its
        planner JSON (surfaced through ``model_info``). When the VLM is the
        authority for visual search, these drive INSPECT/STOP instead of the
        simulator's instance segmentation.
        """
        model_info = response.get("model_info") or {}
        target_visible = bool(model_info.get("target_visible", False))
        try:
            target_confidence = float(model_info.get("target_confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            target_confidence = 0.0
        target_confidence = max(0.0, min(1.0, target_confidence))
        vision_input_used = bool(model_info.get("vision_input_used", False))
        return {
            "target_visible": target_visible and vision_input_used,
            "target_confidence": target_confidence,
            "vision_input_used": vision_input_used,
        }

    def _apply_vlm_target_confirmation(
        self,
        response: dict[str, Any],
        *,
        confirmation: dict[str, Any],
        grounded_target: dict[str, Any] | None,
        action_type: str,
        confirmed_steps: int,
    ) -> None:
        """Commit an INSPECT/STOP decision driven by the VLM's visual judgment.

        Segmentation, when present, is attached as cross-validation evidence
        only; the authoritative confidence comes from the VLM.
        """
        done = action_type == "STOP"
        confidence = confirmation["target_confidence"]
        response["action"]["type"] = action_type
        response["action"]["args"] = {"reason": "confirmed by VLM visual grounding"}
        response["confidence"] = confidence
        response["done"] = done
        response["planner_source"] = "model_planner"
        cross_check = None
        if grounded_target is not None:
            cross_check = {
                "object_type": grounded_target.get("object_type"),
                "object_id": grounded_target.get("object_id"),
                "segmentation_confidence": grounded_target.get("confidence"),
                "agreement": True,
            }
        response["skill_call"] = {
            "name": action_type,
            "args": response["action"]["args"],
            "preconditions": ["target visually confirmed by the VLM"],
            "expected_observation": (
                "episode terminates with a VLM-confirmed target"
                if done
                else "target remains visible for one more confirmation step"
            ),
        }
        response["thought"] = (
            f"The VLM visually confirms the requested target with confidence "
            f"{confidence:.2f}. "
            f"{'The agent stops after confirmation.' if done else 'The agent inspects once before final confirmation.'}"
        )
        preserved_trace = (response.get("structured_thought") or {}).get("decision_trace")
        response["structured_thought"] = {
            "observation": f"VLM 视觉确认目标可见，置信度 {confidence:.2f}",
            "reasoning": (
                "VLM 已在当前画面中确认目标物体。"
                + ("已完成确认，停止搜索。" if done else "需要再次检查确认。")
            ),
            "action": "停止" if done else "仔细检查",
            "confidence": f"{confidence:.3f}",
        }
        if preserved_trace:
            response["structured_thought"]["decision_trace"] = preserved_trace
        observation = response.setdefault("observation", {})
        observation["target_visible"] = True
        observation["scene_summary"] = (
            "Target object is visually grounded by the VLM."
        )
        observation["target_confirmation_source"] = "vlm"
        if cross_check is not None:
            observation["segmentation_cross_check"] = cross_check

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
        preserved_trace = (response.get("structured_thought") or {}).get("decision_trace")
        response["structured_thought"] = {
            "observation": f"AI2-THOR 分割确认目标为 {target['object_type']}，位于 {target['region']}，置信度 {target['confidence']:.2f}",
            "reasoning": f"模拟器实例分割已确认目标物体。{'已完成确认，停止搜索。' if done else '需要再次检查确认。'}",
            "action": "停止" if done else "仔细检查",
            "confidence": f"{target['confidence']:.3f}"
        }
        if preserved_trace:
            response["structured_thought"]["decision_trace"] = preserved_trace
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
        preserved_trace = (response.get("structured_thought") or {}).get("decision_trace")
        response["structured_thought"] = {
            "observation": "AI2-THOR 实例分割尚未确认目标物体",
            "reasoning": f"模拟器分割未检测到目标，继续搜索。当前置信度 {response['confidence']:.2f}",
            "action": action_name_cn,
            "confidence": f"{response['confidence']:.3f}"
        }
        if preserved_trace:
            response["structured_thought"]["decision_trace"] = preserved_trace
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

    def _interaction_final_state(
        self,
        *,
        instruction: str,
        metadata: dict[str, Any],
    ) -> dict[str, Any] | None:
        target_terms = self._target_terms(instruction)
        if len(target_terms) < 2 or not self._requests_put_interaction(instruction):
            return None

        objects = list(metadata.get("objects") or [])
        moved_object = self._best_object_for_terms(objects, [target_terms[0]])
        receptacle = self._best_object_for_terms(
            objects,
            self._receptacle_terms(target_terms[1]),
        )
        if not moved_object or not receptacle:
            return None
        moved_object_id = str(moved_object.get("objectId") or "")
        receptacle_object_id = str(receptacle.get("objectId") or "")
        parent_receptacles = list(moved_object.get("parentReceptacles") or [])
        receptacle_object_ids = list(receptacle.get("receptacleObjectIds") or [])
        inventory_objects = list(metadata.get("inventoryObjects") or [])

        final_state = {
            "placement": {
                "movedObjectId": moved_object_id,
                "movedObjectType": str(moved_object.get("objectType") or ""),
                "receptacleObjectId": receptacle_object_id,
                "receptacleObjectType": str(receptacle.get("objectType") or ""),
                "parentReceptacles": parent_receptacles,
                "receptacleObjectIds": receptacle_object_ids,
                "inventoryObjects": inventory_objects,
            },
            "source": "ai2thor_final_metadata",
        }
        if target_terms[0] == "vase" and target_terms[1] in {"box", "cardboardbox"}:
            final_state.update(
                {
                    "vaseObjectId": moved_object_id,
                    "boxObjectId": receptacle_object_id,
                    "vaseParentReceptacles": parent_receptacles,
                    "boxReceptacleObjectIds": receptacle_object_ids,
                    "inventoryObjects": inventory_objects,
                }
            )
        return final_state

    @staticmethod
    def _requests_put_interaction(instruction: str) -> bool:
        lower = instruction.lower()
        return any(
            marker in lower
            for marker in ("put", "place into", "place in", "放入", "放到", "放进", "放在")
        )

    @staticmethod
    def _receptacle_terms(term: str) -> list[str]:
        if term == "box":
            return ["box", "cardboardbox"]
        return [term]

    def _best_object_for_terms(
        self,
        objects: list[dict[str, Any]],
        terms: list[str],
    ) -> dict[str, Any] | None:
        candidates: list[dict[str, Any]] = []
        for item in objects:
            object_type = self._normalize(str(item.get("objectType") or ""))
            object_id = str(item.get("objectId") or "")
            if self._matches_target(object_type, object_id, terms):
                candidates.append(item)
        if not candidates:
            return None
        candidates.sort(
            key=lambda item: (
                not bool(item.get("visible")),
                self._object_distance_sort_key(item),
                str(item.get("objectId") or ""),
            )
        )
        return candidates[0]

    @staticmethod
    def _object_distance_sort_key(item: dict[str, Any]) -> float:
        try:
            distance = float(item.get("distance"))
        except (TypeError, ValueError):
            return float("inf")
        return distance if math.isfinite(distance) else float("inf")

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

    @staticmethod
    def _teleport_initial_pose(controller: Any, pose: dict[str, Any]) -> Any:
        event = controller.step(
            action="TeleportFull",
            position={
                "x": float(pose["x"]),
                "y": float(pose["y"]),
                "z": float(pose["z"]),
            },
            rotation={
                "x": 0.0,
                "y": float(pose.get("rotation", 0.0)),
                "z": 0.0,
            },
            horizon=float(pose.get("horizon", 0.0)),
            standing=bool(pose.get("standing", True)),
        )
        if not event.metadata.get("lastActionSuccess"):
            message = event.metadata.get("errorMessage") or "TeleportFull failed"
            raise RuntimeError(f"initial pose teleport failed: {message}")
        return event

    @staticmethod
    def _metadata_agent_position(metadata: dict[str, Any]) -> dict[str, float]:
        agent = metadata.get("agent") or {}
        position = agent.get("position") or {}
        return {
            "x": float(position.get("x", 0.0)),
            "y": float(position.get("y", 0.0)),
            "z": float(position.get("z", 0.0)),
        }

    @staticmethod
    def _metadata_agent_heading(metadata: dict[str, Any]) -> float:
        agent = metadata.get("agent") or {}
        rotation = agent.get("rotation") or {}
        try:
            return float(rotation.get("y", 0.0))
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _side_of_threshold(
        value: float,
        threshold: float,
        *,
        epsilon: float = 1e-5,
    ) -> int:
        delta = value - threshold
        if delta < -epsilon:
            return -1
        if delta > epsilon:
            return 1
        return 0

    @staticmethod
    def _is_door_metadata(item: dict[str, Any]) -> bool:
        label = " ".join(
            str(item.get(key) or "")
            for key in ("objectType", "objectId", "name")
        ).lower()
        return "door" in label

    @classmethod
    def _door_objects(cls, *metadata_items: dict[str, Any]) -> list[dict[str, Any]]:
        by_id: dict[str, dict[str, Any]] = {}
        anonymous: list[dict[str, Any]] = []
        for metadata in metadata_items:
            for item in metadata.get("objects", []):
                if not isinstance(item, dict) or not cls._is_door_metadata(item):
                    continue
                object_id = str(item.get("objectId") or "")
                if object_id:
                    by_id[object_id] = item
                else:
                    anonymous.append(item)
        return [*by_id.values(), *anonymous]

    @staticmethod
    def _xz_from_value(value: Any) -> tuple[float, float] | None:
        if isinstance(value, dict):
            if "x" not in value or "z" not in value:
                return None
            try:
                return float(value["x"]), float(value["z"])
            except (TypeError, ValueError):
                return None
        if isinstance(value, (list, tuple)):
            if len(value) >= 3:
                try:
                    return float(value[0]), float(value[2])
                except (TypeError, ValueError):
                    return None
            if len(value) == 2:
                try:
                    return float(value[0]), float(value[1])
                except (TypeError, ValueError):
                    return None
        return None

    @classmethod
    def _door_center_xz(cls, door: dict[str, Any]) -> tuple[float, float] | None:
        center = cls._xz_from_value(door.get("position"))
        if center is not None:
            return center
        for key in ("objectOrientedBoundingBox", "axisAlignedBoundingBox"):
            box = door.get(key)
            if isinstance(box, dict):
                center = cls._xz_from_value(box.get("center"))
                if center is not None:
                    return center
        return None

    @classmethod
    def _door_bbox_xz_points(cls, door: dict[str, Any]) -> list[tuple[float, float]]:
        points: list[tuple[float, float]] = []
        for key in ("objectOrientedBoundingBox", "axisAlignedBoundingBox"):
            box = door.get(key)
            if not isinstance(box, dict):
                continue
            corners = box.get("cornerPoints") or box.get("corners")
            if isinstance(corners, list):
                for corner in corners:
                    point = cls._xz_from_value(corner)
                    if point is not None:
                        points.append(point)
            center = cls._xz_from_value(box.get("center"))
            size = box.get("size")
            if center is not None and isinstance(size, dict):
                try:
                    half_x = abs(float(size.get("x", 0.0))) / 2.0
                    half_z = abs(float(size.get("z", 0.0))) / 2.0
                except (TypeError, ValueError):
                    continue
                if half_x > 0.0 or half_z > 0.0:
                    cx, cz = center
                    points.extend(
                        [
                            (cx - half_x, cz - half_z),
                            (cx - half_x, cz + half_z),
                            (cx + half_x, cz - half_z),
                            (cx + half_x, cz + half_z),
                        ]
                    )
        unique: dict[tuple[int, int], tuple[float, float]] = {}
        for x, z in points:
            unique[(round(x, 5), round(z, 5))] = (x, z)
        return list(unique.values())

    @staticmethod
    def _normalize_xz_vector(
        x: float,
        z: float,
    ) -> tuple[float, float] | None:
        length = math.hypot(x, z)
        if length <= 1e-7:
            return None
        return x / length, z / length

    @classmethod
    def _principal_xz_axis(
        cls,
        points: list[tuple[float, float]],
    ) -> tuple[tuple[float, float], float] | None:
        if len(points) < 2:
            return None
        mean_x = sum(point[0] for point in points) / len(points)
        mean_z = sum(point[1] for point in points) / len(points)
        centered = [(x - mean_x, z - mean_z) for x, z in points]
        var_x = sum(x * x for x, _ in centered) / len(centered)
        var_z = sum(z * z for _, z in centered) / len(centered)
        cov_xz = sum(x * z for x, z in centered) / len(centered)
        if var_x + var_z <= 1e-8:
            return None
        angle = 0.5 * math.atan2(2.0 * cov_xz, var_x - var_z)
        axis = cls._normalize_xz_vector(math.cos(angle), math.sin(angle))
        if axis is None:
            return None
        projections = [x * axis[0] + z * axis[1] for x, z in points]
        length = max(projections) - min(projections)
        if length <= 1e-5:
            return None
        return axis, length

    @staticmethod
    def _door_rotation_y(door: dict[str, Any]) -> float | None:
        rotation = door.get("rotation")
        if not isinstance(rotation, dict) or "y" not in rotation:
            return None
        try:
            return math.radians(float(rotation["y"]))
        except (TypeError, ValueError):
            return None

    def _door_threshold_geometries(
        self,
        door: dict[str, Any],
        *,
        before_position: dict[str, float],
        after_position: dict[str, float],
    ) -> list[DoorThresholdGeometry]:
        center = self._door_center_xz(door)
        if center is None:
            return []

        geometries: list[DoorThresholdGeometry] = []
        points = self._door_bbox_xz_points(door)
        axis = self._principal_xz_axis(points)
        if axis is not None:
            tangent, length = axis
            normal = (-tangent[1], tangent[0])
            geometries.append(
                DoorThresholdGeometry(
                    source="ai2thor_door_bounding_box",
                    center=center,
                    tangent=tangent,
                    normal=normal,
                    half_length=max(length / 2.0, DEFAULT_GRID_SIZE_METERS),
                )
            )

        if not geometries:
            yaw = self._door_rotation_y(door)
            if yaw is not None:
                tangent = self._normalize_xz_vector(math.sin(yaw), math.cos(yaw))
                if tangent is not None:
                    geometries.append(
                        DoorThresholdGeometry(
                            source="ai2thor_door_rotation",
                            center=center,
                            tangent=tangent,
                            normal=(-tangent[1], tangent[0]),
                            half_length=None,
                        )
                    )

        if not geometries:
            motion = self._normalize_xz_vector(
                after_position["x"] - before_position["x"],
                after_position["z"] - before_position["z"],
            )
            if motion is not None:
                geometries.append(
                    DoorThresholdGeometry(
                        source="ai2thor_door_position_and_agent_motion",
                        center=center,
                        tangent=(-motion[1], motion[0]),
                        normal=motion,
                        half_length=None,
                    )
                )
        return geometries

    @staticmethod
    def _signed_threshold_distance(
        position: dict[str, float],
        geometry: DoorThresholdGeometry,
    ) -> float:
        return (
            (position["x"] - geometry.center[0]) * geometry.normal[0]
            + (position["z"] - geometry.center[1]) * geometry.normal[1]
        )

    @staticmethod
    def _dominant_threshold_axis(
        geometry: DoorThresholdGeometry,
    ) -> tuple[str, float]:
        if abs(geometry.normal[1]) >= abs(geometry.normal[0]):
            return "z", geometry.center[1]
        return "x", geometry.center[0]

    @staticmethod
    def _threshold_crossing(
        *,
        start: dict[str, float],
        end: dict[str, float],
        geometry: DoorThresholdGeometry,
    ) -> dict[str, Any]:
        side_epsilon = 1e-4
        segment_tolerance = DEFAULT_GRID_SIZE_METERS * 0.75
        unbounded_center_tolerance = DEFAULT_GRID_SIZE_METERS * 3.0
        start_distance = AI2ThorVisualSearchDemo._signed_threshold_distance(
            start,
            geometry,
        )
        end_distance = AI2ThorVisualSearchDemo._signed_threshold_distance(
            end,
            geometry,
        )
        start_side = AI2ThorVisualSearchDemo._side_of_threshold(
            start_distance,
            0.0,
            epsilon=side_epsilon,
        )
        end_side = AI2ThorVisualSearchDemo._side_of_threshold(
            end_distance,
            0.0,
            epsilon=side_epsilon,
        )
        crossed_line = (
            start_side != 0
            and end_side != 0
            and start_side != end_side
        )

        intersection: dict[str, float] | None = None
        tangent_offset: float | None = None
        center_distance: float | None = None
        within_threshold_segment = False
        if crossed_line:
            denominator = start_distance - end_distance
            if abs(denominator) > side_epsilon:
                ratio = start_distance / denominator
                if -side_epsilon <= ratio <= 1.0 + side_epsilon:
                    ix = start["x"] + (end["x"] - start["x"]) * ratio
                    iz = start["z"] + (end["z"] - start["z"]) * ratio
                    intersection = {"x": ix, "z": iz}
                    tangent_offset = (
                        (ix - geometry.center[0]) * geometry.tangent[0]
                        + (iz - geometry.center[1]) * geometry.tangent[1]
                    )
                    center_distance = math.hypot(
                        ix - geometry.center[0],
                        iz - geometry.center[1],
                    )
                    if geometry.half_length is None:
                        within_threshold_segment = (
                            center_distance <= unbounded_center_tolerance
                        )
                    else:
                        within_threshold_segment = (
                            abs(tangent_offset)
                            <= geometry.half_length + segment_tolerance
                        )

        return {
            "crossed": crossed_line and within_threshold_segment,
            "line_crossed": crossed_line,
            "within_threshold_segment": within_threshold_segment,
            "start_side": start_side,
            "end_side": end_side,
            "start_signed_distance": start_distance,
            "end_signed_distance": end_distance,
            "intersection": intersection,
            "tangent_offset": tangent_offset,
            "center_distance": center_distance,
        }

    @classmethod
    def _door_relation_to_agent(
        cls,
        *,
        door: dict[str, Any],
        start_metadata: dict[str, Any],
        requested_relation: str | None,
    ) -> dict[str, Any]:
        center = cls._door_center_xz(door)
        start_position = cls._metadata_agent_position(start_metadata)
        heading = cls._metadata_agent_heading(start_metadata)
        if center is None:
            return {
                "requested_relation": requested_relation,
                "relation_to_agent": None,
                "relation_verified": requested_relation is None,
                "relation_score": None,
                "relation_frame": "agent_initial_heading",
                "relation_reason": "door center is unavailable",
            }
        yaw = math.radians(heading)
        right_vector = (math.cos(yaw), -math.sin(yaw))
        dx = center[0] - start_position["x"]
        dz = center[1] - start_position["z"]
        right_score = dx * right_vector[0] + dz * right_vector[1]
        epsilon = DEFAULT_GRID_SIZE_METERS * 0.2
        relation_to_agent = (
            "right"
            if right_score > epsilon
            else "left"
            if right_score < -epsilon
            else "center"
        )
        return {
            "requested_relation": requested_relation,
            "relation_to_agent": relation_to_agent,
            "relation_verified": (
                requested_relation is None
                or relation_to_agent == requested_relation
            ),
            "relation_score": right_score,
            "relation_frame": "agent_initial_heading",
            "relation_reason": (
                f"door center is {relation_to_agent} of initial heading"
            ),
        }

    def _door_crossing_context(
        self,
        *,
        instruction: str,
        start_metadata: dict[str, Any],
        before_metadata: dict[str, Any],
        after_metadata: dict[str, Any],
        selected_door_object_id: str | None = None,
    ) -> dict[str, Any] | None:
        doors = self._door_objects(start_metadata, before_metadata, after_metadata)
        if selected_door_object_id:
            selected_door_object_id = str(selected_door_object_id)
            doors = [
                item
                for item in doors
                if str(item.get("objectId") or "") == selected_door_object_id
            ]
        if not doors:
            return None
        selected_door_object_id = str(selected_door_object_id or "")
        before_position = self._metadata_agent_position(before_metadata)
        after_position = self._metadata_agent_position(after_metadata)
        start_position = self._metadata_agent_position(start_metadata)
        normalized_instruction = instruction.lower()
        requested_relation = (
            "right"
            if "right" in normalized_instruction or "右" in normalized_instruction
            else None
        )
        candidates: list[dict[str, Any]] = []
        for door in doors:
            relation_evidence = self._door_relation_to_agent(
                door=door,
                start_metadata=start_metadata,
                requested_relation=requested_relation,
            )
            geometries = self._door_threshold_geometries(
                door,
                before_position=before_position,
                after_position=after_position,
            )
            for geometry in geometries:
                step_crossing = self._threshold_crossing(
                    start=before_position,
                    end=after_position,
                    geometry=geometry,
                )
                episode_crossing = self._threshold_crossing(
                    start=start_position,
                    end=after_position,
                    geometry=geometry,
                )
                axis, threshold = self._dominant_threshold_axis(geometry)
                door_object_id = str(door.get("objectId") or "")
                door_selection_verified = (
                    bool(selected_door_object_id)
                    and door_object_id == selected_door_object_id
                )
                threshold_distance = (
                    abs(float(step_crossing["center_distance"]))
                    if step_crossing["center_distance"] is not None
                    else min(
                        abs(float(step_crossing["start_signed_distance"])),
                        abs(float(step_crossing["end_signed_distance"])),
                    )
                )
                endpoint_a = None
                endpoint_b = None
                if geometry.half_length is not None:
                    endpoint_a = {
                        "x": geometry.center[0]
                        - geometry.tangent[0] * geometry.half_length,
                        "z": geometry.center[1]
                        - geometry.tangent[1] * geometry.half_length,
                    }
                    endpoint_b = {
                        "x": geometry.center[0]
                        + geometry.tangent[0] * geometry.half_length,
                        "z": geometry.center[1]
                        + geometry.tangent[1] * geometry.half_length,
                    }
                candidates.append(
                    {
                        "doorObjectId": door_object_id,
                        "doorObjectType": str(door.get("objectType") or "Door"),
                        "axis": axis,
                        "threshold": threshold,
                        "threshold_geometry": {
                            "source": geometry.source,
                            "center": {
                                "x": geometry.center[0],
                                "z": geometry.center[1],
                            },
                            "endpoint_a": endpoint_a,
                            "endpoint_b": endpoint_b,
                            "normal": {
                                "x": geometry.normal[0],
                                "z": geometry.normal[1],
                            },
                            "tangent": {
                                "x": geometry.tangent[0],
                                "z": geometry.tangent[1],
                            },
                            "half_length": geometry.half_length,
                            "finite_segment": geometry.half_length is not None,
                        },
                        "start_position": start_position,
                        "before_agent_pose": before_position,
                        "after_agent_pose": after_position,
                        "start_side": episode_crossing["start_side"],
                        "before_side": step_crossing["start_side"],
                        "after_side": step_crossing["end_side"],
                        "crossed_threshold": bool(step_crossing["crossed"]),
                        "step_crossed_threshold": bool(step_crossing["crossed"]),
                        "episode_crossed_threshold": bool(
                            episode_crossing["crossed"]
                        ),
                        "line_crossed": bool(step_crossing["line_crossed"]),
                        "within_threshold_segment": bool(
                            step_crossing["within_threshold_segment"]
                        ),
                        "intersection": step_crossing["intersection"],
                        "before_signed_distance": step_crossing[
                            "start_signed_distance"
                        ],
                        "after_signed_distance": step_crossing[
                            "end_signed_distance"
                        ],
                        "requested_relation": requested_relation,
                        "relation_to_agent": relation_evidence[
                            "relation_to_agent"
                        ],
                        "relation_verified": relation_evidence[
                            "relation_verified"
                        ],
                        "relation_score": relation_evidence["relation_score"],
                        "relation_frame": relation_evidence["relation_frame"],
                        "relation_reason": relation_evidence["relation_reason"],
                        "door_selection_verified": door_selection_verified,
                        "selectedDoorObjectId": selected_door_object_id or None,
                        "selection_source": (
                            "agent_selected_door"
                            if selected_door_object_id
                            else "unbound_candidate_door"
                        ),
                        "source": "ai2thor_agent_pose_and_door_metadata",
                        "distance_from_threshold": threshold_distance,
                    }
                )
        if not candidates:
            return None
        candidates.sort(
            key=lambda item: (
                not bool(item["door_selection_verified"]),
                not bool(item["crossed_threshold"]),
                str((item["threshold_geometry"] or {}).get("source"))
                != "ai2thor_door_bounding_box",
                float(item["distance_from_threshold"]),
                str(item["doorObjectId"]),
            )
        )
        selected = dict(candidates[0])
        selected.pop("distance_from_threshold", None)
        return selected

    def _selected_exit_door_object_id(
        self,
        *,
        response: dict[str, Any],
        bound_target_object_id: str | None,
        before_metadata: dict[str, Any],
        after_metadata: dict[str, Any],
    ) -> str | None:
        door_ids = {
            str(item.get("objectId") or "")
            for item in self._door_objects(before_metadata, after_metadata)
            if item.get("objectId")
        }

        def verified_door_id(
            value: Any,
            *,
            explicit_door_key: bool = False,
        ) -> str | None:
            object_id = str(value or "").strip()
            if not object_id:
                return None
            if object_id in door_ids:
                return object_id
            if explicit_door_key and "door" in object_id.lower():
                return object_id
            return None

        selected = verified_door_id(bound_target_object_id)
        if selected:
            return selected

        containers: list[dict[str, Any]] = []
        for container in (
            (response.get("action") or {}).get("args"),
            (response.get("skill_call") or {}).get("args"),
            response.get("interaction_binding") or {},
            (response.get("interaction_binding") or {}).get("args"),
            (response.get("interaction_binding") or {}).get("target_object"),
            response.get("observation") or {},
            (response.get("observation") or {}).get("best_candidate"),
        ):
            if isinstance(container, dict):
                containers.append(container)

        candidates = (response.get("observation") or {}).get("candidates")
        if isinstance(candidates, list):
            containers.extend(item for item in candidates if isinstance(item, dict))

        id_keys = (
            "doorObjectId",
            "door_object_id",
            "selectedDoorObjectId",
            "selected_door_object_id",
            "targetDoorObjectId",
            "target_door_object_id",
            "targetObjectId",
            "target_object_id",
            "objectId",
            "object_id",
            "clicked_object_id",
        )
        for container in containers:
            type_hint = " ".join(
                str(container.get(key) or "")
                for key in ("label", "object_type", "objectType", "name")
            ).lower()
            for key in id_keys:
                explicit_door_key = "door" in key.lower()
                selected = verified_door_id(
                    container.get(key),
                    explicit_door_key=explicit_door_key,
                )
                if selected and (
                    explicit_door_key or "door" in type_hint or selected in door_ids
                ):
                    return selected
        return None

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
        map_font = load_render_font(14)
        map_small_font = load_render_font(12)

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
                    font=map_small_font,
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
            font=map_font,
        )
        action_text = (
            f" | after {planned_action}" if planned_action else ""
        )
        draw.text(
            (22, 38),
            f"heading {heading:.0f} deg{action_text}",
            fill=(100, 240, 220),
            font=map_small_font,
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
        map_font = load_render_font(14)
        map_small_font = load_render_font(12)
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
                draw.text(
                    (px + 11, py - 8),
                    f"Target: {label}"[:28],
                    fill=(135, 35, 25),
                    font=map_small_font,
                )

        ax, ay = project(
            float(agent_position.get("x", 0.0)),
            float(agent_position.get("z", 0.0)),
        )
        heading = float(agent.get("rotation", {}).get("y", 0.0)) % 360.0
        nose, left, right = self._heading_triangle(ax, ay, heading)
        draw.polygon([nose, left, right], fill=(0, 126, 167), outline=(0, 70, 96))
        draw.text(
            (24, 24),
            f"AI2-THOR {self.scene} reachable map",
            fill=(15, 23, 42),
            font=map_font,
        )
        action_suffix = f" after {planned_action}" if planned_action else ""
        draw.text(
            (24, 44),
            f"Heading {heading:.0f} deg{action_suffix} (0 deg = +Z / up)",
            fill=(55, 65, 78),
            font=map_small_font,
        )
        draw.text(
            (24, 64),
            "gray reachable | teal path | blue robot | red target",
            fill=(55, 65, 78),
            font=map_small_font,
        )
        if best_candidate:
            draw.text(
                (24, 486),
                f"Best candidate: {best_candidate.get('label')} {best_candidate.get('confidence')}",
                fill=(170, 35, 24),
                font=map_small_font,
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
        title_font = load_render_font(24)
        body_font = load_render_font(18)
        small_font = load_render_font(16)
        draw.text(
            (34, 28),
            f"AI2-THOR x Agent | {self.scene}",
            fill=(245, 248, 252),
            font=title_font,
        )
        self._wrapped_text(
            draw,
            f"Instruction: {instruction}",
            34,
            62,
            1480,
            fill=(180, 195, 210),
            font=body_font,
            line_height=24,
            max_lines=1,
        )
        draw.text(
            (34, 92),
            self._post_action_observation_label(response),
            fill=(57, 217, 198),
            font=small_font,
        )
        canvas.paste(obs.resize((900, 506)), (34, 116))
        canvas.paste(topdown.resize((420, 420)), (980, 116))
        draw.rectangle([34, 116, 934, 622], outline=(57, 217, 198), width=3)
        draw.rectangle([980, 116, 1400, 536], outline=(88, 166, 255), width=3)
        panel_x = 980
        panel_y = 570
        draw.rounded_rectangle([panel_x, panel_y, 1560, 850], radius=8, fill=(242, 246, 250))
        draw.text(
            (panel_x + 20, panel_y + 18),
            f"Decision before action | Step {step_id}",
            fill=(15, 23, 42),
            font=body_font,
        )
        draw.text(
            (panel_x + 20, panel_y + 52),
            f"Selected action: {response['action']['type']}",
            fill=(0, 126, 120),
            font=body_font,
        )
        draw.text(
            (panel_x + 20, panel_y + 86),
            f"Confidence: {response['confidence']:.3f}",
            fill=(190, 45, 35),
            font=small_font,
        )
        self._wrapped_text(
            draw,
            "Thought: " + response["thought"],
            panel_x + 20,
            panel_y + 126,
            540,
            fill=(22, 31, 43),
            font=small_font,
            line_height=21,
            max_lines=7,
        )
        visible = ", ".join(visible_objects[:8]) or "None"
        self._wrapped_text(
            draw,
            "Visible: " + visible,
            34,
            656,
            900,
            fill=(220, 230, 242),
            font=small_font,
            line_height=22,
            max_lines=4,
        )
        return canvas

    @staticmethod
    def _post_action_observation_label(
        response: dict[str, Any],
    ) -> str:
        action = str(response["action"]["type"])
        if action in {"STOP", "Done", "ASK_CLARIFY"}:
            return (
                "Observation after action decision: "
                f"{action} (no simulator transition)"
            )
        execution = response.get("execution")
        if isinstance(execution, dict):
            postcondition = execution.get("postcondition")
            if execution.get("success") is False or (
                isinstance(postcondition, dict)
                and postcondition.get("passed") is False
            ):
                return f"Observation after action attempt: {action} (failed)"
        return f"Observation after action: {action}"

    def _wrapped_text(
        self,
        draw: ImageDraw.ImageDraw,
        text: str,
        x: int,
        y: int,
        max_width: int,
        fill: tuple[int, int, int],
        *,
        font: ImageFont.ImageFont,
        line_height: int,
        max_lines: int,
    ) -> None:
        lines: list[str] = []
        current = ""
        units = re.findall(
            r"[A-Za-z0-9_./:+-]+|\s+|.",
            str(text),
            flags=re.DOTALL,
        )
        for unit in units:
            if unit == "\n":
                lines.append(current.rstrip())
                current = ""
                continue
            if unit.isspace():
                unit = " "
            candidate = current + unit
            left, _, right, _ = draw.textbbox(
                (0, 0),
                candidate,
                font=font,
            )
            if current and right - left > max_width:
                lines.append(current.rstrip())
                current = unit.lstrip()
            else:
                current = candidate
        if current:
            lines.append(current.rstrip())

        visible_lines = lines[:max_lines]
        if len(lines) > max_lines and visible_lines:
            visible_lines[-1] = visible_lines[-1].rstrip(" .") + "..."
        for line in visible_lines:
            draw.text((x, y), line, fill=fill, font=font)
            y += line_height

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
