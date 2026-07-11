from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Callable

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.simulation.ai2thor_adapter import AI2ThorVisualSearchDemo
from src.task.config import ROOT


DEFAULT_OUTPUT_DIR = ROOT / "docs" / "ai2thor_outputs" / "final_agent_demos"


@dataclass(frozen=True)
class FinalDemoSpec:
    task_id: str
    scene: str
    instruction: str
    max_steps: int
    expected_outcome: str = "exact_success"
    initial_pose: dict[str, Any] | None = None


FINAL_DEMO_SPECS: tuple[FinalDemoSpec, ...] = (
    FinalDemoSpec(
        task_id="television",
        scene="FloorPlan211",
        instruction="Find the television in the room",
        max_steps=12,
    ),
    FinalDemoSpec(
        task_id="vase_into_box",
        scene="FloorPlan203",
        instruction="把花瓶放到纸箱里",
        max_steps=20,
    ),
    FinalDemoSpec(
        task_id="right_door_exit",
        scene="FloorPlan402",
        instruction="找到右边的门，然后走出去",
        max_steps=20,
    ),
    FinalDemoSpec(
        task_id="sofa_sit",
        scene="FloorPlan211",
        instruction="找到房间里的沙发并坐下",
        max_steps=20,
        expected_outcome="approximate_success",
    ),
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _relative_or_absolute(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _serializable_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def verify_video(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"video is missing or empty: {path}")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"video is not decodable: {path}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    readable = 0
    last_frame_ok = False
    while True:
        ok, _frame = capture.read()
        if not ok:
            break
        readable += 1
        last_frame_ok = True
    capture.release()
    if frame_count <= 0 or readable <= 0 or not last_frame_ok:
        raise RuntimeError(f"video has no readable frames: {path}")
    return {
        "path": _serializable_path(path),
        "bytes": path.stat().st_size,
        "frame_count": frame_count,
        "readable_frames": readable,
        "width": width,
        "height": height,
        "decodable": True,
    }


def _require_ai2thor_steps(*, spec: FinalDemoSpec, steps: list[dict[str, Any]]) -> None:
    for index, step in enumerate(steps):
        if step.get("backend") != "ai2thor":
            raise RuntimeError(
                f"{spec.task_id}: step {index} backend is not strict AI2-THOR"
            )
        for field in ("frame_path", "observation_path", "topdown_path"):
            value = step.get(field)
            if not value:
                raise RuntimeError(f"{spec.task_id}: step {index} missing {field}")
            path = _relative_or_absolute(value)
            if not path.exists() or path.stat().st_size <= 0:
                raise RuntimeError(
                    f"{spec.task_id}: step {index} {field} is missing or empty: {path}"
                )


def _require_real_vlm_usage(*, spec: FinalDemoSpec, steps: list[dict[str, Any]]) -> None:
    for step in steps:
        if step.get("planner_source") != "model_planner":
            continue
        model_info = step.get("model_info") or {}
        if not bool(model_info.get("vision_input_used")):
            continue
        provider = str(model_info.get("provider") or model_info.get("provider_used") or "")
        model = str(model_info.get("model") or model_info.get("model_used") or "")
        status = str(model_info.get("status") or "")
        error_fields = [
            model_info.get("error"),
            model_info.get("errors"),
            model_info.get("parse_error"),
            model_info.get("exception"),
        ]
        if (
            status == "ok"
            and provider
            and model
            and not any(token in provider.lower() for token in ("fake", "mock", "test"))
            and not any(token in model.lower() for token in ("fake", "mock", "test"))
            and not any(error_fields)
        ):
            return
    raise RuntimeError(f"{spec.task_id}: no successful real VLM vision call recorded")


def _require_closed_loop_trace(*, spec: FinalDemoSpec, steps: list[dict[str, Any]]) -> None:
    saw_agent_decision = False
    saw_ai2thor_execution = False
    saw_environment_feedback = False
    saw_verifier_status = False
    for index, step in enumerate(steps):
        if step.get("planner_source"):
            saw_agent_decision = True
        execution = step.get("execution")
        if isinstance(execution, dict):
            if execution.get("mode") and execution.get("actor") == "agent":
                saw_ai2thor_execution = True
            elif execution.get("action") and "success" in execution:
                saw_ai2thor_execution = True
        robot = step.get("robot")
        if (
            isinstance(robot, dict)
            and step.get("observation_path")
            and step.get("map_view_source")
        ):
            saw_environment_feedback = True
        completion = step.get("completion_status")
        if isinstance(completion, dict) and "evidence_ledger" in completion:
            saw_verifier_status = True
        if step.get("backend") != "ai2thor":
            raise RuntimeError(
                f"{spec.task_id}: step {index} did not use AI2-THOR feedback"
            )
    missing = []
    if not saw_agent_decision:
        missing.append("EmbodiedSearchAgent decision")
    if not saw_ai2thor_execution:
        missing.append("AI2-THOR executor result")
    if not saw_environment_feedback:
        missing.append("environment feedback")
    if not saw_verifier_status:
        missing.append("verifier evidence")
    if missing:
        raise RuntimeError(
            f"{spec.task_id}: closed loop trace missing {', '.join(missing)}"
        )


def _require_right_door_evidence(evidence: dict[str, Any]) -> None:
    if evidence.get("crossed_threshold") is not True:
        raise RuntimeError("right_door_exit: missing threshold crossing evidence")
    if not evidence.get("doorObjectId"):
        raise RuntimeError(
            "right_door_exit: missing runtime door object evidence"
        )
    if evidence.get("axis") not in {"x", "z"}:
        raise RuntimeError(
            "right_door_exit: crossing axis must come from door metadata, "
            f"got {evidence.get('axis')}"
        )
    if not isinstance(evidence.get("threshold"), (int, float)):
        raise RuntimeError("right_door_exit: missing runtime door threshold")
    for pose_field in ("start_position", "before_agent_pose", "after_agent_pose"):
        pose = evidence.get(pose_field) or {}
        if not all(isinstance(pose.get(key), (int, float)) for key in ("x", "y", "z")):
            raise RuntimeError(
                f"right_door_exit: missing {pose_field} runtime pose"
            )
    selected_door_id = evidence.get("selectedDoorObjectId")
    if not selected_door_id or selected_door_id != evidence.get("doorObjectId"):
        raise RuntimeError("right_door_exit: runtime right door selection not verified")
    if evidence.get("door_selection_verified") is not True:
        raise RuntimeError("right_door_exit: runtime right door selection not verified")
    if evidence.get("source") != "ai2thor_agent_pose_and_door_metadata":
        raise RuntimeError("right_door_exit: unexpected exit evidence source")


def _require_television_target_evidence(completion: dict[str, Any]) -> None:
    if not (
        completion.get("target_located")
        or completion.get("target_visible")
        or completion.get("target_visible_in_environment")
        or completion.get("approach_verified")
    ):
        raise RuntimeError("television: missing target evidence")


def _require_vase_into_box_evidence(
    *,
    completion: dict[str, Any],
    steps: list[dict[str, Any]],
) -> None:
    final_state = completion.get("final_state") or {}
    vase_id = str(final_state.get("vaseObjectId") or "")
    box_id = str(final_state.get("boxObjectId") or "")
    vase_parents = set(map(str, final_state.get("vaseParentReceptacles") or []))
    box_contents = set(map(str, final_state.get("boxReceptacleObjectIds") or []))
    inventory = final_state.get("inventoryObjects") or []
    if (
        not vase_id
        or not box_id
        or box_id not in vase_parents
        or vase_id not in box_contents
        or inventory
    ):
        raise RuntimeError("vase_into_box: missing receptacle postcondition final state")

    saw_pickup_postcondition = False
    saw_put_postcondition = False
    for step in steps:
        execution = step.get("execution") or {}
        if not isinstance(execution, dict):
            continue
        action = str(execution.get("action") or step.get("action") or "")
        postcondition = execution.get("postcondition") or {}
        evidence = postcondition.get("evidence") or {}
        if action == "PickupObject" and postcondition.get("passed") is True:
            inventory_ids = set(map(str, evidence.get("inventoryObjectIds") or []))
            saw_pickup_postcondition = vase_id in inventory_ids
        if action == "PutObject" and postcondition.get("passed") is True:
            placed_ids = set(map(str, evidence.get("placedObjectIds") or []))
            receptacle_ids = set(map(str, evidence.get("receptacleObjectIds") or []))
            released_ids = set(map(str, evidence.get("releasedObjectIds") or []))
            saw_put_postcondition = (
                str(evidence.get("receptacleObjectId") or "") == box_id
                and vase_id in (placed_ids | receptacle_ids | released_ids)
            )
    if not saw_pickup_postcondition or not saw_put_postcondition:
        raise RuntimeError("vase_into_box: missing receptacle postcondition")


def _require_sofa_sit_evidence(completion: dict[str, Any]) -> None:
    actions = set(completion.get("successful_actions") or [])
    if not completion.get("approach_verified") or "Crouch" not in actions:
        raise RuntimeError("sofa_sit: approach and Crouch evidence are required")


def verify_demo_summary(
    *,
    spec: FinalDemoSpec,
    summary: dict[str, Any],
    expected_summary_path: Path | None = None,
    expected_video_path: Path | None = None,
) -> dict[str, Any]:
    if summary.get("episode_id") != spec.task_id:
        raise RuntimeError(
            f"{spec.task_id}: summary episode_id mismatch {summary.get('episode_id')}"
        )
    if expected_summary_path is not None:
        summary_path_value = _relative_or_absolute(summary.get("summary_path") or "")
        if summary_path_value.resolve() != expected_summary_path.resolve():
            raise RuntimeError(f"{spec.task_id}: summary_path does not match current run")
    if expected_video_path is not None:
        video_path_value = _relative_or_absolute(summary.get("video_path") or "")
        if video_path_value.resolve() != expected_video_path.resolve():
            raise RuntimeError(f"{spec.task_id}: video_path does not match current run")

    steps = summary.get("steps")
    if not isinstance(steps, list) or not steps:
        raise RuntimeError(f"{spec.task_id}: summary has no steps")
    _require_ai2thor_steps(spec=spec, steps=steps)
    _require_real_vlm_usage(spec=spec, steps=steps)
    _require_closed_loop_trace(spec=spec, steps=steps)

    final_step = steps[-1]
    if final_step.get("scene") != spec.scene:
        raise RuntimeError(f"{spec.task_id}: final step scene mismatch")
    if final_step.get("done") is not True:
        raise RuntimeError(f"{spec.task_id}: final step did not terminate")

    completion = final_step.get("completion_status") or {}
    if completion.get("complete") is not True:
        raise RuntimeError(
            f"{spec.task_id}: completion not verified: {completion.get('reason')}"
        )
    if completion.get("outcome") != spec.expected_outcome:
        raise RuntimeError(
            f"{spec.task_id}: expected outcome {spec.expected_outcome}, "
            f"got {completion.get('outcome')}"
        )
    if spec.task_id == "right_door_exit":
        _require_right_door_evidence(completion.get("exit_evidence") or {})
    if spec.task_id == "vase_into_box":
        actions = set(completion.get("successful_actions") or [])
        if not {"PickupObject", "PutObject"}.issubset(actions):
            raise RuntimeError(f"{spec.task_id}: missing pickup/put success actions")
        _require_vase_into_box_evidence(completion=completion, steps=steps)
    if spec.task_id == "television":
        _require_television_target_evidence(completion)
    if spec.task_id == "sofa_sit" and completion.get("completion_mode") != "approximate_sit":
        raise RuntimeError(f"{spec.task_id}: sofa task must be marked approximate_sit")
    if spec.task_id == "sofa_sit":
        _require_sofa_sit_evidence(completion)

    return {
        "step_count": len(steps),
        "final_action": final_step.get("action"),
        "final_done": final_step.get("done"),
        "completion_status": completion,
        "vision_input_used": True,
    }


def run_one_demo(
    spec: FinalDemoSpec,
    *,
    output_dir: Path,
    demo_factory: Callable[[FinalDemoSpec], AI2ThorVisualSearchDemo] | None = None,
) -> dict[str, Any]:
    session_id = f"final-agent-demos-{spec.task_id}"
    episode_id = spec.task_id
    demo = (
        demo_factory(spec)
        if demo_factory is not None
        else AI2ThorVisualSearchDemo(scene=spec.scene)
    )
    result = demo.run_demo(
        instruction=spec.instruction,
        max_steps=spec.max_steps,
        session_id=session_id,
        episode_id=episode_id,
        initial_pose=spec.initial_pose,
    )
    summary_path = _relative_or_absolute(result.summary_path)
    video_path = _relative_or_absolute(result.video_path)
    summary = _read_json(summary_path)
    summary_verification = verify_demo_summary(
        spec=spec,
        summary=summary,
        expected_summary_path=summary_path,
        expected_video_path=video_path,
    )
    video_verification = verify_video(video_path)
    if int(video_verification["readable_frames"]) < int(summary_verification["step_count"]):
        raise RuntimeError(
            f"{spec.task_id}: video readable frame count is below summary step count"
        )
    record = {
        "task_id": spec.task_id,
        "scene": spec.scene,
        "instruction": spec.instruction,
        "status": "passed",
        "summary_path": _serializable_path(summary_path),
        "video_path": _serializable_path(video_path),
        "summary_verification": summary_verification,
        "video_verification": video_verification,
    }
    task_output_dir = output_dir / spec.task_id
    task_output_dir.mkdir(parents=True, exist_ok=True)
    (task_output_dir / "verification.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return record


def run_all(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    task_ids: set[str] | None = None,
    demo_factory: Callable[[FinalDemoSpec], AI2ThorVisualSearchDemo] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = [
        spec for spec in FINAL_DEMO_SPECS if task_ids is None or spec.task_id in task_ids
    ]
    if not selected:
        raise ValueError("no final demo tasks selected")
    tasks = [
        run_one_demo(spec, output_dir=output_dir, demo_factory=demo_factory)
        for spec in selected
    ]
    result = {"status": "passed", "tasks": tasks}
    (output_dir / "final_agent_demos_manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--task",
        action="append",
        choices=[spec.task_id for spec in FINAL_DEMO_SPECS],
        help="Run only one task. Repeat to select multiple tasks.",
    )
    args = parser.parse_args()
    result = run_all(
        output_dir=args.output_dir.resolve(),
        task_ids=set(args.task) if args.task else None,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
