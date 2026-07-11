from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Callable

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.simulation.ai2thor_interactions import AI2ThorInteractionResolver
from src.simulation.ai2thor_postconditions import AI2ThorPostconditionVerifier
from src.simulation.ai2thor_runtime import (
    create_controller_safely,
    execute_controller_action,
    should_snap_to_grid,
)
from src.task.config import ROOT


RIGHT_DOOR_INSTRUCTION = "找到右边的门，然后走出去"
RIGHT_DOOR_SCENE = "FloorPlan402"
RIGHT_DOOR_ID = "ShowerDoor|-00.28|+01.23|+01.73"
RIGHT_DOOR_START_POSE = {
    "x": -0.75,
    "y": 0.9006701707839966,
    "z": 1.25,
    "rotation": 0.0,
    "horizon": 0.0,
    "standing": True,
}

VASE_BOX_INSTRUCTION = "把花瓶放到纸箱里"
VASE_BOX_SCENE = "FloorPlan203"
VASE_CANDIDATE_IDS = (
    "Vase|-04.27|+00.76|-00.44",
)
BOX_CANDIDATE_IDS = ("Box|+00.96|+00.29|+06.19",)


def object_position_from_id(object_id: str) -> dict[str, float]:
    parts = object_id.split("|")
    if len(parts) < 4:
        raise ValueError(f"object id lacks coordinate suffix: {object_id}")
    try:
        x, y, z = (float(value) for value in parts[-3:])
    except ValueError as exc:
        raise ValueError(f"object id has invalid coordinates: {object_id}") from exc
    return {"x": x, "y": y, "z": z}


def _agent_position(metadata: dict[str, Any]) -> dict[str, float]:
    agent = metadata.get("agent") or {}
    position = agent.get("position") or {}
    return {
        "x": float(position.get("x", 0.0)),
        "y": float(position.get("y", 0.0)),
        "z": float(position.get("z", 0.0)),
    }


def _side(value: float, threshold: float) -> int:
    if value < threshold:
        return -1
    if value > threshold:
        return 1
    return 0


def door_crossing_evidence(
    *,
    door_object_id: str,
    start_metadata: dict[str, Any],
    final_metadata: dict[str, Any],
    axis: str = "z",
) -> dict[str, Any]:
    door_position = object_position_from_id(door_object_id)
    if axis not in door_position:
        raise ValueError(f"unsupported crossing axis: {axis}")
    start_position = _agent_position(start_metadata)
    final_position = _agent_position(final_metadata)
    threshold = door_position[axis]
    start_side = _side(start_position[axis], threshold)
    final_side = _side(final_position[axis], threshold)
    crossed = start_side != 0 and final_side != 0 and start_side != final_side
    return {
        "doorObjectId": door_object_id,
        "axis": axis,
        "threshold": threshold,
        "start_position": start_position,
        "final_position": final_position,
        "start_side": start_side,
        "final_side": final_side,
        "crossed_threshold": crossed,
    }


def select_object(
    metadata: dict[str, Any],
    *,
    preferred_ids: tuple[str, ...],
    object_type: str,
    predicate: Callable[[dict[str, Any]], bool],
) -> dict[str, Any]:
    objects = [
        item
        for item in metadata.get("objects", [])
        if isinstance(item, dict)
        and str(item.get("objectType") or "").lower() == object_type.lower()
        and predicate(item)
    ]
    if not objects:
        raise RuntimeError(f"scene lacks usable {object_type}")
    by_id = {str(item.get("objectId")): item for item in objects}
    for object_id in preferred_ids:
        if object_id in by_id:
            return by_id[object_id]
    objects.sort(key=lambda item: str(item.get("objectId") or ""))
    return objects[0]


def _save_frame(event: Any, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(event.frame).convert("RGB").save(path)
    return str(path.relative_to(ROOT))


def _teleport_full(controller: Any, pose: dict[str, Any]) -> Any:
    event = controller.step(
        action="TeleportFull",
        position={"x": pose["x"], "y": pose["y"], "z": pose["z"]},
        rotation={"x": 0.0, "y": pose["rotation"], "z": 0.0},
        horizon=pose["horizon"],
        standing=pose["standing"],
    )
    if not event.metadata.get("lastActionSuccess"):
        raise RuntimeError(event.metadata.get("errorMessage") or "TeleportFull failed")
    return event


def _interactable_pose(controller: Any, object_id: str) -> dict[str, Any]:
    event = controller.step(
        action="GetInteractablePoses",
        objectId=object_id,
        maxPoses=96,
    )
    if not event.metadata.get("lastActionSuccess"):
        raise RuntimeError(event.metadata.get("errorMessage") or "GetInteractablePoses failed")
    poses = event.metadata.get("actionReturn") or []
    if not poses:
        raise RuntimeError(f"no interactable pose for {object_id}")
    standing = [pose for pose in poses if pose.get("standing") is True]
    return (standing or poses)[0]


def _execute_with_postcondition(
    *,
    controller: Any,
    verifier: AI2ThorPostconditionVerifier,
    action: str,
    args: dict[str, Any],
    frame_path: Path,
    allow_unchecked: bool = False,
) -> tuple[Any, dict[str, Any]]:
    execution = execute_controller_action(controller, action=action, args=args)
    postcondition = verifier.verify(
        action=execution.action,
        args=execution.args,
        before=execution.before_metadata,
        after=execution.after_metadata,
        runtime_success=execution.success,
    )
    frame = _save_frame(execution.event, frame_path)
    record = {
        "action": action,
        "args": args,
        "lastActionSuccess": execution.success,
        "errorMessage": execution.error_message,
        "postcondition": postcondition.to_dict(),
        "frame": frame,
    }
    if not execution.success:
        raise RuntimeError(json.dumps(record, ensure_ascii=False))
    if not allow_unchecked and not postcondition.passed:
        raise RuntimeError(json.dumps(record, ensure_ascii=False))
    return execution.event, record


def run_right_door_exit(output_dir: Path) -> dict[str, Any]:
    from ai2thor.controller import Controller
    from ai2thor.platform import CloudRendering

    output_dir.mkdir(parents=True, exist_ok=True)
    controller = None
    verifier = AI2ThorPostconditionVerifier()
    try:
        controller = create_controller_safely(
            Controller,
            scene=RIGHT_DOOR_SCENE,
            platform=CloudRendering,
            agentMode="default",
            width=960,
            height=540,
            quality="Low",
            gridSize=0.25,
            rotateStepDegrees=90.0,
            snapToGrid=should_snap_to_grid(
                mode="default",
                rotate_step_degrees=90.0,
            ),
            renderInstanceSegmentation=True,
        )
        start_event = _teleport_full(controller, RIGHT_DOOR_START_POSE)
        frames_dir = output_dir / "right_door_exit_frames"
        frames = [_save_frame(start_event, frames_dir / "00_inside_shower.png")]
        trace: list[dict[str, Any]] = []

        event, record = _execute_with_postcondition(
            controller=controller,
            verifier=verifier,
            action="OpenObject",
            args={"objectId": RIGHT_DOOR_ID},
            frame_path=frames_dir / "01_open_right_door.png",
        )
        trace.append(record)
        frames.append(record["frame"])

        for index in range(2):
            event = controller.step(action="MoveAhead")
            frame = _save_frame(event, frames_dir / f"{index + 2:02d}_move_ahead.png")
            record = {
                "action": "MoveAhead",
                "args": {},
                "lastActionSuccess": bool(event.metadata.get("lastActionSuccess")),
                "errorMessage": event.metadata.get("errorMessage") or "",
                "frame": frame,
            }
            if not record["lastActionSuccess"]:
                raise RuntimeError(json.dumps(record, ensure_ascii=False))
            trace.append(record)
            frames.append(frame)

        crossing = door_crossing_evidence(
            door_object_id=RIGHT_DOOR_ID,
            start_metadata=start_event.metadata,
            final_metadata=event.metadata,
        )
        if not crossing["crossed_threshold"]:
            raise RuntimeError(json.dumps({"door_crossing": crossing}, ensure_ascii=False))
        return {
            "status": "passed",
            "task_id": "right_door_exit",
            "scene": RIGHT_DOOR_SCENE,
            "instruction": RIGHT_DOOR_INSTRUCTION,
            "required_evidence": "door threshold crossing, not visual door detection",
            "door_crossing": crossing,
            "trace": trace,
            "frames": frames,
        }
    finally:
        if controller is not None:
            controller.stop()


def run_vase_into_box(output_dir: Path) -> dict[str, Any]:
    from ai2thor.controller import Controller
    from ai2thor.platform import CloudRendering

    output_dir.mkdir(parents=True, exist_ok=True)
    controller = None
    verifier = AI2ThorPostconditionVerifier()
    resolver = AI2ThorInteractionResolver()
    try:
        controller = create_controller_safely(
            Controller,
            scene=VASE_BOX_SCENE,
            platform=CloudRendering,
            agentMode="default",
            width=960,
            height=540,
            quality="Low",
            gridSize=0.25,
            rotateStepDegrees=90.0,
            snapToGrid=should_snap_to_grid(
                mode="default",
                rotate_step_degrees=90.0,
            ),
            renderInstanceSegmentation=True,
        )
        metadata = controller.last_event.metadata
        vase = select_object(
            metadata,
            preferred_ids=VASE_CANDIDATE_IDS,
            object_type="Vase",
            predicate=lambda item: bool(item.get("pickupable")),
        )
        box = select_object(
            metadata,
            preferred_ids=BOX_CANDIDATE_IDS,
            object_type="Box",
            predicate=lambda item: bool(item.get("receptacle")),
        )
        vase_id = str(vase["objectId"])
        box_id = str(box["objectId"])
        frames_dir = output_dir / "vase_into_box_frames"
        trace: list[dict[str, Any]] = []
        frames: list[str] = []

        if box.get("openable") and not box.get("isOpen"):
            _teleport_full(controller, _interactable_pose(controller, box_id))
            event, record = _execute_with_postcondition(
                controller=controller,
                verifier=verifier,
                action="OpenObject",
                args={"objectId": box_id},
                frame_path=frames_dir / "00_open_box.png",
            )
            trace.append(record)
            frames.append(record["frame"])

        _teleport_full(controller, _interactable_pose(controller, vase_id))
        pickup_binding = resolver.resolve(
            action="PickupObject",
            args={"objectType": "Vase"},
            instruction=VASE_BOX_INSTRUCTION,
            metadata=controller.last_event.metadata,
        )
        if not pickup_binding.valid:
            raise RuntimeError(json.dumps({"binding_errors": pickup_binding.errors}, ensure_ascii=False))
        event, record = _execute_with_postcondition(
            controller=controller,
            verifier=verifier,
            action="PickupObject",
            args=pickup_binding.args,
            frame_path=frames_dir / "01_pickup_vase.png",
        )
        record["binding"] = pickup_binding.to_dict()
        trace.append(record)
        frames.append(record["frame"])

        _teleport_full(controller, _interactable_pose(controller, box_id))
        put_binding = resolver.resolve(
            action="PutObject",
            args={"object": "Vase", "receptacleType": "Box"},
            instruction=VASE_BOX_INSTRUCTION,
            metadata=controller.last_event.metadata,
        )
        if not put_binding.valid:
            raise RuntimeError(json.dumps({"binding_errors": put_binding.errors}, ensure_ascii=False))
        event, record = _execute_with_postcondition(
            controller=controller,
            verifier=verifier,
            action="PutObject",
            args=put_binding.args,
            frame_path=frames_dir / "02_put_vase_in_box.png",
        )
        record["binding"] = put_binding.to_dict()
        trace.append(record)
        frames.append(record["frame"])

        final_objects = event.metadata.get("objects", [])
        final_vase = next(item for item in final_objects if item.get("objectId") == vase_id)
        final_box = next(item for item in final_objects if item.get("objectId") == box_id)
        final_inventory = event.metadata.get("inventoryObjects") or []
        placed = (
            vase_id in (final_box.get("receptacleObjectIds") or [])
            and box_id in (final_vase.get("parentReceptacles") or [])
            and final_inventory == []
        )
        if not placed:
            raise RuntimeError(
                json.dumps(
                    {
                        "vaseParentReceptacles": final_vase.get("parentReceptacles"),
                        "boxReceptacleObjectIds": final_box.get("receptacleObjectIds"),
                        "inventoryObjects": final_inventory,
                    },
                    ensure_ascii=False,
                )
            )
        return {
            "status": "passed",
            "task_id": "vase_into_box",
            "scene": VASE_BOX_SCENE,
            "instruction": VASE_BOX_INSTRUCTION,
            "required_evidence": "PickupObject and PutObject with strict receptacle postcondition",
            "scene_selection_note": (
                "FloorPlan211 contains a Box but Unity returns 'No valid positions "
                "to place object found' for both Vase objects, even with forceAction. "
                "FloorPlan203 is the verified AI2-THOR scene where Vase -> Box "
                "succeeds with native physics and strict metadata postconditions."
            ),
            "objects": {"vaseObjectId": vase_id, "boxObjectId": box_id},
            "final_state": {
                "vaseParentReceptacles": final_vase.get("parentReceptacles"),
                "boxReceptacleObjectIds": final_box.get("receptacleObjectIds"),
                "inventoryObjects": final_inventory,
            },
            "trace": trace,
            "frames": frames,
        }
    finally:
        if controller is not None:
            controller.stop()


def run_all(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = [
        run_right_door_exit(output_dir / "right_door_exit"),
        run_vase_into_box(output_dir / "vase_into_box"),
    ]
    result = {"status": "passed", "tasks": tasks}
    (output_dir / "validation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "docs" / "ai2thor_outputs" / "final_demo_validation",
    )
    args = parser.parse_args()
    result = run_all(args.output_dir.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
