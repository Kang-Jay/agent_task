from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image

from src.simulation.ai2thor_postconditions import AI2ThorPostconditionVerifier
from src.simulation.ai2thor_runtime import (
    create_controller_safely,
    should_snap_to_grid,
)
from src.task.config import ROOT


SCENE = "FloorPlan1"
FRIDGE_ID = "Fridge|-02.10|+00.00|+01.07"
EGG_ID = "Egg|-02.04|+00.81|+01.24"
BOWL_ID = "Bowl|+00.27|+01.10|-00.75"


def _object(metadata: dict[str, Any], object_id: str) -> dict[str, Any]:
    return next(
        item
        for item in metadata.get("objects", [])
        if item.get("objectId") == object_id
    )


def _interactable_pose(controller: Any, object_id: str) -> dict[str, Any]:
    event = controller.step(
        action="GetInteractablePoses",
        objectId=object_id,
        maxPoses=64,
    )
    if not event.metadata.get("lastActionSuccess"):
        raise RuntimeError(event.metadata.get("errorMessage"))
    poses = event.metadata.get("actionReturn") or []
    if not poses:
        raise RuntimeError(f"No interactable pose for {object_id}")
    return poses[0]


def _teleport_to_pose(controller: Any, pose: dict[str, Any]) -> Any:
    event = controller.step(
        action="TeleportFull",
        position={"x": pose["x"], "y": pose["y"], "z": pose["z"]},
        rotation={"x": 0.0, "y": pose["rotation"], "z": 0.0},
        horizon=pose["horizon"],
        standing=pose["standing"],
    )
    if not event.metadata.get("lastActionSuccess"):
        raise RuntimeError(event.metadata.get("errorMessage"))
    return event


def _save_frame(event: Any, path: Path) -> None:
    Image.fromarray(event.frame).convert("RGB").save(path)


def _execute_verified(
    *,
    controller: Any,
    verifier: AI2ThorPostconditionVerifier,
    action: str,
    args: dict[str, Any],
    before_event: Any,
    frame_path: Path,
) -> tuple[Any, dict[str, Any]]:
    event = controller.step(action=action, **args)
    postcondition = verifier.verify(
        action=action,
        args=args,
        before=before_event.metadata,
        after=event.metadata,
        runtime_success=bool(event.metadata.get("lastActionSuccess")),
    )
    _save_frame(event, frame_path)
    record = {
        "action": action,
        "args": args,
        "lastActionSuccess": bool(event.metadata.get("lastActionSuccess")),
        "errorMessage": event.metadata.get("errorMessage") or "",
        "postcondition": postcondition.to_dict(),
        "frame": str(frame_path.relative_to(ROOT)),
        "forceAction": False,
    }
    if not record["lastActionSuccess"] or not postcondition.passed:
        raise RuntimeError(json.dumps(record, ensure_ascii=False))
    return event, record


def run_validation(output_dir: Path) -> dict[str, Any]:
    from ai2thor.controller import Controller
    from ai2thor.platform import CloudRendering

    output_dir.mkdir(parents=True, exist_ok=True)
    controller = None
    verifier = AI2ThorPostconditionVerifier()
    rotate_step_degrees = 30.0
    try:
        controller = create_controller_safely(
            Controller,
            scene=SCENE,
            platform=CloudRendering,
            agentMode="default",
            width=960,
            height=540,
            quality="Low",
            gridSize=0.25,
            rotateStepDegrees=rotate_step_degrees,
            snapToGrid=should_snap_to_grid(
                mode="default",
                rotate_step_degrees=rotate_step_degrees,
            ),
            renderInstanceSegmentation=True,
        )
        initial_event = controller.last_event
        _save_frame(initial_event, output_dir / "00_initial.png")
        initial_fridge = _object(initial_event.metadata, FRIDGE_ID)
        initial_egg = _object(initial_event.metadata, EGG_ID)
        if initial_fridge.get("isOpen"):
            raise RuntimeError("Validation fixture requires the Fridge to start closed")
        if FRIDGE_ID not in (initial_egg.get("parentReceptacles") or []):
            raise RuntimeError("Validation fixture requires the Egg inside the Fridge")

        trace: list[dict[str, Any]] = []

        before_open = _teleport_to_pose(
            controller,
            _interactable_pose(controller, FRIDGE_ID),
        )
        open_event, record = _execute_verified(
            controller=controller,
            verifier=verifier,
            action="OpenObject",
            args={"objectId": FRIDGE_ID},
            before_event=before_open,
            frame_path=output_dir / "01_open_fridge.png",
        )
        trace.append(record)

        before_pickup = _teleport_to_pose(
            controller,
            _interactable_pose(controller, EGG_ID),
        )
        pickup_event, record = _execute_verified(
            controller=controller,
            verifier=verifier,
            action="PickupObject",
            args={"objectId": EGG_ID},
            before_event=before_pickup,
            frame_path=output_dir / "02_pickup_egg.png",
        )
        trace.append(record)

        before_put = _teleport_to_pose(
            controller,
            _interactable_pose(controller, BOWL_ID),
        )
        put_event, record = _execute_verified(
            controller=controller,
            verifier=verifier,
            action="PutObject",
            args={"objectId": BOWL_ID},
            before_event=before_put,
            frame_path=output_dir / "03_put_egg_in_bowl.png",
        )
        trace.append(record)

        final_egg = _object(put_event.metadata, EGG_ID)
        final_bowl = _object(put_event.metadata, BOWL_ID)
        result = {
            "status": "passed",
            "scene": SCENE,
            "runtime_configuration": {
                "width": 960,
                "height": 540,
                "rotateStepDegrees": rotate_step_degrees,
                "snapToGrid": False,
                "renderInstanceSegmentation": True,
            },
            "dependency": "Egg starts inside a closed Fridge",
            "chain": ["OpenObject", "PickupObject", "PutObject"],
            "setup_actions": ["GetInteractablePoses", "TeleportFull"],
            "all_interaction_force_action_false": True,
            "initial_state": {
                "fridgeIsOpen": initial_fridge.get("isOpen"),
                "eggParentReceptacles": initial_egg.get("parentReceptacles"),
            },
            "final_state": {
                "eggParentReceptacles": final_egg.get("parentReceptacles"),
                "bowlReceptacleObjectIds": final_bowl.get(
                    "receptacleObjectIds"
                ),
                "inventoryObjects": put_event.metadata.get(
                    "inventoryObjects", []
                ),
            },
            "trace": trace,
        }
        summary_path = output_dir / "validation.json"
        summary_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return result
    finally:
        if controller is not None:
            controller.stop()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT
        / "docs"
        / "ai2thor_outputs"
        / "interaction_chain_validation",
    )
    args = parser.parse_args()
    result = run_validation(args.output_dir.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
