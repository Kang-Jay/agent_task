from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image

from src.agent.task_semantics import TaskSemantics
from src.simulation.ai2thor_interactions import AI2ThorInteractionResolver
from src.simulation.ai2thor_postconditions import AI2ThorPostconditionVerifier
from src.simulation.ai2thor_runtime import (
    create_controller_safely,
    should_snap_to_grid,
)
from src.simulation.task_verifier import TaskVerifier
from src.task.config import ROOT


SCENE = "FloorPlan211"
INSTRUCTION = "找到房间里的沙发并坐下"


def _select_sofa(metadata: dict[str, Any]) -> dict[str, Any]:
    sofas = [
        item
        for item in metadata.get("objects", [])
        if item.get("objectType") == "Sofa"
    ]
    if not sofas:
        raise RuntimeError("Scene does not contain a Sofa")
    sofas.sort(key=lambda item: str(item.get("objectId") or ""))
    return sofas[0]


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


def run_validation(output_dir: Path) -> dict[str, Any]:
    from ai2thor.controller import Controller
    from ai2thor.platform import CloudRendering

    output_dir.mkdir(parents=True, exist_ok=True)
    controller = None
    postconditions = AI2ThorPostconditionVerifier()
    task_verifier = TaskVerifier()
    task_plan = TaskSemantics().analyze(INSTRUCTION, mode="default")
    resolver = AI2ThorInteractionResolver()
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
        sofa = _select_sofa(controller.last_event.metadata)
        sofa_id = str(sofa["objectId"])
        before_crouch = _teleport_to_pose(
            controller,
            _interactable_pose(controller, sofa_id),
        )
        _save_frame(before_crouch, output_dir / "00_approached_sofa.png")

        context_before = resolver.build_context(before_crouch.metadata)
        visible_sofa_before = next(
            (
                item
                for item in context_before["objects"]
                if item.get("objectId") == sofa_id and item.get("visible")
            ),
            None,
        )
        if visible_sofa_before is None or visible_sofa_before.get("distance") is None:
            raise RuntimeError("Sofa approach lacks visible finite-distance evidence")

        crouch_event = controller.step(action="Crouch")
        postcondition = postconditions.verify(
            action="Crouch",
            args={},
            before=before_crouch.metadata,
            after=crouch_event.metadata,
            runtime_success=bool(
                crouch_event.metadata.get("lastActionSuccess")
            ),
        )
        _save_frame(crouch_event, output_dir / "01_crouched_near_sofa.png")
        context_after = resolver.build_context(crouch_event.metadata)
        verification = task_verifier.verify(
            task_plan,
            steps=[
                {
                    "executed_action": {"type": "Crouch", "args": {}},
                    "action_success": postcondition.passed,
                }
            ],
            target_visible=True,
            confidence=1.0,
            stop_confidence_threshold=1.0,
            environment_context=context_after,
        )
        if not postcondition.passed or verification.outcome != "approximate_success":
            raise RuntimeError(
                json.dumps(
                    {
                        "postcondition": postcondition.to_dict(),
                        "verification": verification.to_dict(),
                    },
                    ensure_ascii=False,
                )
            )

        result = {
            "status": "passed",
            "scene": SCENE,
            "instruction": INSTRUCTION,
            "completion_mode": task_plan.completion_mode,
            "limitation": "native_sit_on_furniture_state_unavailable",
            "approximation": "approach Sofa and execute Crouch",
            "sofa": {
                "objectId": sofa_id,
                "distanceBeforeCrouch": visible_sofa_before["distance"],
            },
            "crouch": {
                "lastActionSuccess": bool(
                    crouch_event.metadata.get("lastActionSuccess")
                ),
                "postcondition": postcondition.to_dict(),
                "agentIsStanding": (
                    crouch_event.metadata.get("agent") or {}
                ).get("isStanding"),
            },
            "taskVerification": verification.to_dict(),
            "frames": [
                str(
                    (output_dir / "00_approached_sofa.png").relative_to(ROOT)
                ),
                str(
                    (output_dir / "01_crouched_near_sofa.png").relative_to(ROOT)
                ),
            ],
        }
        (output_dir / "validation.json").write_text(
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
        default=(
            ROOT
            / "docs"
            / "ai2thor_outputs"
            / "sofa_approximation_validation"
        ),
    )
    args = parser.parse_args()
    result = run_validation(args.output_dir.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
