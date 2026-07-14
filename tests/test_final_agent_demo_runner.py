from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from tools import run_final_agent_demos
from tools.run_final_agent_demos import (
    FINAL_DEMO_SPECS,
    FinalDemoSpec,
    run_all,
    verify_demo_summary,
)


@dataclass
class _FakeDemoResult:
    summary_path: str
    video_path: str


def _write_artifact(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"artifact")
    return str(path)


def _real_model_info() -> dict[str, object]:
    return {
        "status": "ok",
        "provider": "openai",
        "model": "gpt-4o",
        "vision_input_used": True,
    }


def _spec(task_id: str) -> FinalDemoSpec:
    return next(spec for spec in FINAL_DEMO_SPECS if spec.task_id == task_id)


def _base_completion(spec: FinalDemoSpec) -> dict[str, object]:
    return {
        "complete": True,
        "outcome": spec.expected_outcome,
        "reason": "verified",
        "successful_actions": [],
        "completion_mode": "exact",
        "evidence_ledger": [{"predicate": "fixture", "passed": True}],
    }


def _selected_right_door_evidence(
    *,
    door_object_id: str = "Door|selected-right|+02.00|+01.00|+03.50",
    selected_door_object_id: str | None = None,
    door_selection_verified: bool = True,
) -> dict[str, object]:
    return {
        "doorObjectId": door_object_id,
        "doorObjectType": "Door",
        "axis": "z",
        "threshold": 3.5,
        "start_position": {"x": 2.0, "y": 0.9, "z": 3.0},
        "before_agent_pose": {"x": 2.0, "y": 0.9, "z": 3.25},
        "after_agent_pose": {"x": 2.0, "y": 0.9, "z": 3.75},
        "start_side": -1,
        "before_side": -1,
        "after_side": 1,
        "crossed_threshold": True,
        "step_crossed_threshold": True,
        "requested_relation": "right",
        "relation_to_agent": "right",
        "relation_verified": True,
        "relation_frame": "agent_initial_heading",
        "door_selection_verified": door_selection_verified,
        "selectedDoorObjectId": selected_door_object_id or door_object_id,
        "selection_source": "agent_selected_door",
        "source": "ai2thor_agent_pose_and_door_metadata",
    }


def _passed_postcondition(evidence: dict[str, object]) -> dict[str, object]:
    return {
        "checked": True,
        "passed": True,
        "reason": "verified",
        "evidence": evidence,
    }


def _interaction_step(
    *,
    action: str,
    object_id: str,
    postcondition: dict[str, object] | None = None,
) -> dict[str, object]:
    execution: dict[str, object] = {
        "action": action,
        "args": {"objectId": object_id},
        "success": True,
        "mode": "default",
        "actor": "agent",
    }
    if postcondition is not None:
        execution["postcondition"] = postcondition
    return {"action": action, "execution": execution}


def _step_for(
    *,
    root: Path,
    spec: FinalDemoSpec,
    index: int,
    action: str = "Done",
    done: bool = False,
    completion_status: dict[str, object] | None = None,
    execution: dict[str, object] | None = None,
) -> dict[str, object]:
    artifact_dir = root / spec.task_id / str(index)
    return {
        "backend": "ai2thor",
        "scene": spec.scene,
        "frame_path": _write_artifact(artifact_dir / "frame.png"),
        "observation_path": _write_artifact(artifact_dir / "observation.png"),
        "topdown_path": _write_artifact(artifact_dir / "topdown.png"),
        "action": action,
        "done": done,
        "planner_source": "model_planner",
        "model_info": _real_model_info(),
        "completion_status": completion_status
        if completion_status is not None
        else {"complete": False, "outcome": "in_progress"},
        "execution": execution
        if execution is not None
        else {
            "action": action,
            "args": {},
            "success": True,
            "mode": "default",
            "actor": "agent",
        },
        "robot": {"x": float(index), "y": 0.0, "heading": 0.0},
        "map_view_source": "unity_third_party_camera",
    }


def _summary_for(
    *,
    root: Path,
    spec: FinalDemoSpec,
    completion: dict[str, object] | None = None,
    prior_steps: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    steps: list[dict[str, object]] = []
    for index, overrides in enumerate(prior_steps or []):
        steps.append(_step_for(root=root, spec=spec, index=index, **overrides))
    steps.append(
        _step_for(
            root=root,
            spec=spec,
            index=len(steps),
            action="Done",
            done=True,
            completion_status=completion or _base_completion(spec),
        )
    )
    return {"episode_id": spec.task_id, "steps": steps}


def _vase_completion(vase_id: str = "Vase|fixture", box_id: str = "Box|fixture") -> dict[str, object]:
    spec = _spec("vase_into_box")
    completion = _base_completion(spec)
    completion["successful_actions"] = ["PickupObject", "PutObject"]
    completion["final_state"] = {
        "vaseObjectId": vase_id,
        "boxObjectId": box_id,
        "vaseParentReceptacles": [box_id],
        "boxReceptacleObjectIds": [vase_id],
        "inventoryObjects": [],
    }
    return completion


def _vase_steps(vase_id: str = "Vase|fixture", box_id: str = "Box|fixture") -> list[dict[str, object]]:
    return [
        _interaction_step(
            action="PickupObject",
            object_id=vase_id,
            postcondition=_passed_postcondition({"inventoryObjectIds": [vase_id]}),
        ),
        _interaction_step(
            action="PutObject",
            object_id=box_id,
            postcondition=_passed_postcondition(
                {
                    "receptacleObjectId": box_id,
                    "releasedObjectIds": [vase_id],
                    "placedObjectIds": [vase_id],
                    "receptacleObjectIds": [vase_id],
                }
            ),
        ),
    ]


class _FakeDemo:
    def __init__(self, spec: FinalDemoSpec, root: Path):
        self.spec = spec
        self.root = root
        self.calls: list[dict[str, object]] = []

    def run_demo(self, **kwargs: object) -> _FakeDemoResult:
        self.calls.append(dict(kwargs))
        task_dir = self.root / self.spec.task_id
        summary_path = task_dir / "summary.json"
        video_path = task_dir / "demo.mp4"
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_path.write_bytes(b"fake-video")

        completion = _base_completion(self.spec)
        prior_steps: list[dict[str, object]] = []
        if self.spec.task_id == "television":
            completion["target_located"] = True
            completion["target_visible"] = True
        elif self.spec.task_id == "right_door_exit":
            completion["exit_evidence"] = _selected_right_door_evidence()
        elif self.spec.task_id == "vase_into_box":
            completion = _vase_completion()
            prior_steps = _vase_steps()
        elif self.spec.task_id == "sofa_sit":
            completion["completion_mode"] = "approximate_sit"
            completion["successful_actions"] = ["Crouch"]
            completion["target_located"] = True
            completion["approach_verified"] = True

        summary_path.write_text(
            json.dumps(
                {
                    **_summary_for(
                        root=self.root,
                        spec=self.spec,
                        completion=completion,
                        prior_steps=prior_steps,
                    ),
                    "video_path": str(video_path),
                    "summary_path": str(summary_path),
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return _FakeDemoResult(summary_path=str(summary_path), video_path=str(video_path))


class FinalAgentDemoRunnerTests(unittest.TestCase):
    def test_manifest_contains_required_four_tasks(self) -> None:
        self.assertEqual(
            {spec.task_id for spec in FINAL_DEMO_SPECS},
            {"television", "vase_into_box", "right_door_exit", "sofa_sit"},
        )
        scenes = {spec.task_id: spec.scene for spec in FINAL_DEMO_SPECS}
        self.assertEqual(scenes["right_door_exit"], "FloorPlan402")
        self.assertEqual(scenes["vase_into_box"], "FloorPlan203")
        self.assertTrue(_spec("vase_into_box").instruction)
        self.assertTrue(_spec("right_door_exit").instruction)

    def test_verify_demo_summary_rejects_missing_real_vlm_call(self) -> None:
        spec = _spec("television")
        with TemporaryDirectory() as temporary_directory:
            summary = _summary_for(root=Path(temporary_directory), spec=spec)
            summary["steps"][-1]["model_info"] = {"vision_input_used": False}
            with self.assertRaisesRegex(RuntimeError, "no successful real VLM vision call"):
                verify_demo_summary(spec=spec, summary=summary)

    def test_verify_demo_summary_rejects_oracle_grounding_in_strict_run(self) -> None:
        spec = _spec("television")
        completion = _base_completion(spec)
        completion["target_located"] = True
        completion["target_visible"] = True
        with TemporaryDirectory() as temporary_directory:
            summary = _summary_for(
                root=Path(temporary_directory),
                spec=spec,
                completion=completion,
            )
            summary["steps"][-1]["planner_source"] = "simulator_oracle"
            summary["steps"][-1]["model_info"] = _real_model_info()

            with self.assertRaisesRegex(
                RuntimeError,
                "forbidden planner_source=simulator_oracle",
            ):
                verify_demo_summary(spec=spec, summary=summary)


    def test_verify_demo_summary_accepts_selected_right_door_evidence_without_fixed_id(self) -> None:
        spec = _spec("right_door_exit")
        completion = _base_completion(spec)
        completion["exit_evidence"] = _selected_right_door_evidence(
            door_object_id="Door|runtime-selected|+10.00|+00.00|+03.50"
        )

        with TemporaryDirectory() as temporary_directory:
            result = verify_demo_summary(
                spec=spec,
                summary=_summary_for(root=Path(temporary_directory), spec=spec, completion=completion),
            )

        evidence = result["completion_status"]["exit_evidence"]
        self.assertEqual(evidence["doorObjectId"], "Door|runtime-selected|+10.00|+00.00|+03.50")
        self.assertEqual(evidence["selectedDoorObjectId"], evidence["doorObjectId"])

    def test_verify_demo_summary_rejects_unselected_crossed_door(self) -> None:
        spec = _spec("right_door_exit")
        completion = _base_completion(spec)
        completion["exit_evidence"] = _selected_right_door_evidence(
            door_object_id="Door|crossed-but-not-selected|+00.00|+00.00|+03.50",
            selected_door_object_id="Door|model-selected-other|+01.00|+00.00|+03.50",
        )

        with TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(RuntimeError, "right door selection"):
                verify_demo_summary(
                    spec=spec,
                    summary=_summary_for(root=Path(temporary_directory), spec=spec, completion=completion),
                )

    def test_verify_demo_summary_rejects_selected_door_without_right_relation(self) -> None:
        spec = _spec("right_door_exit")
        completion = _base_completion(spec)
        evidence = _selected_right_door_evidence()
        evidence["relation_to_agent"] = "left"
        evidence["relation_verified"] = False
        completion["exit_evidence"] = evidence

        with TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(RuntimeError, "right-side"):
                verify_demo_summary(
                    spec=spec,
                    summary=_summary_for(
                        root=Path(temporary_directory),
                        spec=spec,
                        completion=completion,
                    ),
                )

    def test_verify_demo_summary_rejects_non_ai2thor_backend(self) -> None:
        spec = _spec("television")
        with TemporaryDirectory() as temporary_directory:
            summary = _summary_for(root=Path(temporary_directory), spec=spec)
            summary["steps"][-1]["backend"] = "local_ppt_style"
            with self.assertRaisesRegex(RuntimeError, "backend is not strict AI2-THOR"):
                verify_demo_summary(spec=spec, summary=summary)

    def test_vase_into_box_requires_real_pickup_put_postconditions_and_final_state(self) -> None:
        spec = _spec("vase_into_box")
        with TemporaryDirectory() as temporary_directory:
            result = verify_demo_summary(
                spec=spec,
                summary=_summary_for(
                    root=Path(temporary_directory),
                    spec=spec,
                    completion=_vase_completion(),
                    prior_steps=_vase_steps(),
                ),
            )

        final_state = result["completion_status"]["final_state"]
        self.assertEqual(final_state["inventoryObjects"], [])
        self.assertEqual(final_state["vaseParentReceptacles"], ["Box|fixture"])

    def test_vase_into_box_rejects_success_actions_without_put_postcondition(self) -> None:
        spec = _spec("vase_into_box")
        completion = _vase_completion()
        completion["final_state"] = {
            "vaseObjectId": "Vase|fixture",
            "boxObjectId": "Box|fixture",
            "vaseParentReceptacles": [],
            "boxReceptacleObjectIds": [],
            "inventoryObjects": [],
        }

        with TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(RuntimeError, "receptacle postcondition"):
                verify_demo_summary(
                    spec=spec,
                    summary=_summary_for(
                        root=Path(temporary_directory),
                        spec=spec,
                        completion=completion,
                        prior_steps=_vase_steps(),
                    ),
                )

    def test_television_search_rejects_premature_done_without_target_evidence(self) -> None:
        spec = _spec("television")
        completion = _base_completion(spec)
        completion["target_located"] = False
        completion["target_visible"] = False

        with TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(RuntimeError, "target evidence"):
                verify_demo_summary(
                    spec=spec,
                    summary=_summary_for(root=Path(temporary_directory), spec=spec, completion=completion),
                )

    def test_sofa_sit_rejects_premature_done_before_approach_and_crouch(self) -> None:
        spec = _spec("sofa_sit")
        completion = _base_completion(spec)
        completion["outcome"] = "approximate_success"
        completion["completion_mode"] = "approximate_sit"
        completion["target_located"] = True
        completion["approach_verified"] = False
        completion["successful_actions"] = []

        with TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(RuntimeError, "approach.*Crouch|Crouch.*approach"):
                verify_demo_summary(
                    spec=spec,
                    summary=_summary_for(root=Path(temporary_directory), spec=spec, completion=completion),
                )

    def test_run_all_uses_real_agent_runner_contract_without_unity(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fake_demos: dict[str, _FakeDemo] = {}

            def factory(spec: FinalDemoSpec) -> _FakeDemo:
                demo = _FakeDemo(spec, root)
                fake_demos[spec.task_id] = demo
                return demo

            with patch.object(
                run_final_agent_demos,
                "verify_video",
                return_value={"decodable": True, "frame_count": 3, "readable_frames": 3},
            ):
                result = run_all(output_dir=root / "out", demo_factory=factory)

        self.assertEqual(result["status"], "passed")
        self.assertEqual(len(result["tasks"]), 4)
        self.assertEqual(
            fake_demos["right_door_exit"].calls[0]["initial_pose"],
            _spec("right_door_exit").initial_pose,
        )
        self.assertEqual(
            fake_demos["vase_into_box"].calls[0]["instruction"],
            _spec("vase_into_box").instruction,
        )


if __name__ == "__main__":
    unittest.main()
