from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.agent.controller import EmbodiedSearchAgent
from src.memory.session_memory import SessionMemory
from src.task.config import AgentConfig, load_config
from src.types.schema import AgentRequest, ExecutionSubgoal, TaskExecutionPlan


class _FakePlanningAdapter:
    def __init__(self, *, valid_plan: bool = True):
        self.valid_plan = valid_plan
        self.plan_task_calls = 0
        self.plan_action_calls = 0
        self.last_action_payload = None

    def available(self) -> bool:
        return True

    def plan_task(self, payload):
        self.plan_task_calls += 1
        ordered = [
            "locate_target",
            "approach_target",
            "execute_crouch",
            "verify_posture",
        ]
        if not self.valid_plan:
            ordered = ordered[:-1]
        return {
            "task_summary": "Locate, approach, crouch, and verify.",
            "ordered_subgoal_ids": ordered,
            "failure_policy": "Refresh evidence and choose another valid action.",
            "vision_input_used": True,
        }

    def plan_action(self, payload):
        self.plan_action_calls += 1
        self.last_action_payload = payload
        return {
            "thought_summary": "Continue the current subgoal.",
            "task_progress": "The current subgoal is still active.",
            "action": {"type": "TURN_RIGHT", "args": {"angle": 30}},
            "confidence": 0.4,
            "vision_input_used": True,
            "provider_used": "test",
            "model_used": "test-vlm",
        }


class TaskExecutionPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temporary_directory = TemporaryDirectory()
        default_config = load_config()
        raw = deepcopy(default_config.raw)
        raw["data"]["trajectory_dir"] = str(
            Path(cls._temporary_directory.name) / "trajectories"
        )
        cls.config = AgentConfig(raw=raw, path=default_config.path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary_directory.cleanup()

    def _plan(self, session_id: str) -> TaskExecutionPlan:
        return TaskExecutionPlan(
            plan_id=f"{session_id}:v1",
            instruction="找到沙发并坐下",
            task_summary="Locate the sofa, approach it, crouch, and verify posture.",
            task_types=["visual_search", "navigate_to", "sit_approximation"],
            completion_mode="approximate_sit",
            subgoals=[
                ExecutionSubgoal(
                    id="locate_target",
                    description="Locate the sofa",
                    success_evidence="target observation",
                ),
                ExecutionSubgoal(
                    id="approach_target",
                    description="Approach the sofa",
                    success_evidence="finite simulator distance",
                ),
                ExecutionSubgoal(
                    id="execute_crouch",
                    description="Execute Crouch",
                    success_evidence="successful Crouch",
                ),
                ExecutionSubgoal(
                    id="verify_posture",
                    description="Verify crouched posture",
                    success_evidence="agent.isStanding=false",
                ),
            ],
            current_subgoal_id="locate_target",
            status="in_progress",
            source="semantic_fallback",
            failure_policy="replan after failed execution",
            limitations=["native_sit_on_furniture_state_unavailable"],
        )

    def test_plan_serialization_is_complete(self) -> None:
        plan = self._plan("serialize")
        payload = plan.to_dict()
        self.assertEqual(payload["plan_id"], "serialize:v1")
        self.assertEqual(payload["current_subgoal_id"], "locate_target")
        self.assertEqual(len(payload["subgoals"]), 4)
        self.assertEqual(payload["subgoals"][0]["status"], "pending")

    def test_session_keeps_one_plan_and_exports_it(self) -> None:
        memory = SessionMemory(self.config)
        state = memory.get_or_create("persist-plan", "找到沙发并坐下")
        first = memory.set_execution_plan(state, self._plan("persist-plan"))
        second = memory.set_execution_plan(state, self._plan("persist-plan"))
        self.assertIs(first, second)
        exported = memory.export_trace("persist-plan")
        self.assertEqual(exported["execution_plan"]["plan_id"], "persist-plan:v1")

    def test_progress_advances_only_from_verified_evidence(self) -> None:
        memory = SessionMemory(self.config)
        state = memory.get_or_create("advance-plan", "找到沙发并坐下")
        memory.set_execution_plan(state, self._plan("advance-plan"))

        plan = memory.update_execution_plan(
            state,
            {
                "complete": False,
                "target_located": True,
                "approach_verified": True,
                "target_distance": 1.2,
                "successful_actions": [],
                "subgoal_progress": [],
            },
            step_id=0,
        )
        self.assertIsNotNone(plan)
        self.assertEqual(plan.current_subgoal_id, "execute_crouch")
        self.assertEqual(plan.subgoals[0].status, "completed")
        self.assertEqual(plan.subgoals[1].status, "completed")
        self.assertEqual(plan.subgoals[2].status, "in_progress")

    def test_completion_closes_plan_after_all_predicates_pass(self) -> None:
        memory = SessionMemory(self.config)
        state = memory.get_or_create("complete-plan", "找到沙发并坐下")
        memory.set_execution_plan(state, self._plan("complete-plan"))
        plan = memory.update_execution_plan(
            state,
            {
                "complete": True,
                "target_located": True,
                "approach_verified": True,
                "target_distance": 1.2,
                "successful_actions": ["Crouch"],
                "subgoal_progress": [
                    {"id": "verify_posture", "complete": True, "evidence": "isStanding=false"}
                ],
            },
            step_id=3,
        )
        self.assertIsNotNone(plan)
        self.assertEqual(plan.status, "completed")
        self.assertIsNone(plan.current_subgoal_id)
        self.assertTrue(all(subgoal.status == "completed" for subgoal in plan.subgoals))

    def test_agent_generates_global_plan_once_and_reuses_it(self) -> None:
        adapter = _FakePlanningAdapter()
        agent = EmbodiedSearchAgent(self.config, model_adapter=adapter)
        image_path = str(self.config.image_dir / "ep_red_cup_visible_000.png")
        first = agent.step(
            AgentRequest(
                session_id="global-plan-once",
                instruction="找到沙发并坐下",
                observation_image=image_path,
                step_id=0,
            )
        )
        second = agent.step(
            AgentRequest(
                session_id="global-plan-once",
                instruction="找到沙发并坐下",
                observation_image=image_path,
                step_id=1,
            )
        )
        self.assertEqual(adapter.plan_task_calls, 1)
        self.assertEqual(adapter.plan_action_calls, 2)
        self.assertEqual(
            first.execution_plan["plan_id"],
            second.execution_plan["plan_id"],
        )
        self.assertEqual(first.execution_plan["source"], "model_planner")
        self.assertTrue(first.execution_plan["vision_input_used"])
        self.assertEqual(
            adapter.last_action_payload["execution_plan"]["current_subgoal_id"],
            second.execution_plan["current_subgoal_id"],
        )

    def test_invalid_model_plan_uses_explicit_semantic_fallback(self) -> None:
        adapter = _FakePlanningAdapter(valid_plan=False)
        agent = EmbodiedSearchAgent(self.config, model_adapter=adapter)
        response = agent.step(
            AgentRequest(
                session_id="invalid-global-plan",
                instruction="找到沙发并坐下",
                observation_image=str(
                    self.config.image_dir / "ep_red_cup_visible_000.png"
                ),
                step_id=0,
            )
        )
        self.assertEqual(adapter.plan_task_calls, 1)
        self.assertEqual(response.execution_plan["source"], "semantic_fallback")
        self.assertFalse(response.execution_plan["vision_input_used"])
        self.assertEqual(
            [item["id"] for item in response.execution_plan["subgoals"]],
            [
                "locate_target",
                "approach_target",
                "execute_crouch",
                "verify_posture",
            ],
        )


if __name__ == "__main__":
    unittest.main()
