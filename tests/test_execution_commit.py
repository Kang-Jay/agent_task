from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.agent.controller import EmbodiedSearchAgent
from src.agent.model_adapter import ModelAdapter
from src.task.config import AgentConfig, load_config
from src.types.schema import AgentRequest


class ExecutionCommitTests(unittest.TestCase):
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

    def test_executed_action_replaces_proposal_in_memory_and_trace(self) -> None:
        agent = EmbodiedSearchAgent(
            self.config,
            model_adapter=ModelAdapter(credentials=[]),
        )
        session_id = "execution-commit"
        response = agent.step(
            AgentRequest(
                session_id=session_id,
                instruction="Find the red cup on the table",
                observation_image=str(
                    self.config.image_dir / "ep_red_cup_visible_000.png"
                ),
                step_id=0,
            )
        )
        response_dict = response.to_dict()
        proposed_action = response_dict["action"]
        response_dict["action"] = {
            "type": "TURN_RIGHT",
            "args": {"angle": 30},
        }
        response_dict["done"] = False
        response_dict["planner_source"] = "simulator_oracle"
        response_dict["skill_call"] = {
            "name": "TURN_RIGHT",
            "args": {"angle": 30},
            "preconditions": [],
            "expected_observation": "camera heading changes",
        }

        committed = agent.commit_execution(
            session_id,
            response_dict,
            step_id=0,
            action_success=True,
            robot_before={"x": 0.0, "y": 0.0, "heading": 0.0},
            robot_after={"x": 0.0, "y": 0.0, "heading": 30.0},
        )

        trace = agent.export_trace(session_id)
        step = trace["steps"][-1]
        self.assertEqual(step["proposed_action"], proposed_action)
        self.assertEqual(step["action"]["type"], "TURN_RIGHT")
        self.assertEqual(step["executed_action"]["type"], "TURN_RIGHT")
        self.assertEqual(step["planner_source"], "simulator_oracle")
        self.assertTrue(step["action_success"])
        self.assertEqual(step["robot_after"]["heading"], 30.0)
        self.assertIn("Last action=TURN_RIGHT", committed["memory_summary"])

    def test_step_id_commit_does_not_modify_newer_pending_step(self) -> None:
        agent = EmbodiedSearchAgent(
            self.config,
            model_adapter=ModelAdapter(credentials=[]),
        )
        session_id = "execution-step-isolation"
        image_path = str(self.config.image_dir / "ep_red_cup_visible_000.png")
        first = agent.step(
            AgentRequest(
                session_id=session_id,
                instruction="Find the red cup on the table",
                observation_image=image_path,
                step_id=0,
            )
        )
        agent.step(
            AgentRequest(
                session_id=session_id,
                instruction="Find the red cup on the table",
                observation_image=image_path,
                step_id=1,
            )
        )
        agent.commit_execution(
            session_id,
            first.to_dict(),
            step_id=0,
            action_success=True,
        )
        trace = agent.export_trace(session_id)
        self.assertTrue(trace["steps"][0]["action_success"])
        self.assertNotIn("action_success", trace["steps"][1])

    def test_post_action_verification_completes_crouch_same_step(self) -> None:
        agent = EmbodiedSearchAgent(
            self.config,
            model_adapter=ModelAdapter(credentials=[]),
        )
        session_id = "execution-same-step-completion"
        image_path = str(
            self.config.image_dir / "ep_red_cup_visible_000.png"
        )
        context_before = {
            "agent": {"isStanding": True},
            "objects": [
                {
                    "objectId": "Sofa|1",
                    "objectType": "Sofa",
                    "visible": True,
                    "distance": 1.2,
                }
            ],
            "approach": {
                "verified": True,
                "objectId": "Sofa|1",
                "source": "ai2thor_interactable_pose",
            },
        }
        response = agent.step(
            AgentRequest(
                session_id=session_id,
                instruction="找到沙发并坐下",
                observation_image=image_path,
                step_id=0,
                environment_context=context_before,
            )
        )
        response_dict = response.to_dict()
        response_dict["action"] = {"type": "Crouch", "args": {}}
        response_dict["done"] = False
        context_after = {
            **context_before,
            "agent": {"isStanding": False},
        }

        committed = agent.commit_execution(
            session_id,
            response_dict,
            step_id=0,
            action_success=True,
            environment_context=context_after,
        )

        self.assertTrue(committed["done"])
        self.assertTrue(committed["completion_status"]["complete"])
        self.assertEqual(
            committed["completion_status"]["outcome"],
            "approximate_success",
        )
        self.assertEqual(
            committed["execution_plan"]["status"],
            "completed",
        )
        trace = agent.export_trace(session_id)
        self.assertTrue(trace["steps"][0]["done"])
        self.assertTrue(
            trace["steps"][0]["completion_status"]["complete"]
        )

    def test_failed_crouch_stays_incomplete_same_step(self) -> None:
        agent = EmbodiedSearchAgent(
            self.config,
            model_adapter=ModelAdapter(credentials=[]),
        )
        session_id = "execution-same-step-failure"
        context = {
            "agent": {"isStanding": True},
            "objects": [
                {
                    "objectId": "Sofa|1",
                    "objectType": "Sofa",
                    "visible": True,
                }
            ],
            "approach": {
                "verified": True,
                "objectId": "Sofa|1",
                "source": "ai2thor_interactable_pose",
            },
        }
        response = agent.step(
            AgentRequest(
                session_id=session_id,
                instruction="找到沙发并坐下",
                observation_image=str(
                    self.config.image_dir
                    / "ep_red_cup_visible_000.png"
                ),
                step_id=0,
                environment_context=context,
            )
        )
        response_dict = response.to_dict()
        response_dict["action"] = {"type": "Crouch", "args": {}}
        response_dict["done"] = False

        committed = agent.commit_execution(
            session_id,
            response_dict,
            step_id=0,
            action_success=False,
            environment_context=context,
        )

        self.assertFalse(committed["done"])
        self.assertFalse(committed["completion_status"]["complete"])
        self.assertIn(
            "Crouch",
            committed["completion_status"]["missing_actions"],
        )

    def test_committed_done_action_does_not_complete_without_verifier(self) -> None:
        agent = EmbodiedSearchAgent(
            self.config,
            model_adapter=ModelAdapter(credentials=[]),
        )
        session_id = "execution-done-needs-verifier"
        environment_context = {
            "agent": {"isStanding": True},
            "objects": [
                {
                    "objectId": "Vase|1",
                    "objectType": "Vase",
                    "visible": True,
                    "pickupable": True,
                },
                {
                    "objectId": "Box|1",
                    "objectType": "Box",
                    "visible": True,
                    "receptacle": True,
                },
            ],
        }
        response = agent.step(
            AgentRequest(
                session_id=session_id,
                instruction="把花瓶放到纸箱里",
                observation_image=str(
                    self.config.image_dir
                    / "ep_red_cup_visible_000.png"
                ),
                step_id=0,
                environment_context=environment_context,
            )
        )
        response_dict = response.to_dict()
        response_dict["action"] = {"type": "Done", "args": {}}
        response_dict["done"] = True

        committed = agent.commit_execution(
            session_id,
            response_dict,
            step_id=0,
            action_success=True,
            environment_context=environment_context,
        )

        self.assertFalse(committed["done"])
        self.assertFalse(committed["completion_status"]["complete"])
        self.assertIn(
            "PickupObject",
            committed["completion_status"]["missing_actions"],
        )
        trace = agent.export_trace(session_id)
        self.assertFalse(trace["steps"][0]["done"])

    def test_latest_step_cannot_be_committed_twice(self) -> None:
        agent = EmbodiedSearchAgent(
            self.config,
            model_adapter=ModelAdapter(credentials=[]),
        )
        session_id = "execution-duplicate"
        response = agent.step(
            AgentRequest(
                session_id=session_id,
                instruction="Find the red cup",
                observation_image=str(
                    self.config.image_dir
                    / "ep_red_cup_visible_000.png"
                ),
                step_id=0,
            )
        )
        agent.commit_execution(
            session_id,
            response.to_dict(),
            action_success=True,
        )
        with self.assertRaisesRegex(ValueError, "already committed"):
            agent.commit_execution(
                session_id,
                response.to_dict(),
                action_success=True,
            )

    def test_session_id_rejects_path_traversal(self) -> None:
        agent = EmbodiedSearchAgent(
            self.config,
            model_adapter=ModelAdapter(credentials=[]),
        )
        with self.assertRaises(ValueError):
            agent.reset("../outside")

    def test_session_cannot_silently_change_instruction(self) -> None:
        agent = EmbodiedSearchAgent(
            self.config,
            model_adapter=ModelAdapter(credentials=[]),
        )
        state = agent.memory.get_or_create("same-session", "Find the cup")
        self.assertEqual(state.instruction, "Find the cup")
        with self.assertRaises(ValueError):
            agent.memory.get_or_create("same-session", "Find the television")


if __name__ == "__main__":
    unittest.main()
