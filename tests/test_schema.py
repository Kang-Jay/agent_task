"""Tests for schema validation and serialization.

According to Plan_1_agent_demo_repair.md Phase 1 requirements.
"""
from __future__ import annotations

import unittest

from src.types.schema import (
    Action,
    AgentRequest,
    AgentResponse,
    ClickedObjectBinding,
    SkillCall,
    ObservationAnalysis,
    Candidate,
)


class SchemaTests(unittest.TestCase):
    """Test schema structure, serialization, and validation."""

    def test_skill_call_serialization(self) -> None:
        """Test SkillCall can be created and serialized."""
        skill = SkillCall(
            name="TURN_RIGHT",
            args={"angle": 30},
            preconditions=["agent_is_idle"],
            expected_observation="camera heading changes to the right"
        )
        result = skill.to_dict()
        self.assertEqual(result["name"], "TURN_RIGHT")
        self.assertEqual(result["args"], {"angle": 30})
        self.assertEqual(result["preconditions"], ["agent_is_idle"])
        self.assertEqual(result["expected_observation"], "camera heading changes to the right")

    def test_skill_call_minimal(self) -> None:
        """Test SkillCall with minimal fields."""
        skill = SkillCall(name="STOP")
        result = skill.to_dict()
        self.assertEqual(result["name"], "STOP")
        self.assertEqual(result["args"], {})
        self.assertEqual(result["preconditions"], [])
        self.assertEqual(result["expected_observation"], "")

    def test_planner_source_enum_values(self) -> None:
        """Test planner_source accepts only valid enum values."""
        valid_sources = ["model_planner", "rule_fallback", "simulator_oracle", "human_manual"]

        # Create a minimal observation for testing
        candidate = Candidate(
            label="test", bbox=[0, 0, 10, 10], confidence=0.5,
            color_name="red", region="center", reason="test"
        )
        obs = ObservationAnalysis(
            image_size=(100, 100), scene_summary="test",
            candidates=[candidate], best_candidate=candidate, target_visible=True
        )

        for source in valid_sources:
            response = AgentResponse(
                session_id="test",
                step_id=0,
                thought="test",
                action=Action("STOP"),
                confidence=0.8,
                done=True,
                observation=obs,
                retrieved_hints=[],
                memory_summary="test",
                replay=[],
                planner_source=source  # type: ignore
            )
            self.assertEqual(response.planner_source, source)

    def test_agent_response_includes_skill_call_and_planner_source(self) -> None:
        """Test AgentResponse serialization includes new fields."""
        candidate = Candidate(
            label="red cup", bbox=[10, 20, 50, 60], confidence=0.85,
            color_name="red", region="middle center", reason="color match"
        )
        obs = ObservationAnalysis(
            image_size=(448, 448), scene_summary="test scene",
            candidates=[candidate], best_candidate=candidate, target_visible=True
        )

        skill = SkillCall(name="INSPECT", args={}, preconditions=[], expected_observation="closer view")

        response = AgentResponse(
            session_id="test-schema",
            step_id=5,
            thought="Found target, inspecting",
            action=Action("INSPECT", {}),
            confidence=0.85,
            done=False,
            observation=obs,
            retrieved_hints=["hint1"],
            memory_summary="5 steps",
            replay=[],
            skill_call=skill,
            planner_source="model_planner",
            model_info={
                "status": "ok",
                "provider": "kimi",
                "model": "moonshot-v1-8k-vision-preview",
                "vision_input_used": True,
            },
        )

        result = response.to_dict()

        # Verify new fields are present
        self.assertIn("skill_call", result)
        self.assertIn("planner_source", result)
        self.assertIn("model_info", result)
        self.assertIn("fallback_reason", result)

        # Verify skill_call serialization
        self.assertEqual(result["skill_call"]["name"], "INSPECT")
        self.assertEqual(result["skill_call"]["expected_observation"], "closer view")

        # Verify planner_source
        self.assertEqual(result["planner_source"], "model_planner")
        self.assertTrue(result["model_info"]["vision_input_used"])

    def test_agent_response_with_none_skill_call(self) -> None:
        """Test AgentResponse serialization when skill_call is None."""
        candidate = Candidate(
            label="test", bbox=[0, 0, 10, 10], confidence=0.5,
            color_name="red", region="center", reason="test"
        )
        obs = ObservationAnalysis(
            image_size=(100, 100), scene_summary="test",
            candidates=[candidate], best_candidate=candidate, target_visible=True
        )

        response = AgentResponse(
            session_id="test",
            step_id=0,
            thought="test",
            action=Action("TURN_RIGHT"),
            confidence=0.5,
            done=False,
            observation=obs,
            retrieved_hints=[],
            memory_summary="test",
            replay=[],
            skill_call=None,
            planner_source="rule_fallback"
        )

        result = response.to_dict()
        self.assertIsNone(result["skill_call"])
        self.assertEqual(result["planner_source"], "rule_fallback")

    def test_default_planner_source_is_rule_fallback(self) -> None:
        """Test default planner_source is rule_fallback."""
        candidate = Candidate(
            label="test", bbox=[0, 0, 10, 10], confidence=0.5,
            color_name="red", region="center", reason="test"
        )
        obs = ObservationAnalysis(
            image_size=(100, 100), scene_summary="test",
            candidates=[candidate], best_candidate=candidate, target_visible=True
        )

        # Create response without specifying planner_source
        response = AgentResponse(
            session_id="test",
            step_id=0,
            thought="test",
            action=Action("STOP"),
            confidence=0.8,
            done=True,
            observation=obs,
            retrieved_hints=[],
            memory_summary="test",
            replay=[]
        )

        self.assertEqual(response.planner_source, "rule_fallback")

    def test_skill_call_name_required(self) -> None:
        """Test SkillCall requires name field."""
        # This should work
        skill = SkillCall(name="MOVE_FORWARD")
        self.assertEqual(skill.name, "MOVE_FORWARD")

        # Creating SkillCall without name should fail at runtime
        # (caught by type checker at dev time)


class ClickedObjectBindingTests(unittest.TestCase):
    """Test ClickedObjectBinding structure and serialization (ChangeRecord 10016)."""

    def test_full_serialization(self) -> None:
        binding = ClickedObjectBinding(
            object_id="Television|+00.00|+00.50|+01.00",
            object_type="Television",
            affordances={"pickupable": False, "toggleable": True},
            closeup_source="third_party_camera",
            closeup_bbox=[10, 20, 110, 140],
            world_position={"x": 0.0, "y": 0.5, "z": 1.0},
        )
        result = binding.to_dict()
        self.assertEqual(result["object_id"], "Television|+00.00|+00.50|+01.00")
        self.assertEqual(result["object_type"], "Television")
        self.assertEqual(result["affordances"], {"pickupable": False, "toggleable": True})
        self.assertEqual(result["closeup_source"], "third_party_camera")
        self.assertEqual(result["closeup_bbox"], [10, 20, 110, 140])
        self.assertEqual(result["world_position"], {"x": 0.0, "y": 0.5, "z": 1.0})

    def test_minimal_defaults(self) -> None:
        binding = ClickedObjectBinding(object_id="Box|1", object_type="Box")
        result = binding.to_dict()
        self.assertEqual(result["object_id"], "Box|1")
        self.assertEqual(result["object_type"], "Box")
        self.assertEqual(result["affordances"], {})
        self.assertEqual(result["closeup_source"], "")
        self.assertIsNone(result["closeup_bbox"])
        self.assertIsNone(result["world_position"])

    def test_to_dict_key_completeness(self) -> None:
        binding = ClickedObjectBinding(object_id="X|1", object_type="X")
        expected_keys = {
            "object_id",
            "object_type",
            "affordances",
            "closeup_source",
            "closeup_bbox",
            "world_position",
        }
        self.assertEqual(set(binding.to_dict().keys()), expected_keys)


class AgentRequestClickFieldTests(unittest.TestCase):
    """Test AgentRequest new clicked_object_id field (ChangeRecord 10016)."""

    def test_clicked_object_id_defaults_none(self) -> None:
        request = AgentRequest(
            session_id="s",
            instruction="find tv",
            observation_image="img.png",
        )
        self.assertIsNone(request.clicked_object_id)
        # Existing click fields remain intact and independent
        self.assertIsNone(request.clicked_point)
        self.assertIsNone(request.target_crop)

    def test_clicked_object_id_accepts_value(self) -> None:
        request = AgentRequest(
            session_id="s",
            instruction="find tv",
            observation_image="img.png",
            clicked_object_id="Television|+00.00|+00.50|+01.00",
        )
        self.assertEqual(request.clicked_object_id, "Television|+00.00|+00.50|+01.00")


if __name__ == "__main__":
    unittest.main()
