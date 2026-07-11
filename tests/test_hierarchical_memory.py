from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.agent.controller import EmbodiedSearchAgent
from src.agent.model_adapter import ModelAdapter
from src.memory.hierarchical_memory import (
    EvidenceReference,
    HierarchicalMemoryStore,
    MEMORY_LAYERS,
)
from src.task.config import AgentConfig, load_config
from src.types.schema import AgentRequest


class CapturingLayeredMemoryAdapter:
    def __init__(self) -> None:
        self.payload: dict[str, object] | None = None

    def available(self) -> bool:
        return True

    def plan_action(self, payload: dict[str, object]) -> dict[str, object]:
        self.payload = payload
        return {
            "thought_summary": "Use verified layered memory evidence.",
            "action": {"type": "TURN_RIGHT", "args": {"angle": 30}},
            "confidence": 0.7,
            "provider_used": "test",
            "model_used": "layered-memory-adapter",
            "vision_input_used": True,
        }


class HierarchicalMemoryStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.db_path = (
            Path(self.temporary_directory.name) / "hierarchical.sqlite3"
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    @staticmethod
    def evidence(session_id: str, step_id: int) -> EvidenceReference:
        return EvidenceReference(
            session_id=session_id,
            step_id=step_id,
            source="unit_test",
            reference=f"trace://{session_id}#step={step_id}",
            details={"verified": True},
        )

    def test_duplicate_identity_merges_evidence(self) -> None:
        store = HierarchicalMemoryStore(
            self.db_path,
            capacity=10,
            failure_capacity=4,
        )
        first_id = store.upsert(
            layer="object",
            identity_key="FloorPlan1|Cup|1",
            session_id="session-a",
            instruction="Find the cup",
            subject="Cup|1",
            summary="Cup was visible on the table.",
            evidence=self.evidence("session-a", 0),
            confidence=0.7,
            metadata={"region": "middle center"},
        )
        second_id = store.upsert(
            layer="object",
            identity_key="FloorPlan1|Cup|1",
            session_id="session-b",
            instruction="Locate the cup",
            subject="Cup|1",
            summary="Cup remained visible on the table.",
            evidence=self.evidence("session-b", 2),
            confidence=0.9,
            metadata={"region": "middle center"},
        )

        self.assertEqual(first_id, second_id)
        record = store.get(first_id)
        assert record is not None
        self.assertEqual(record["occurrence_count"], 2)
        self.assertEqual(len(record["evidence"]), 2)
        self.assertEqual(record["session_id"], "session-b")

    def test_capacity_and_failure_capacity_expire_oldest_records(self) -> None:
        store = HierarchicalMemoryStore(
            self.db_path,
            capacity=3,
            failure_capacity=1,
        )
        for index in range(2):
            store.upsert(
                layer="failure",
                identity_key=f"failure-{index}",
                session_id=f"failure-session-{index}",
                instruction="Find the cup",
                subject="MOVE_FORWARD",
                summary=f"Movement failure {index} near the cup.",
                evidence=self.evidence(f"failure-session-{index}", index),
                success=False,
            )
        for index in range(3):
            store.upsert(
                layer="skill",
                identity_key=f"skill-{index}",
                session_id=f"skill-session-{index}",
                instruction="Find the cup",
                subject="TURN_RIGHT",
                summary=f"Skill result {index} for cup search.",
                evidence=self.evidence(f"skill-session-{index}", index),
                success=True,
            )

        self.assertEqual(store.count(layer="failure"), 0)
        self.assertEqual(store.count(), 3)
        skill_results = store.search(
            "cup skill",
            top_k=10,
            layers=["skill"],
        )
        self.assertEqual(
            {item["identity_key"] for item in skill_results},
            {"skill-0", "skill-1", "skill-2"},
        )

    def test_grouped_search_persists_all_layers_and_evidence(self) -> None:
        store = HierarchicalMemoryStore(
            self.db_path,
            capacity=20,
            failure_capacity=8,
        )
        for index, layer in enumerate(MEMORY_LAYERS):
            store.upsert(
                layer=layer,
                identity_key=f"{layer}-cup",
                session_id=f"source-{layer}",
                instruction="Find the red cup",
                subject=f"{layer} cup",
                summary=f"{layer} evidence for the red cup.",
                evidence=self.evidence(f"source-{layer}", index),
                success=False if layer == "failure" else True,
                confidence=0.6 + index / 100,
            )

        reopened = HierarchicalMemoryStore(
            self.db_path,
            capacity=20,
            failure_capacity=8,
        )
        grouped = reopened.search_grouped(
            "Locate the red cup",
            top_k=len(MEMORY_LAYERS),
            exclude_session_id="unrelated-current-session",
        )

        self.assertEqual(set(grouped), set(MEMORY_LAYERS))
        self.assertEqual(
            sum(len(items) for items in grouped.values()),
            len(MEMORY_LAYERS),
        )
        for layer in MEMORY_LAYERS:
            self.assertEqual(len(grouped[layer]), 1)
            self.assertEqual(grouped[layer][0]["layer"], layer)
            self.assertEqual(
                grouped[layer][0]["evidence"][0]["source"],
                "unit_test",
            )

    def test_search_excludes_current_session(self) -> None:
        store = HierarchicalMemoryStore(
            self.db_path,
            capacity=10,
            failure_capacity=4,
        )
        for session_id in ("current", "prior"):
            store.upsert(
                layer="episode",
                identity_key=f"{session_id}:0",
                session_id=session_id,
                instruction="Find the television",
                subject="episode",
                summary="Turned toward the television.",
                evidence=self.evidence(session_id, 0),
                success=True,
            )

        results = store.search(
            "Locate the television",
            top_k=3,
            layers=["episode"],
            exclude_session_id="current",
        )

        self.assertEqual(
            [item["session_id"] for item in results],
            ["prior"],
        )

    def test_controller_persists_and_injects_layered_memory(self) -> None:
        default_config = load_config()
        raw = deepcopy(default_config.raw)
        raw["data"]["trajectory_dir"] = str(
            Path(self.temporary_directory.name) / "trajectories"
        )
        config = AgentConfig(raw=raw, path=default_config.path)
        image_path = config.image_dir / "ep_red_cup_visible_000.png"

        first_agent = EmbodiedSearchAgent(
            config,
            model_adapter=ModelAdapter(credentials=[]),
        )
        first_response = first_agent.step(
            AgentRequest(
                session_id="layered-source",
                instruction="Find the red cup",
                observation_image=str(image_path),
                step_id=0,
                environment_context={
                    "scene": "FloorPlan1",
                    "objects": [
                        {
                            "objectId": "Cup|1",
                            "objectType": "Cup",
                            "visible": True,
                        }
                    ],
                },
            )
        ).to_dict()
        first_agent.commit_execution(
            "layered-source",
            first_response,
            action_success=False,
            robot_before={"x": 0.0, "z": 0.0, "heading": 0.0},
            robot_after={"x": 0.0, "z": 0.0, "heading": 30.0},
            environment={
                "backend": "test",
                "scene": "FloorPlan1",
                "failure_reason": "collision",
            },
            environment_context={
                "scene": "FloorPlan1",
                "agent": {
                    "position": {"x": 0.0, "y": 0.0, "z": 0.0},
                    "rotation": {"y": 30.0},
                },
                "objects": [
                    {
                        "objectId": "Cup|1",
                        "objectType": "Cup",
                        "visible": True,
                    }
                ],
            },
        )

        counts = first_agent.memory.hierarchical_store.layer_counts()
        self.assertEqual(counts["failure"], 1)
        for layer in ("object", "spatial", "task", "skill", "episode"):
            self.assertGreaterEqual(counts[layer], 1)

        capturing_adapter = CapturingLayeredMemoryAdapter()
        restarted_agent = EmbodiedSearchAgent(
            config,
            model_adapter=capturing_adapter,
        )
        response = restarted_agent.step(
            AgentRequest(
                session_id="layered-retry",
                instruction="Locate the red cup",
                observation_image=str(image_path),
                step_id=0,
                environment_context={"scene": "FloorPlan1"},
            )
        )

        self.assertIsNotNone(capturing_adapter.payload)
        assert capturing_adapter.payload is not None
        layered = capturing_adapter.payload["layered_memories"]
        self.assertEqual(set(layered), set(MEMORY_LAYERS))
        self.assertLessEqual(
            sum(len(items) for items in layered.values()),
            config.raw["memory"]["retrieval_top_k"],
        )
        failures = restarted_agent.memory.hierarchical_store.search(
            "red cup collision failed",
            top_k=config.raw["memory"]["retrieval_top_k"],
            layers=["failure"],
            exclude_session_id="layered-retry",
        )
        self.assertEqual(len(failures), 1)
        failure = failures[0]
        self.assertEqual(failure["metadata"]["failure_reason"], "collision")
        self.assertEqual(
            failure["evidence"][0]["session_id"],
            "layered-source",
        )
        self.assertIn("layered_memories", response.search_map)
        trace = first_agent.export_trace("layered-source")
        self.assertIn(
            "hierarchical_memory_ids",
            trace["steps"][0],
        )


if __name__ == "__main__":
    unittest.main()
