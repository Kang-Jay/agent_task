from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.agent.controller import EmbodiedSearchAgent
from src.agent.model_adapter import ModelAdapter
from src.memory.episodic_store import EpisodicMemoryStore
from src.task.config import AgentConfig, load_config
from src.types.schema import AgentRequest


class CapturingModelAdapter:
    def __init__(self) -> None:
        self.payload: dict[str, object] | None = None

    def available(self) -> bool:
        return True

    def plan_action(self, payload: dict[str, object]) -> dict[str, object]:
        self.payload = payload
        return {
            "thought_summary": "Use the recalled execution lesson.",
            "action": {"type": "TURN_RIGHT", "args": {"angle": 30}},
            "confidence": 0.7,
            "provider_used": "test",
            "model_used": "capturing-adapter",
            "vision_input_used": True,
        }


class EpisodicMemoryStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.db_path = Path(self.temporary_directory.name) / "episodic.sqlite3"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_memory_persists_across_store_instances(self) -> None:
        store = EpisodicMemoryStore(self.db_path, capacity=10)
        store.add(
            namespace="visual_search",
            session_id="session-a",
            instruction="Find the red cup",
            action="TURN_RIGHT",
            action_success=True,
            confidence=0.7,
            region="middle right",
            lesson="Turning right exposed the red cup.",
        )

        reopened = EpisodicMemoryStore(self.db_path, capacity=10)
        results = reopened.search(
            "Locate the red cup",
            namespace="visual_search",
            top_k=3,
        )

        self.assertEqual(reopened.count(namespace="visual_search"), 1)
        self.assertEqual(results[0]["session_id"], "session-a")
        self.assertEqual(results[0]["action"], "TURN_RIGHT")

    def test_related_memory_ranks_before_unrelated_memory(self) -> None:
        store = EpisodicMemoryStore(self.db_path, capacity=10)
        store.add(
            namespace="visual_search",
            session_id="cup-session",
            instruction="Find the red cup on the table",
            action="INSPECT",
            action_success=True,
            confidence=0.9,
            region="middle center",
            lesson="Inspect the red cup near the table center.",
        )
        store.add(
            namespace="visual_search",
            session_id="tv-session",
            instruction="Find the television",
            action="TURN_LEFT",
            action_success=True,
            confidence=0.95,
            region="upper left",
            lesson="Turn left to reveal the television.",
        )

        results = store.search(
            "Search for the red cup",
            namespace="visual_search",
            top_k=3,
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["session_id"], "cup-session")

    def test_search_excludes_current_session(self) -> None:
        store = EpisodicMemoryStore(self.db_path, capacity=10)
        for session_id in ("current", "prior"):
            store.add(
                namespace="visual_search",
                session_id=session_id,
                instruction="寻找红色杯子",
                action="MOVE_FORWARD",
                action_success=True,
                confidence=0.8,
                region="middle center",
                lesson="向前移动后看到了红色杯子。",
            )

        results = store.search(
            "请寻找红色杯子",
            namespace="visual_search",
            top_k=3,
            exclude_session_id="current",
        )

        self.assertEqual([item["session_id"] for item in results], ["prior"])

    def test_capacity_prunes_oldest_memories(self) -> None:
        store = EpisodicMemoryStore(self.db_path, capacity=2)
        for index in range(3):
            store.add(
                namespace="visual_search",
                session_id=f"session-{index}",
                instruction="Find the blue book",
                action="TURN_RIGHT",
                action_success=True,
                confidence=0.5 + index / 10,
                region="middle right",
                lesson=f"memory-{index}",
            )

        results = store.search(
            "Find the blue book",
            namespace="visual_search",
            top_k=10,
        )

        self.assertEqual(store.count(namespace="visual_search"), 2)
        self.assertEqual(
            {item["session_id"] for item in results},
            {"session-1", "session-2"},
        )

    def test_agent_reloads_failed_execution_memory_and_injects_model_payload(self) -> None:
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
                session_id="failed-cup-search",
                instruction="Find the red cup",
                observation_image=str(image_path),
                step_id=0,
            )
        ).to_dict()
        first_agent.commit_execution(
            "failed-cup-search",
            first_response,
            action_success=False,
            environment={"backend": "test", "scene": "unit-room"},
        )
        first_agent.commit_execution(
            "failed-cup-search",
            first_response,
            action_success=False,
            environment={"backend": "test", "scene": "unit-room"},
        )
        self.assertEqual(
            first_agent.memory.episodic_store.count(namespace="visual_search"),
            1,
        )

        capturing_adapter = CapturingModelAdapter()
        restarted_agent = EmbodiedSearchAgent(
            config,
            model_adapter=capturing_adapter,
        )
        response = restarted_agent.step(
            AgentRequest(
                session_id="retry-cup-search",
                instruction="Locate the red cup",
                observation_image=str(image_path),
                step_id=0,
            )
        )

        self.assertEqual(len(response.recalled_memories), 1)
        self.assertIn("failed", response.recalled_memories[0]["lesson"])
        self.assertIsNotNone(capturing_adapter.payload)
        assert capturing_adapter.payload is not None
        injected = capturing_adapter.payload["episodic_memories"]
        self.assertEqual(len(injected), 1)
        self.assertEqual(injected[0]["session_id"], "failed-cup-search")
        self.assertEqual(
            response.search_map["recalled_memories"][0]["id"],
            response.recalled_memories[0]["id"],
        )
        self.assertTrue(
            response.structured_thought["observation"].startswith("当前画面")
        )
        self.assertIn(
            response.structured_thought["action"],
            {
                "向前移动",
                "向左转",
                "向右转",
                "向上看",
                "向下看",
                "仔细检查",
                "停止",
                "请求澄清",
            },
        )


if __name__ == "__main__":
    unittest.main()
