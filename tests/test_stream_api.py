from __future__ import annotations

import json
import unittest
from unittest.mock import patch

import httpx

from src.simulation.ai2thor_adapter import AI2ThorVisualSearchDemo, SimulatorStatus
from src.simulation.room_simulator import DemoResult, DemoStep
from src.simulation.stream_protocol import StreamEventEmitter
from src.ui.app import active_stream_sessions, app


class StreamAPITests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        active_stream_sessions.clear()

    async def test_ai2thor_stream_returns_ordered_ndjson_events(self):
        def fake_run_demo(self, **kwargs):
            emitter = StreamEventEmitter(kwargs["episode_id"], kwargs["emit"])
            emitter.emit(
                "task_parsed",
                instruction=kwargs["instruction"],
                task_plan={"task_types": ["visual_search"]},
            )
            step = DemoStep(
                frame_path="docs/test/frame.png",
                observation_path="docs/test/observation.png",
                topdown_path="docs/test/map.png",
                thought="目标未确认，继续搜索。",
                action="RotateRight",
                confidence=0.42,
                done=False,
                robot={"x": 0.0, "y": 0.0, "heading": 0.0},
                best_candidate=None,
                visible_objects=["Floor"],
                backend="ai2thor",
                scene=self.scene,
            )
            emitter.emit("step_completed", step_id=0, step=step.__dict__)
            result = DemoResult(
                steps=[step],
                video_path="docs/test/demo.mp4",
                summary_path="docs/test/summary.json",
                episode_id=kwargs["episode_id"],
                output_dir="docs/test",
            )
            emitter.emit("episode_completed", result=result.to_dict())
            return result

        status = SimulatorStatus(
            available=True,
            backend="ai2thor",
            scene="FloorPlan211",
            message="ready",
            diagnostics={},
        )
        transport = httpx.ASGITransport(app=app)
        with (
            patch.object(AI2ThorVisualSearchDemo, "status", return_value=status),
            patch.object(AI2ThorVisualSearchDemo, "run_demo", new=fake_run_demo),
        ):
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                async with client.stream(
                    "POST",
                    "/api/demo/ai2thor/stream",
                    json={
                        "session_id": "stream-test",
                        "instruction": "找到电视",
                        "scene": "FloorPlan211",
                        "agent_mode": "default",
                    },
                ) as response:
                    self.assertEqual(response.status_code, 200)
                    episode_id = response.headers["x-episode-id"]
                    messages = [
                        json.loads(line)
                        async for line in response.aiter_lines()
                        if line
                    ]

        self.assertEqual(
            [message["event"] for message in messages],
            ["task_parsed", "step_completed", "episode_completed"],
        )
        self.assertEqual(
            [message["sequence"] for message in messages],
            [0, 1, 2],
        )
        self.assertTrue(
            all(message["episode_id"] == episode_id for message in messages)
        )
        self.assertNotIn("stream-test", active_stream_sessions)


if __name__ == "__main__":
    unittest.main()
