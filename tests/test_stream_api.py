from __future__ import annotations

import asyncio
import json
import threading
import time
import unittest
from unittest.mock import AsyncMock, patch

import httpx
from starlette.requests import Request

from src.simulation.ai2thor_adapter import AI2ThorVisualSearchDemo, SimulatorStatus
from src.simulation.room_simulator import DemoResult, DemoStep
from src.simulation.stream_protocol import PROTOCOL_VERSION, StreamEventEmitter
from src.ui.app import active_stream_runs, active_stream_sessions, app


class StreamAPITests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        active_stream_sessions.clear()
        active_stream_runs.clear()

    @staticmethod
    def _status() -> SimulatorStatus:
        return SimulatorStatus(
            available=True,
            backend="ai2thor",
            scene="FloorPlan211",
            message="ready",
            diagnostics={},
        )

    @staticmethod
    def _messages(response: httpx.Response) -> list[dict[str, object]]:
        return [
            json.loads(line)
            for line in response.text.splitlines()
            if line
        ]

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
                completion_status={
                    "complete": True,
                    "outcome": "exact_success",
                    "reason": "target interaction verified",
                },
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

        transport = httpx.ASGITransport(app=app)
        with (
            patch.object(
                AI2ThorVisualSearchDemo,
                "status",
                return_value=self._status(),
            ),
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
                    run_id = response.headers["x-run-id"]
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
        self.assertTrue(all(message["run_id"] == run_id for message in messages))
        self.assertTrue(
            all(
                message["protocol_version"] == PROTOCOL_VERSION
                for message in messages
            )
        )
        self.assertTrue(
            all(
                message["event_seq"] == message["sequence"]
                for message in messages
            )
        )
        self.assertTrue(
            all(
                message["event_id"]
                == f"{run_id}:{episode_id}:{message['sequence']}"
                for message in messages
            )
        )
        terminal_messages = [
            message for message in messages if message.get("terminal")
        ]
        self.assertEqual(len(terminal_messages), 1)
        self.assertTrue(terminal_messages[0]["task_success"])
        self.assertEqual(
            terminal_messages[0]["terminal_reason"],
            "target interaction verified",
        )
        self.assertNotIn("stream-test", active_stream_sessions)
        self.assertNotIn(run_id, active_stream_runs)

    async def test_worker_exception_becomes_one_terminal_error_event(self):
        def fake_run_demo(self, **kwargs):
            emitter = StreamEventEmitter(kwargs["episode_id"], kwargs["emit"])
            emitter.emit("task_parsed", instruction=kwargs["instruction"])
            emitter.emit(
                "error",
                error_type="RuntimeError",
                message="Unity launch failed",
            )
            raise RuntimeError("Unity launch failed")

        transport = httpx.ASGITransport(app=app)
        with (
            patch.object(
                AI2ThorVisualSearchDemo,
                "status",
                return_value=self._status(),
            ),
            patch.object(AI2ThorVisualSearchDemo, "run_demo", new=fake_run_demo),
        ):
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                response = await client.post(
                    "/api/demo/ai2thor/stream",
                    json={
                        "session_id": "stream-error",
                        "instruction": "找到电视",
                    },
                )

        self.assertEqual(response.status_code, 200)
        messages = self._messages(response)
        terminal_messages = [
            message for message in messages if message.get("terminal")
        ]
        self.assertEqual(len(terminal_messages), 1)
        terminal = terminal_messages[0]
        self.assertEqual(terminal["event"], "error")
        self.assertFalse(terminal["task_success"])
        self.assertEqual(terminal["terminal_reason"], "internal_error")
        self.assertEqual(terminal["payload"]["error_type"], "RuntimeError")
        self.assertEqual(
            terminal["payload"]["message"],
            "Unity launch failed",
        )
        self.assertNotIn("stream-error", active_stream_sessions)
        self.assertNotIn(response.headers["x-run-id"], active_stream_runs)

    async def test_duplicate_and_out_of_order_worker_events_are_dropped(self):
        def fake_run_demo(self, **kwargs):
            run_id = "worker-run"
            episode_id = kwargs["episode_id"]
            kwargs["emit"](
                {
                    "protocol_version": PROTOCOL_VERSION,
                    "run_id": run_id,
                    "episode_id": episode_id,
                    "event": "task_parsed",
                    "event_seq": 0,
                    "event_id": f"{run_id}:{episode_id}:0",
                    "sequence": 0,
                    "payload": {"instruction": kwargs["instruction"]},
                }
            )
            kwargs["emit"](
                {
                    "protocol_version": PROTOCOL_VERSION,
                    "run_id": run_id,
                    "episode_id": episode_id,
                    "event": "task_parsed",
                    "event_seq": 0,
                    "event_id": f"{run_id}:{episode_id}:0",
                    "sequence": 0,
                    "payload": {"instruction": "duplicate"},
                }
            )
            kwargs["emit"](
                {
                    "protocol_version": PROTOCOL_VERSION,
                    "run_id": run_id,
                    "episode_id": episode_id,
                    "event": "model_decision",
                    "event_seq": 2,
                    "event_id": f"{run_id}:{episode_id}:2",
                    "sequence": 2,
                    "payload": {"step_id": 0, "proposed_action": {"type": "MoveAhead"}},
                }
            )
            kwargs["emit"](
                {
                    "protocol_version": PROTOCOL_VERSION,
                    "run_id": run_id,
                    "episode_id": episode_id,
                    "event": "observation_ready",
                    "event_seq": 1,
                    "event_id": f"{run_id}:{episode_id}:1",
                    "sequence": 1,
                    "payload": {"step_id": 0},
                }
            )
            step = DemoStep(
                frame_path="docs/test/frame.png",
                observation_path="docs/test/observation.png",
                topdown_path="docs/test/map.png",
                thought="done",
                action="STOP",
                confidence=0.9,
                done=True,
                robot={"x": 0.0, "y": 0.0, "heading": 0.0},
                best_candidate=None,
                visible_objects=["Television"],
                backend="ai2thor",
                scene=self.scene,
                completion_status={
                    "complete": True,
                    "outcome": "exact_success",
                    "reason": "target interaction verified",
                },
            )
            result = DemoResult(
                steps=[step],
                video_path="docs/test/demo.mp4",
                summary_path="docs/test/summary.json",
                episode_id=episode_id,
                output_dir="docs/test",
            )
            kwargs["emit"](
                {
                    "protocol_version": PROTOCOL_VERSION,
                    "run_id": run_id,
                    "episode_id": episode_id,
                    "event": "episode_completed",
                    "event_seq": 3,
                    "event_id": f"{run_id}:{episode_id}:3",
                    "sequence": 3,
                    "payload": {"result": result.to_dict()},
                }
            )
            return result

        transport = httpx.ASGITransport(app=app)
        with (
            patch.object(
                AI2ThorVisualSearchDemo,
                "status",
                return_value=self._status(),
            ),
            patch.object(AI2ThorVisualSearchDemo, "run_demo", new=fake_run_demo),
        ):
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                response = await client.post(
                    "/api/demo/ai2thor/stream",
                    json={
                        "session_id": "stream-dedupe",
                        "instruction": "找到电视",
                    },
                )

        self.assertEqual(response.status_code, 200)
        messages = self._messages(response)
        self.assertEqual(
            [message["event"] for message in messages],
            ["task_parsed", "model_decision", "episode_completed"],
        )
        self.assertEqual(
            [message["sequence"] for message in messages],
            [0, 1, 2],
        )
        self.assertEqual(len([m for m in messages if m.get("terminal")]), 1)

    async def test_events_after_worker_terminal_are_not_forwarded(self):
        def fake_run_demo(self, **kwargs):
            emitter = StreamEventEmitter(kwargs["episode_id"], kwargs["emit"])
            emitter.emit("task_parsed", instruction=kwargs["instruction"])
            step = DemoStep(
                frame_path="docs/test/frame.png",
                observation_path="docs/test/observation.png",
                topdown_path="docs/test/map.png",
                thought="done",
                action="STOP",
                confidence=0.9,
                done=True,
                robot={"x": 0.0, "y": 0.0, "heading": 0.0},
                best_candidate=None,
                visible_objects=["Television"],
                backend="ai2thor",
                scene=self.scene,
                completion_status={
                    "complete": True,
                    "outcome": "exact_success",
                    "reason": "target interaction verified",
                },
            )
            result = DemoResult(
                steps=[step],
                video_path="docs/test/demo.mp4",
                summary_path="docs/test/summary.json",
                episode_id=kwargs["episode_id"],
                output_dir="docs/test",
            )
            emitter.emit("episode_completed", result=result.to_dict())
            kwargs["emit"](
                {
                    "event": "step_completed",
                    "sequence": 3,
                    "event_seq": 3,
                    "event_id": "late-step-after-terminal",
                    "payload": {"step_id": 99, "step": step.__dict__},
                }
            )
            return result

        transport = httpx.ASGITransport(app=app)
        with (
            patch.object(
                AI2ThorVisualSearchDemo,
                "status",
                return_value=self._status(),
            ),
            patch.object(AI2ThorVisualSearchDemo, "run_demo", new=fake_run_demo),
        ):
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                response = await client.post(
                    "/api/demo/ai2thor/stream",
                    json={
                        "session_id": "stream-terminal-isolation",
                        "instruction": "找到电视",
                    },
                )

        self.assertEqual(response.status_code, 200)
        messages = self._messages(response)
        self.assertEqual(
            [message["event"] for message in messages],
            ["task_parsed", "episode_completed"],
        )
        self.assertEqual(len([m for m in messages if m.get("terminal")]), 1)

    async def test_late_cancel_after_pending_success_keeps_success_terminal(self):
        pending_terminal_created = threading.Event()

        def fake_run_demo(self, **kwargs):
            emitter = StreamEventEmitter(kwargs["episode_id"], kwargs["emit"])
            emitter.emit("task_parsed", instruction=kwargs["instruction"])
            step = DemoStep(
                frame_path="docs/test/frame.png",
                observation_path="docs/test/observation.png",
                topdown_path="docs/test/map.png",
                thought="done",
                action="STOP",
                confidence=0.9,
                done=True,
                robot={"x": 0.0, "y": 0.0, "heading": 0.0},
                best_candidate=None,
                visible_objects=["Television"],
                backend="ai2thor",
                scene=self.scene,
                completion_status={
                    "complete": True,
                    "outcome": "exact_success",
                    "reason": "target interaction verified",
                },
            )
            result = DemoResult(
                steps=[step],
                video_path="docs/test/demo.mp4",
                summary_path="docs/test/summary.json",
                episode_id=kwargs["episode_id"],
                output_dir="docs/test",
            )
            emitter.emit("episode_completed", result=result.to_dict())
            pending_terminal_created.set()
            kwargs["cancel_event"].wait(1.0)
            StreamEventEmitter.raise_if_cancelled(kwargs["cancel_event"])
            return result

        transport = httpx.ASGITransport(app=app)
        with (
            patch.object(
                AI2ThorVisualSearchDemo,
                "status",
                return_value=self._status(),
            ),
            patch.object(AI2ThorVisualSearchDemo, "run_demo", new=fake_run_demo),
        ):
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                request_task = asyncio.create_task(
                    client.post(
                        "/api/demo/ai2thor/stream",
                        json={
                            "session_id": "stream-late-cancel",
                            "instruction": "找到电视",
                        },
                    )
                )
                for _ in range(100):
                    await asyncio.sleep(0.01)
                    if pending_terminal_created.is_set() and active_stream_runs:
                        break
                self.assertTrue(
                    pending_terminal_created.is_set(),
                    "pending terminal was not created",
                )
                run_id = next(iter(active_stream_runs))
                cancel_response = await client.post(
                    f"/api/demo/ai2thor/stream/{run_id}/cancel"
                )
                response = await asyncio.wait_for(request_task, timeout=2.0)

        self.assertEqual(cancel_response.status_code, 200)
        messages = self._messages(response)
        terminal_messages = [
            message for message in messages if message.get("terminal")
        ]
        self.assertEqual(len(terminal_messages), 1)
        terminal = terminal_messages[0]
        self.assertEqual(terminal["event"], "episode_completed")
        self.assertTrue(terminal["task_success"])
        self.assertEqual(
            terminal["terminal_reason"],
            "target interaction verified",
        )
        self.assertNotIn("stream-late-cancel", active_stream_sessions)
        self.assertNotIn(run_id, active_stream_runs)

    async def test_explicit_cancel_targets_run_and_emits_one_terminal_event(self):
        def fake_run_demo(self, **kwargs):
            emitter = StreamEventEmitter(kwargs["episode_id"], kwargs["emit"])
            emitter.emit("task_parsed", instruction=kwargs["instruction"])
            while not kwargs["cancel_event"].wait(0.01):
                time.sleep(0)
            emitter.emit(
                "episode_cancelled",
                message="AI2-THOR run cancelled by client",
            )
            StreamEventEmitter.raise_if_cancelled(kwargs["cancel_event"])

        transport = httpx.ASGITransport(app=app)
        with (
            patch.object(
                AI2ThorVisualSearchDemo,
                "status",
                return_value=self._status(),
            ),
            patch.object(AI2ThorVisualSearchDemo, "run_demo", new=fake_run_demo),
        ):
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                request_task = asyncio.create_task(
                    client.post(
                        "/api/demo/ai2thor/stream",
                        json={
                            "session_id": "stream-cancel",
                            "instruction": "找到电视",
                        },
                    )
                )
                run_id = ""
                for _ in range(100):
                    await asyncio.sleep(0.01)
                    if active_stream_runs:
                        run_id = next(iter(active_stream_runs))
                        break
                self.assertTrue(run_id, "stream run was not registered")
                cancel_response = await client.post(
                    f"/api/demo/ai2thor/stream/{run_id}/cancel"
                )
                response = await asyncio.wait_for(request_task, timeout=2.0)

        self.assertEqual(cancel_response.status_code, 200)
        self.assertEqual(cancel_response.json()["run_id"], run_id)
        self.assertTrue(cancel_response.json()["cancel_requested"])
        self.assertEqual(response.headers["x-run-id"], run_id)
        messages = self._messages(response)
        terminal_messages = [
            message for message in messages if message.get("terminal")
        ]
        self.assertEqual(len(terminal_messages), 1)
        self.assertEqual(terminal_messages[0]["event"], "episode_cancelled")
        self.assertFalse(terminal_messages[0]["task_success"])
        self.assertEqual(
            terminal_messages[0]["terminal_reason"],
            "cancelled",
        )
        self.assertNotIn("stream-cancel", active_stream_sessions)
        self.assertNotIn(run_id, active_stream_runs)

        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            missing = await client.post(
                f"/api/demo/ai2thor/stream/{run_id}/cancel"
            )
        self.assertEqual(missing.status_code, 404)

    async def test_client_disconnect_sets_cancel_and_cleans_up_run(self):
        cancellation_observed = False

        def fake_run_demo(self, **kwargs):
            nonlocal cancellation_observed
            emitter = StreamEventEmitter(kwargs["episode_id"], kwargs["emit"])
            emitter.emit("task_parsed", instruction=kwargs["instruction"])
            cancellation_observed = kwargs["cancel_event"].wait(1.0)
            StreamEventEmitter.raise_if_cancelled(kwargs["cancel_event"])

        transport = httpx.ASGITransport(app=app)
        with (
            patch.object(
                AI2ThorVisualSearchDemo,
                "status",
                return_value=self._status(),
            ),
            patch.object(AI2ThorVisualSearchDemo, "run_demo", new=fake_run_demo),
            patch.object(
                Request,
                "is_disconnected",
                new=AsyncMock(return_value=True),
            ),
        ):
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                response = await client.post(
                    "/api/demo/ai2thor/stream",
                    json={
                        "session_id": "stream-disconnect",
                        "instruction": "找到电视",
                    },
                )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(cancellation_observed)
        terminal_messages = [
            message
            for message in self._messages(response)
            if message.get("terminal")
        ]
        self.assertEqual(len(terminal_messages), 1)
        self.assertEqual(terminal_messages[0]["event"], "episode_cancelled")
        self.assertNotIn("stream-disconnect", active_stream_sessions)
        self.assertNotIn(response.headers["x-run-id"], active_stream_runs)


if __name__ == "__main__":
    unittest.main()
