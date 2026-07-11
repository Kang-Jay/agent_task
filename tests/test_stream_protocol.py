from __future__ import annotations

import json
import threading
import unittest

from src.simulation.stream_protocol import (
    DuplicateTerminalEvent,
    PROTOCOL_VERSION,
    StreamCancelled,
    StreamClosed,
    StreamEventEmitter,
    encode_ndjson,
)


class StreamProtocolTests(unittest.TestCase):
    def test_emitter_sequences_messages_and_calls_callback(self):
        received: list[dict[str, object]] = []
        emitter = StreamEventEmitter("episode-1", received.append)

        first = emitter.emit("task_parsed", task_type="visual_search")
        second = emitter.emit("step_completed", step_id=0)

        self.assertEqual(first["sequence"], 0)
        self.assertEqual(second["sequence"], 1)
        self.assertEqual(first["event_seq"], 0)
        self.assertEqual(second["event_seq"], 1)
        self.assertEqual(received, [first, second])
        self.assertEqual(second["payload"]["step_id"], 0)
        self.assertEqual(first["protocol_version"], PROTOCOL_VERSION)
        self.assertEqual(first["episode_id"], "episode-1")
        self.assertEqual(first["run_id"], second["run_id"])
        self.assertEqual(
            first["event_id"],
            f"{first['run_id']}:episode-1:0",
        )
        self.assertIsNone(first["task_success"])
        self.assertIsNone(first["terminal_reason"])

    def test_ndjson_is_utf8_and_round_trips(self):
        message = StreamEventEmitter("episode-2").emit(
            "model_decision",
            thought="继续搜索电视",
        )
        encoded = encode_ndjson(message)

        self.assertTrue(encoded.endswith(b"\n"))
        self.assertEqual(json.loads(encoded.decode("utf-8")), message)

    def test_explicit_run_id_and_protocol_version_are_preserved(self):
        emitter = StreamEventEmitter(
            "episode-explicit",
            run_id="run-explicit",
            protocol_version="2.1-test",
        )

        message = emitter.emit("task_started")

        self.assertEqual(message["run_id"], "run-explicit")
        self.assertEqual(message["protocol_version"], "2.1-test")
        self.assertEqual(
            message["event_id"],
            "run-explicit:episode-explicit:0",
        )

    def test_default_run_ids_isolate_independent_runs(self):
        first = StreamEventEmitter("same-episode").emit("task_started")
        second = StreamEventEmitter("same-episode").emit("task_started")

        self.assertNotEqual(first["run_id"], second["run_id"])
        self.assertNotEqual(first["event_id"], second["event_id"])

    def test_terminal_event_has_required_outcome_and_closes_stream(self):
        received: list[dict[str, object]] = []
        emitter = StreamEventEmitter(
            "episode-terminal",
            received.append,
            run_id="run-terminal",
        )

        terminal = emitter.emit_terminal(
            task_success=True,
            terminal_reason="task_completed",
            result={"steps": 3},
        )

        self.assertTrue(emitter.terminal_emitted)
        self.assertIs(emitter.terminal_message, terminal)
        self.assertEqual(terminal["event"], "terminal")
        self.assertTrue(terminal["task_success"])
        self.assertEqual(terminal["terminal_reason"], "task_completed")
        self.assertTrue(terminal["payload"]["task_success"])
        self.assertEqual(
            terminal["payload"]["terminal_reason"],
            "task_completed",
        )
        self.assertEqual(received, [terminal])
        with self.assertRaises(DuplicateTerminalEvent):
            emitter.emit_terminal(
                task_success=False,
                terminal_reason="duplicate",
            )
        with self.assertRaises(StreamClosed):
            emitter.emit("step_completed", step_id=4)
        self.assertEqual(received, [terminal])

    def test_direct_terminal_event_requires_valid_outcome(self):
        emitter = StreamEventEmitter("episode-invalid-terminal")

        with self.assertRaises(ValueError):
            emitter.emit("terminal", terminal_reason="missing_success")
        with self.assertRaises(ValueError):
            emitter.emit(
                "terminal",
                task_success="yes",
                terminal_reason="invalid_success",
            )
        with self.assertRaises(ValueError):
            emitter.emit(
                "terminal",
                task_success=False,
                terminal_reason=" ",
            )
        self.assertFalse(emitter.terminal_emitted)
        self.assertEqual(emitter.emit("task_started")["sequence"], 0)

    def test_concurrent_emits_have_monotonic_delivery_and_unique_ids(self):
        received: list[dict[str, object]] = []
        emitter = StreamEventEmitter(
            "episode-concurrent",
            received.append,
            run_id="run-concurrent",
        )
        start = threading.Barrier(9)

        def produce(worker_id: int) -> None:
            start.wait()
            emitter.emit("worker_event", worker_id=worker_id)

        threads = [
            threading.Thread(target=produce, args=(worker_id,))
            for worker_id in range(8)
        ]
        for thread in threads:
            thread.start()
        start.wait()
        for thread in threads:
            thread.join()

        self.assertEqual(
            [message["event_seq"] for message in received],
            list(range(8)),
        )
        self.assertEqual(
            [message["sequence"] for message in received],
            list(range(8)),
        )
        self.assertEqual(
            len({message["event_id"] for message in received}),
            8,
        )

    def test_concurrent_terminal_attempts_emit_exactly_one_terminal(self):
        received: list[dict[str, object]] = []
        emitter = StreamEventEmitter(
            "episode-terminal-race",
            received.append,
            run_id="run-terminal-race",
        )
        start = threading.Barrier(7)
        outcomes: list[str] = []
        outcomes_lock = threading.Lock()

        def terminate(worker_id: int) -> None:
            start.wait()
            try:
                emitter.emit_terminal(
                    task_success=(worker_id == 0),
                    terminal_reason=f"worker_{worker_id}",
                )
                outcome = "emitted"
            except DuplicateTerminalEvent:
                outcome = "duplicate"
            with outcomes_lock:
                outcomes.append(outcome)

        threads = [
            threading.Thread(target=terminate, args=(worker_id,))
            for worker_id in range(6)
        ]
        for thread in threads:
            thread.start()
        start.wait()
        for thread in threads:
            thread.join()

        self.assertEqual(outcomes.count("emitted"), 1)
        self.assertEqual(outcomes.count("duplicate"), 5)
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0]["event"], "terminal")

    def test_cancelled_event_raises(self):
        cancel_event = threading.Event()
        cancel_event.set()

        with self.assertRaises(StreamCancelled):
            StreamEventEmitter.raise_if_cancelled(cancel_event)


if __name__ == "__main__":
    unittest.main()
