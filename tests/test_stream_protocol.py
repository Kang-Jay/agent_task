from __future__ import annotations

import json
import threading
import unittest

from src.simulation.stream_protocol import (
    StreamCancelled,
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
        self.assertEqual(received, [first, second])
        self.assertEqual(second["payload"]["step_id"], 0)

    def test_ndjson_is_utf8_and_round_trips(self):
        message = StreamEventEmitter("episode-2").emit(
            "model_decision",
            thought="继续搜索电视",
        )
        encoded = encode_ndjson(message)

        self.assertTrue(encoded.endswith(b"\n"))
        self.assertEqual(json.loads(encoded.decode("utf-8")), message)

    def test_cancelled_event_raises(self):
        cancel_event = threading.Event()
        cancel_event.set()

        with self.assertRaises(StreamCancelled):
            StreamEventEmitter.raise_if_cancelled(cancel_event)


if __name__ == "__main__":
    unittest.main()
