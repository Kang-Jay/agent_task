from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable


StreamCallback = Callable[[dict[str, Any]], None]


class StreamCancelled(RuntimeError):
    """Raised when a client cancels an active simulator stream."""


@dataclass
class StreamEventEmitter:
    episode_id: str
    callback: StreamCallback | None = None
    _sequence: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def emit(self, event: str, **payload: Any) -> dict[str, Any]:
        if not event:
            raise ValueError("stream event name must not be empty")
        with self._lock:
            message = {
                "event": event,
                "sequence": self._sequence,
                "episode_id": self.episode_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "payload": payload,
            }
            self._sequence += 1
        if self.callback is not None:
            self.callback(message)
        return message

    @staticmethod
    def raise_if_cancelled(cancel_event: threading.Event | None) -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise StreamCancelled("AI2-THOR run cancelled by client")


def encode_ndjson(message: dict[str, Any]) -> bytes:
    return (
        json.dumps(message, ensure_ascii=False, separators=(",", ":")) + "\n"
    ).encode("utf-8")
