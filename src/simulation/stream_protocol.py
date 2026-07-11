from __future__ import annotations

import json
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable


StreamCallback = Callable[[dict[str, Any]], None]
PROTOCOL_VERSION = "2.0"
TERMINAL_EVENT = "terminal"


class StreamCancelled(RuntimeError):
    """Raised when a client cancels an active simulator stream."""


class DuplicateTerminalEvent(RuntimeError):
    """Raised when a stream attempts to emit more than one terminal event."""


class StreamClosed(RuntimeError):
    """Raised when an event is emitted after the stream has terminated."""


@dataclass
class StreamEventEmitter:
    episode_id: str
    callback: StreamCallback | None = None
    run_id: str | None = None
    protocol_version: str = PROTOCOL_VERSION
    _sequence: int = field(default=0, init=False, repr=False)
    _terminal_message: dict[str, Any] | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _lock: threading.RLock = field(
        default_factory=threading.RLock,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        self.episode_id = self._require_identifier("episode_id", self.episode_id)
        if self.run_id is None:
            self.run_id = uuid.uuid4().hex
        else:
            self.run_id = self._require_identifier("run_id", self.run_id)
        self.protocol_version = self._require_identifier(
            "protocol_version",
            self.protocol_version,
        )

    def emit(self, event: str, **payload: Any) -> dict[str, Any]:
        if not isinstance(event, str) or not event.strip():
            raise ValueError("stream event name must not be empty")
        event = event.strip()
        with self._lock:
            if self._terminal_message is not None:
                if event == TERMINAL_EVENT:
                    raise DuplicateTerminalEvent(
                        "stream already emitted its terminal event"
                    )
                raise StreamClosed("stream is closed after its terminal event")

            task_success: bool | None = None
            terminal_reason: str | None = None
            if event == TERMINAL_EVENT:
                task_success, terminal_reason = self._validate_terminal_payload(
                    payload
                )

            event_seq = self._sequence
            message = {
                "protocol_version": self.protocol_version,
                "run_id": self.run_id,
                "episode_id": self.episode_id,
                "event": event,
                "event_seq": event_seq,
                "event_id": f"{self.run_id}:{self.episode_id}:{event_seq}",
                # Compatibility alias for existing stream consumers.
                "sequence": event_seq,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "task_success": task_success,
                "terminal_reason": terminal_reason,
                "payload": payload,
            }
            self._sequence += 1
            if event == TERMINAL_EVENT:
                self._terminal_message = message
            if self.callback is not None:
                # Serialize callback delivery with sequence allocation so
                # concurrent producers cannot deliver events out of order.
                self.callback(message)
            return message

    def emit_terminal(
        self,
        *,
        task_success: bool,
        terminal_reason: str,
        **payload: Any,
    ) -> dict[str, Any]:
        """Emit the stream's one terminal event."""
        return self.emit(
            TERMINAL_EVENT,
            task_success=task_success,
            terminal_reason=terminal_reason,
            **payload,
        )

    @property
    def terminal_emitted(self) -> bool:
        with self._lock:
            return self._terminal_message is not None

    @property
    def terminal_message(self) -> dict[str, Any] | None:
        with self._lock:
            return self._terminal_message

    @staticmethod
    def _require_identifier(name: str, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string")
        return value.strip()

    @staticmethod
    def _validate_terminal_payload(
        payload: dict[str, Any],
    ) -> tuple[bool, str]:
        task_success = payload.get("task_success")
        terminal_reason = payload.get("terminal_reason")
        if not isinstance(task_success, bool):
            raise ValueError("terminal task_success must be a boolean")
        if not isinstance(terminal_reason, str) or not terminal_reason.strip():
            raise ValueError(
                "terminal terminal_reason must be a non-empty string"
            )
        payload["terminal_reason"] = terminal_reason.strip()
        return task_success, terminal_reason.strip()

    @staticmethod
    def raise_if_cancelled(cancel_event: threading.Event | None) -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise StreamCancelled("AI2-THOR run cancelled by client")


def encode_ndjson(message: dict[str, Any]) -> bytes:
    return (
        json.dumps(message, ensure_ascii=False, separators=(",", ":")) + "\n"
    ).encode("utf-8")
