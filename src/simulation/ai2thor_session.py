from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Literal

import numpy as np
from PIL import Image

from src.simulation.ai2thor_actions import AI2ThorActionCatalog, AI2ThorActionExecutor
from src.simulation.ai2thor_postconditions import AI2ThorPostconditionVerifier
from src.simulation.ai2thor_runtime import (
    create_controller_safely,
    should_snap_to_grid,
)
from src.vision.image_io import image_to_data_url


ControllerFactory = Callable[..., Any]


@dataclass
class AI2ThorSession:
    session_id: str
    scene: str
    mode: str
    controller: Any
    last_event: Any
    created_at: str
    worker: ThreadPoolExecutor


class AI2ThorSessionManager:
    def __init__(
        self,
        *,
        controller_factory: ControllerFactory | None = None,
        catalog: AI2ThorActionCatalog | None = None,
    ):
        self._controller_factory = controller_factory
        self.catalog = catalog or AI2ThorActionCatalog()
        self.executor = AI2ThorActionExecutor(self.catalog)
        self.postconditions = AI2ThorPostconditionVerifier()
        self._sessions: dict[str, AI2ThorSession] = {}
        self._lock = threading.RLock()

    def start(
        self,
        *,
        session_id: str,
        scene: str,
        mode: str = "default",
        width: int = 960,
        height: int = 540,
        quality: str = "Low",
        grid_size: float = 0.25,
        rotate_step_degrees: float = 90.0,
        render_instance_segmentation: bool = True,
    ) -> dict[str, Any]:
        if not session_id.strip():
            raise ValueError("session_id must not be empty")
        if mode not in self.catalog.summary()["mode_controllers"]:
            raise ValueError(f"Unsupported AI2-THOR agent mode: {mode}")
        if width < 160 or height < 120:
            raise ValueError("AI2-THOR frame dimensions are too small")
        with self._lock:
            self.close(session_id)
            worker = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix=f"ai2thor-{session_id}",
            )
            try:
                session = worker.submit(
                    self._initialize_session,
                    worker=worker,
                    session_id=session_id,
                    scene=scene,
                    mode=mode,
                    width=width,
                    height=height,
                    quality=quality,
                    grid_size=grid_size,
                    rotate_step_degrees=rotate_step_degrees,
                    render_instance_segmentation=render_instance_segmentation,
                ).result()
            except Exception:
                worker.shutdown(wait=True, cancel_futures=True)
                raise
            self._sessions[session_id] = session
            return self._snapshot(session, session.last_event)

    def execute(
        self,
        *,
        session_id: str,
        action: str,
        args: dict[str, Any] | None = None,
        actor: Literal["agent", "manual", "system"] = "manual",
    ) -> dict[str, Any]:
        with self._lock:
            session = self._require_session(session_id)
            return session.worker.submit(
                self._execute_in_worker,
                session=session,
                action=action,
                args=args,
                actor=actor,
            ).result()

    def snapshot(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            session = self._require_session(session_id)
            return self._snapshot(session, session.last_event)

    def list_sessions(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {
                    "session_id": session.session_id,
                    "scene": session.scene,
                    "mode": session.mode,
                    "created_at": session.created_at,
                }
                for session in self._sessions.values()
            ]

    def close(self, session_id: str) -> bool:
        with self._lock:
            session = self._sessions.pop(session_id, None)
            if session is None:
                return False
            try:
                session.worker.submit(session.controller.stop).result()
            finally:
                session.worker.shutdown(wait=True, cancel_futures=True)
            return True

    def close_all(self) -> None:
        with self._lock:
            for session_id in list(self._sessions):
                self.close(session_id)

    def _require_session(self, session_id: str) -> AI2ThorSession:
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(f"AI2-THOR session not found: {session_id}")
        return session

    def _create_controller(self, **kwargs: Any) -> Any:
        if self._controller_factory is not None:
            return self._controller_factory(**kwargs)
        from ai2thor.controller import Controller  # type: ignore
        from ai2thor.platform import CloudRendering  # type: ignore

        self.catalog.verify_installed_runtime()
        kwargs["platform"] = CloudRendering
        return create_controller_safely(Controller, **kwargs)

    def _initialize_session(
        self,
        *,
        worker: ThreadPoolExecutor,
        session_id: str,
        scene: str,
        mode: str,
        width: int,
        height: int,
        quality: str,
        grid_size: float,
        rotate_step_degrees: float,
        render_instance_segmentation: bool,
    ) -> AI2ThorSession:
        controller = self._create_controller(
            scene=scene,
            agentMode=mode,
            width=width,
            height=height,
            quality=quality,
            gridSize=grid_size,
            rotateStepDegrees=rotate_step_degrees,
            snapToGrid=should_snap_to_grid(
                mode=mode,
                rotate_step_degrees=rotate_step_degrees,
            ),
            renderInstanceSegmentation=render_instance_segmentation,
        )
        return AI2ThorSession(
            session_id=session_id,
            scene=scene,
            mode=mode,
            controller=controller,
            last_event=controller.last_event,
            created_at=datetime.now(timezone.utc).isoformat(),
            worker=worker,
        )

    def _execute_in_worker(
        self,
        *,
        session: AI2ThorSession,
        action: str,
        args: dict[str, Any] | None,
        actor: Literal["agent", "manual", "system"],
    ) -> dict[str, Any]:
        before_metadata = getattr(session.last_event, "metadata", {}) or {}
        execution = self.executor.execute(
            session.controller,
            mode=session.mode,
            action=action,
            args=args,
            actor=actor,
        )
        session.last_event = execution.event
        after_metadata = getattr(execution.event, "metadata", {}) or {}
        postcondition = self.postconditions.verify(
            action=execution.action,
            args=execution.args,
            before=before_metadata,
            after=after_metadata,
            runtime_success=execution.success,
        )
        snapshot = self._snapshot(session, execution.event)
        snapshot["execution"] = execution.to_dict()
        snapshot["postcondition"] = postcondition.to_dict()
        return snapshot

    def _snapshot(self, session: AI2ThorSession, event: Any) -> dict[str, Any]:
        metadata = getattr(event, "metadata", {}) or {}
        agent = metadata.get("agent", {})
        visible_objects = [
            {
                "objectId": item.get("objectId"),
                "objectType": item.get("objectType"),
                "distance": item.get("distance"),
                "pickupable": item.get("pickupable"),
                "receptacle": item.get("receptacle"),
                "openable": item.get("openable"),
                "isOpen": item.get("isOpen"),
                "toggleable": item.get("toggleable"),
                "isToggled": item.get("isToggled"),
            }
            for item in metadata.get("objects", [])
            if item.get("visible")
        ]
        frame = getattr(event, "frame", None)
        frame_url = None
        if frame is not None:
            frame_array = np.asarray(frame)
            frame_url = image_to_data_url(Image.fromarray(frame_array).convert("RGB"))
        return {
            "session_id": session.session_id,
            "scene": session.scene,
            "mode": session.mode,
            "created_at": session.created_at,
            "frame": frame_url,
            "last_action": metadata.get("lastAction"),
            "last_action_success": bool(metadata.get("lastActionSuccess", False)),
            "error_message": str(metadata.get("errorMessage") or ""),
            "action_return": self._json_safe(metadata.get("actionReturn")),
            "agent": self._json_safe(agent),
            "inventory_objects": self._json_safe(metadata.get("inventoryObjects", [])),
            "visible_objects": self._json_safe(visible_objects),
        }

    def _json_safe(self, value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {str(key): self._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [self._json_safe(item) for item in value]
        if hasattr(value, "tolist"):
            return self._json_safe(value.tolist())
        return str(value)
