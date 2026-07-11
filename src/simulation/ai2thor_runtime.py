from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import importlib
import math
import os
from typing import Any


GRID_ALIGNED_ROTATIONS = (0.0, 90.0, 180.0, 270.0)
DEFAULT_GRID_SIZE_METERS = 0.25
DEFAULT_AI2THOR_PLATFORM = "CloudRendering"


def ai2thor_platform_name() -> str:
    """Return the requested AI2-THOR platform name.

    CloudRendering remains the default for existing Linux/remote runs. Local
    WSLg can set AI2THOR_PLATFORM=Linux64 to use the real Linux Unity build
    through D3D12-backed OpenGL when Vulkan CloudRendering is unavailable.
    """
    return (os.getenv("AI2THOR_PLATFORM") or DEFAULT_AI2THOR_PLATFORM).strip()


def resolve_ai2thor_platform(platform_name: str | None = None) -> Any:
    """Resolve an ai2thor.platform class by name with a clear error."""
    requested = (platform_name or ai2thor_platform_name()).strip()
    if not requested:
        requested = DEFAULT_AI2THOR_PLATFORM

    try:
        ai2thor_platform = importlib.import_module("ai2thor.platform")
    except ModuleNotFoundError as exc:
        if requested == DEFAULT_AI2THOR_PLATFORM:
            return None
        raise ValueError(
            "This AI2-THOR installation does not expose ai2thor.platform; "
            "install a platform-aware AI2-THOR build or leave AI2THOR_PLATFORM "
            f"unset for legacy runtime behavior. Requested: {requested!r}."
        ) from exc

    try:
        return getattr(ai2thor_platform, requested)
    except AttributeError as exc:
        available = sorted(
            name
            for name in dir(ai2thor_platform)
            if name and name[0].isupper() and not name.startswith("_")
        )
        raise ValueError(
            f"Unsupported AI2-THOR platform {requested!r}; "
            f"available platforms: {', '.join(available)}"
        ) from exc


def ai2thor_platform_kwargs(platform_name: str | None = None) -> dict[str, Any]:
    """Return Controller kwargs for platform-aware and legacy AI2-THOR builds."""
    platform = resolve_ai2thor_platform(platform_name)
    if platform is None:
        return {}
    return {"platform": platform}


def _metadata_snapshot(event: Any) -> dict[str, Any]:
    """Return a detached snapshot of an AI2-THOR event's metadata."""
    metadata = getattr(event, "metadata", None)
    if metadata is None:
        return {}
    if not isinstance(metadata, dict):
        raise TypeError("AI2-THOR event metadata must be a dictionary")
    return deepcopy(metadata)


@dataclass(frozen=True)
class RuntimeActionExecution:
    """Auditable result of one exact, parameterized Unity action call."""

    action: str
    args: dict[str, Any]
    before_metadata: dict[str, Any]
    after_metadata: dict[str, Any]
    success: bool
    error_message: str
    action_return: Any
    inventory_before: list[Any]
    inventory_after: list[Any]
    event: Any = field(repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "args": deepcopy(self.args),
            "before_metadata": deepcopy(self.before_metadata),
            "after_metadata": deepcopy(self.after_metadata),
            "success": self.success,
            "last_action_success": self.success,
            "error_message": self.error_message,
            "action_return": deepcopy(self.action_return),
            "inventory_before": deepcopy(self.inventory_before),
            "inventory_after": deepcopy(self.inventory_after),
        }


def execute_controller_action(
    controller: Any,
    *,
    action: str,
    args: dict[str, Any] | None = None,
) -> RuntimeActionExecution:
    """Execute an exact AI2-THOR action and retain before/after audit state.

    Action validation and abstract-name normalization belong to the action
    catalog. This runtime boundary deliberately forwards the native action name
    and parameter payload unchanged to ``Controller.step``.
    """
    if not isinstance(action, str) or not action.strip():
        raise ValueError("action must be a non-empty string")
    if args is not None and not isinstance(args, dict):
        raise TypeError("args must be a dictionary or None")

    call_args = deepcopy(args or {})
    if "action" in call_args:
        raise ValueError("args must not contain the reserved 'action' key")

    before_event = getattr(controller, "last_event", None)
    before_metadata = _metadata_snapshot(before_event)

    # Do not catch Controller.step exceptions. Transport, Unity process, and
    # programming failures must retain their original type and traceback.
    event = controller.step(action=action, **call_args)
    after_metadata = _metadata_snapshot(event)
    success = bool(after_metadata.get("lastActionSuccess", False))

    return RuntimeActionExecution(
        action=action,
        args=call_args,
        before_metadata=before_metadata,
        after_metadata=after_metadata,
        success=success,
        error_message=str(after_metadata.get("errorMessage") or ""),
        action_return=deepcopy(after_metadata.get("actionReturn")),
        inventory_before=deepcopy(before_metadata.get("inventoryObjects", [])),
        inventory_after=deepcopy(after_metadata.get("inventoryObjects", [])),
        event=event,
    )


def is_grid_aligned_rotation(rotate_step_degrees: float) -> bool:
    """Return whether AI2-THOR can combine this rotation with snapToGrid."""
    normalized = float(rotate_step_degrees) % 360.0
    return any(
        math.isclose(normalized, allowed, abs_tol=1e-6)
        for allowed in GRID_ALIGNED_ROTATIONS
    )


def should_snap_to_grid(*, mode: str, rotate_step_degrees: float) -> bool:
    """Enable grid snapping only for compatible default-agent rotations."""
    return mode.lower() == "default" and is_grid_aligned_rotation(
        rotate_step_degrees
    )


def create_controller_safely(
    controller_type: type[Any],
    **kwargs: Any,
) -> Any:
    """Retain the partial Controller so failed initialization can be cleaned up."""
    controller = controller_type.__new__(controller_type)
    try:
        controller_type.__init__(controller, **kwargs)
    except BaseException:
        stop = getattr(controller, "stop", None)
        if callable(stop):
            try:
                stop()
            except BaseException:
                pass
        raise
    return controller
