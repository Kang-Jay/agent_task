from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from src.task.config import ROOT


CATALOG_PATH = ROOT / "configs" / "ai2thor_actions_v5.json"
ACTOR_LEVELS = {"agent": 0, "manual": 1, "system": 2}
ABSTRACT_ACTION_ALIASES = {
    "MOVE_FORWARD": "MoveAhead",
    "MOVE_BACK": "MoveBack",
    "MOVE_LEFT": "MoveLeft",
    "MOVE_RIGHT": "MoveRight",
    "TURN_LEFT": "RotateLeft",
    "TURN_RIGHT": "RotateRight",
    "LOOK_UP": "LookUp",
    "LOOK_DOWN": "LookDown",
    "CROUCH": "Crouch",
    "STAND": "Stand",
    "INSPECT": "Pass",
    "STOP": "Done",
}


@dataclass(frozen=True)
class ActionValidation:
    valid: bool
    action: str
    mode: str
    actor: str
    normalized_args: dict[str, Any] = field(default_factory=dict)
    matched_overload: dict[str, Any] | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "action": self.action,
            "mode": self.mode,
            "actor": self.actor,
            "normalized_args": self.normalized_args,
            "matched_overload": self.matched_overload,
            "errors": self.errors,
            "warnings": self.warnings,
        }


@dataclass(frozen=True)
class ActionExecution:
    action: str
    mode: str
    args: dict[str, Any]
    success: bool
    error_message: str
    action_return: Any
    validation: ActionValidation
    event: Any = field(repr=False, default=None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "mode": self.mode,
            "args": self.args,
            "success": self.success,
            "error_message": self.error_message,
            "action_return": self.action_return,
            "validation": self.validation.to_dict(),
        }


class AI2ThorActionCatalog:
    def __init__(self, path: Path = CATALOG_PATH):
        self.path = path
        self._payload = json.loads(path.read_text(encoding="utf-8"))
        self._actions = {action["name"]: action for action in self._payload["actions"]}

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in self._payload.items()
            if key != "actions"
        }

    @property
    def actions(self) -> list[dict[str, Any]]:
        return [dict(action) for action in self._payload["actions"]]

    def summary(self) -> dict[str, Any]:
        return self.metadata

    def verify_runtime(
        self,
        *,
        ai2thor_version: str,
        build_commit: str,
    ) -> dict[str, Any]:
        expected_version = str(self._payload["ai2thor_version"])
        expected_commit = str(self._payload["source"]["commit"])
        errors: list[str] = []
        if str(ai2thor_version) != expected_version:
            errors.append(
                f"AI2-THOR Python version mismatch: expected {expected_version}, "
                f"received {ai2thor_version}"
            )
        if str(build_commit) != expected_commit:
            errors.append(
                f"AI2-THOR Unity build mismatch: expected {expected_commit}, "
                f"received {build_commit}"
            )
        if errors:
            raise RuntimeError("; ".join(errors))
        return {
            "matched": True,
            "ai2thor_version": expected_version,
            "build_commit": expected_commit,
        }

    def verify_installed_runtime(self) -> dict[str, Any]:
        import ai2thor  # type: ignore
        from ai2thor import build  # type: ignore

        return self.verify_runtime(
            ai2thor_version=str(getattr(ai2thor, "__version__", "unknown")),
            build_commit=str(getattr(build, "COMMIT_ID", "unknown")),
        )

    def resolve_name(self, action: str) -> str:
        return ABSTRACT_ACTION_ALIASES.get(action, action)

    def get(self, action: str) -> dict[str, Any] | None:
        return self._actions.get(self.resolve_name(action))

    def list_actions(
        self,
        *,
        mode: str,
        actor: Literal["agent", "manual", "system"] = "agent",
        include_internal: bool = False,
    ) -> list[dict[str, Any]]:
        if mode not in self._payload["mode_controllers"]:
            raise ValueError(f"Unknown AI2-THOR agent mode: {mode}")
        actor_level = ACTOR_LEVELS[actor]
        actions: list[dict[str, Any]] = []
        for action in self._payload["actions"]:
            if not action.get("runtime_available", True):
                continue
            if mode not in action["modes"]:
                continue
            exposure = action["exposure"]
            if exposure == "internal":
                if not include_internal:
                    continue
            elif ACTOR_LEVELS[exposure] > actor_level:
                continue
            if actor == "agent" and mode not in action.get("planner_modes", []):
                continue
            actions.append(action)
        return actions

    def validate(
        self,
        *,
        mode: str,
        action: str,
        args: dict[str, Any] | None = None,
        actor: Literal["agent", "manual", "system"] = "agent",
    ) -> ActionValidation:
        normalized_action = self.resolve_name(action)
        normalized_args = dict(args or {})
        errors: list[str] = []
        warnings: list[str] = []
        if mode not in self._payload["mode_controllers"]:
            errors.append(f"Unknown AI2-THOR agent mode: {mode}")
            return ActionValidation(False, normalized_action, mode, actor, normalized_args, errors=errors)
        if actor not in ACTOR_LEVELS:
            errors.append(f"Unknown actor level: {actor}")
            return ActionValidation(False, normalized_action, mode, actor, normalized_args, errors=errors)

        spec = self._actions.get(normalized_action)
        if spec is None:
            errors.append(f"Unknown AI2-THOR action: {normalized_action}")
            return ActionValidation(False, normalized_action, mode, actor, normalized_args, errors=errors)
        if not spec.get("runtime_available", True):
            errors.append(
                f"Action {normalized_action} is listed but unavailable in this AI2-THOR build"
            )
        if mode not in spec["modes"]:
            errors.append(f"Action {normalized_action} is not available in agent mode {mode}")
        exposure = spec["exposure"]
        if exposure == "internal":
            errors.append(f"Action {normalized_action} is an internal Unity method and is never executable")
        elif ACTOR_LEVELS[exposure] > ACTOR_LEVELS[actor]:
            errors.append(
                f"Action {normalized_action} requires actor level {exposure}; received {actor}"
            )
        elif actor == "agent" and mode not in spec.get("planner_modes", []):
            errors.append(
                f"Action {normalized_action} is not exposed to the autonomous planner in mode {mode}"
            )
        if errors:
            return ActionValidation(False, normalized_action, mode, actor, normalized_args, errors=errors)

        overloads = spec.get("overloads_by_mode", {}).get(mode, [])
        if spec.get("manager_action") and not overloads:
            warnings.append("Manager action parameters are validated by the AI2-THOR runtime")
            return ActionValidation(
                True,
                normalized_action,
                mode,
                actor,
                normalized_args,
                warnings=warnings,
            )
        matched, overload_errors = self._match_overload(overloads, normalized_args)
        if matched is None:
            errors.extend(overload_errors)
        elif matched.get("legacy_server_action"):
            warnings.append(
                "Legacy ServerAction overload accepts a dynamic payload; runtime validation remains authoritative"
            )
        return ActionValidation(
            valid=not errors,
            action=normalized_action,
            mode=mode,
            actor=actor,
            normalized_args=normalized_args,
            matched_overload=matched,
            errors=errors,
            warnings=warnings,
        )

    def _match_overload(
        self,
        overloads: list[dict[str, Any]],
        args: dict[str, Any],
    ) -> tuple[dict[str, Any] | None, list[str]]:
        failures: list[str] = []
        dynamic_overload: dict[str, Any] | None = None
        candidates: list[tuple[int, int, dict[str, Any]]] = []
        for overload in overloads:
            if overload.get("legacy_server_action"):
                dynamic_overload = overload
                continue
            parameters = overload.get("parameters", [])
            accepted = {parameter["name"] for parameter in parameters}
            required = {parameter["name"] for parameter in parameters if parameter["required"]}
            missing = sorted(required - set(args))
            unknown = sorted(set(args) - accepted)
            type_errors = [
                error
                for parameter in parameters
                if parameter["name"] in args
                for error in self._validate_type(parameter, args[parameter["name"]])
            ]
            if missing or unknown or type_errors:
                failures.append(
                    f"{overload['declaring_class']}({', '.join(sorted(accepted))}): "
                    f"missing={missing}, unknown={unknown}, type_errors={type_errors}"
                )
                continue
            candidates.append((len(set(args) & accepted), -len(parameters), overload))
        if candidates:
            candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
            return candidates[0][2], []
        if dynamic_overload is not None:
            return dynamic_overload, []
        if not overloads and not args:
            return {}, []
        return None, [
            "No AI2-THOR overload matched the supplied arguments",
            *failures[:5],
        ]

    @staticmethod
    def _validate_type(parameter: dict[str, Any], value: Any) -> list[str]:
        if value is None:
            return [] if parameter["type"].endswith("?") else [f"{parameter['name']} cannot be null"]
        type_name = parameter["type"].rstrip("?")
        if type_name in {"float", "double", "decimal"} and (
            not isinstance(value, (int, float)) or isinstance(value, bool)
        ):
            return [f"{parameter['name']} must be numeric"]
        if type_name in {"int", "long", "short"} and (
            not isinstance(value, int) or isinstance(value, bool)
        ):
            return [f"{parameter['name']} must be an integer"]
        if type_name == "bool" and not isinstance(value, bool):
            return [f"{parameter['name']} must be boolean"]
        if type_name == "string" and not isinstance(value, str):
            return [f"{parameter['name']} must be a string"]
        if (type_name.endswith("[]") or type_name.startswith(("List<", "IEnumerable<"))) and not isinstance(value, list):
            return [f"{parameter['name']} must be a list"]
        if type_name in {"Vector3", "Vector2", "Quaternion", "Color"} and not isinstance(value, dict):
            return [f"{parameter['name']} must be an object with coordinate fields"]
        return []


class AI2ThorActionExecutor:
    def __init__(self, catalog: AI2ThorActionCatalog | None = None):
        self.catalog = catalog or AI2ThorActionCatalog()

    def execute(
        self,
        controller: Any,
        *,
        mode: str,
        action: str,
        args: dict[str, Any] | None = None,
        actor: Literal["agent", "manual", "system"] = "agent",
    ) -> ActionExecution:
        validation = self.catalog.validate(mode=mode, action=action, args=args, actor=actor)
        if not validation.valid:
            raise ValueError("; ".join(validation.errors))
        event = controller.step(action=validation.action, **validation.normalized_args)
        metadata = getattr(event, "metadata", {}) or {}
        return ActionExecution(
            action=validation.action,
            mode=mode,
            args=validation.normalized_args,
            success=bool(metadata.get("lastActionSuccess", False)),
            error_message=str(metadata.get("errorMessage") or ""),
            action_return=metadata.get("actionReturn"),
            validation=validation,
            event=event,
        )
