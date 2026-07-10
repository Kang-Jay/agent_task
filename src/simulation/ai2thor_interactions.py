from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


OBJECT_ID_ACTIONS = {
    "PickupObject",
    "PutObject",
    "OpenObject",
    "CloseObject",
    "ToggleObjectOn",
    "ToggleObjectOff",
    "SliceObject",
    "BreakObject",
    "CleanObject",
    "DirtyObject",
    "FillObjectWithLiquid",
    "EmptyLiquidFromObject",
    "UseUpObject",
}

ACTION_REQUIREMENTS: dict[str, tuple[str, bool]] = {
    "PickupObject": ("pickupable", True),
    "PutObject": ("receptacle", True),
    "OpenObject": ("openable", True),
    "CloseObject": ("openable", True),
    "ToggleObjectOn": ("toggleable", True),
    "ToggleObjectOff": ("toggleable", True),
    "SliceObject": ("sliceable", True),
    "BreakObject": ("breakable", True),
    "CleanObject": ("dirtyable", True),
    "DirtyObject": ("dirtyable", True),
    "FillObjectWithLiquid": ("canFillWithLiquid", True),
    "EmptyLiquidFromObject": ("canFillWithLiquid", True),
    "UseUpObject": ("canBeUsedUp", True),
}

SEMANTIC_ARG_KEYS = {
    "object",
    "objectType",
    "target",
    "targetObject",
    "receptacle",
    "receptacleType",
}

CONTEXT_OBJECT_KEYS = (
    "objectId",
    "objectType",
    "name",
    "distance",
    "visible",
    "pickupable",
    "receptacle",
    "openable",
    "isOpen",
    "toggleable",
    "isToggled",
    "sliceable",
    "isSliced",
    "breakable",
    "isBroken",
    "dirtyable",
    "isDirty",
    "canFillWithLiquid",
    "isFilledWithLiquid",
    "fillLiquid",
    "canBeUsedUp",
    "isUsedUp",
    "parentReceptacles",
    "receptacleObjectIds",
)


@dataclass(frozen=True)
class InteractionBinding:
    valid: bool
    action: str
    args: dict[str, Any]
    target_object: dict[str, Any] | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "action": self.action,
            "args": self.args,
            "target_object": self.target_object,
            "errors": self.errors,
            "warnings": self.warnings,
        }


class AI2ThorInteractionResolver:
    def build_context(
        self,
        metadata: dict[str, Any],
        *,
        max_objects: int = 40,
    ) -> dict[str, Any]:
        objects = [
            {
                key: item.get(key)
                for key in CONTEXT_OBJECT_KEYS
                if key in item
            }
            for item in metadata.get("objects", [])
        ]
        objects.sort(
            key=lambda item: (
                not bool(item.get("visible")),
                float(item.get("distance") or 1e9),
                str(item.get("objectType") or ""),
            )
        )
        return {
            "agent": metadata.get("agent", {}),
            "inventoryObjects": metadata.get("inventoryObjects", []),
            "objects": objects[:max_objects],
            "lastAction": metadata.get("lastAction"),
            "lastActionSuccess": metadata.get("lastActionSuccess"),
            "errorMessage": metadata.get("errorMessage") or "",
        }

    def resolve(
        self,
        *,
        action: str,
        args: dict[str, Any] | None,
        instruction: str,
        metadata: dict[str, Any],
    ) -> InteractionBinding:
        normalized_args = dict(args or {})
        if action not in OBJECT_ID_ACTIONS:
            return InteractionBinding(True, action, normalized_args)

        objects = list(metadata.get("objects", []))
        inventory = list(metadata.get("inventoryObjects", []))
        if action == "PutObject" and not inventory:
            return InteractionBinding(
                False,
                action,
                normalized_args,
                errors=["PutObject requires an object in inventory"],
            )

        object_id = str(normalized_args.get("objectId") or "").strip()
        if object_id:
            target = next(
                (item for item in objects if item.get("objectId") == object_id),
                None,
            )
            if target is None:
                return InteractionBinding(
                    False,
                    action,
                    normalized_args,
                    errors=[f"objectId is not present in current metadata: {object_id}"],
                )
        else:
            target = self._select_target(
                action=action,
                args=normalized_args,
                instruction=instruction,
                objects=objects,
            )
            if target is None:
                return InteractionBinding(
                    False,
                    action,
                    normalized_args,
                    errors=[
                        "no uniquely grounded object satisfies the action affordance; "
                        "observe or navigate before interacting"
                    ],
                )
            object_id = str(target["objectId"])

        affordance_error = self._affordance_error(action, target)
        if affordance_error:
            return InteractionBinding(
                False,
                action,
                normalized_args,
                target_object=self._context_object(target),
                errors=[affordance_error],
            )
        if not target.get("visible", False) and not normalized_args.get("forceAction", False):
            return InteractionBinding(
                False,
                action,
                normalized_args,
                target_object=self._context_object(target),
                errors=[
                    f"{action} target is not visible: {target.get('objectId')}"
                ],
            )

        normalized_args = {
            key: value
            for key, value in normalized_args.items()
            if key not in SEMANTIC_ARG_KEYS
        }
        normalized_args["objectId"] = object_id
        return InteractionBinding(
            True,
            action,
            normalized_args,
            target_object=self._context_object(target),
        )

    def _select_target(
        self,
        *,
        action: str,
        args: dict[str, Any],
        instruction: str,
        objects: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        target_text = " ".join(
            str(args.get(key) or "")
            for key in SEMANTIC_ARG_KEYS
        ).strip()
        search_text = self._normalize(f"{target_text} {instruction}")
        candidates = [
            item
            for item in objects
            if item.get("objectId") and not self._affordance_error(action, item)
        ]
        if not candidates:
            return None

        ranked: list[tuple[int, float, str, dict[str, Any]]] = []
        for item in candidates:
            object_type = str(item.get("objectType") or "")
            object_name = str(item.get("name") or "")
            identity = self._normalize(f"{object_type} {object_name}")
            text_match = int(
                bool(identity)
                and (
                    identity in search_text
                    or self._normalize(object_type) in search_text
                )
            )
            visible = int(bool(item.get("visible")))
            distance = float(item.get("distance") or 1e9)
            ranked.append(
                (
                    text_match * 10 + visible,
                    -distance,
                    str(item.get("objectId")),
                    item,
                )
            )
        ranked.sort(reverse=True, key=lambda entry: entry[:3])
        best = ranked[0]
        if best[0] <= 1 and len(candidates) > 1 and not target_text:
            return None
        return best[3]

    @staticmethod
    def _affordance_error(
        action: str,
        target: dict[str, Any],
    ) -> str | None:
        requirement = ACTION_REQUIREMENTS.get(action)
        if requirement is not None:
            key, expected = requirement
            if bool(target.get(key)) is not expected:
                return (
                    f"{action} requires {key}={expected} for "
                    f"{target.get('objectId')}"
                )
        state_checks = {
            "OpenObject": ("isOpen", False),
            "CloseObject": ("isOpen", True),
            "ToggleObjectOn": ("isToggled", False),
            "ToggleObjectOff": ("isToggled", True),
            "CleanObject": ("isDirty", True),
            "DirtyObject": ("isDirty", False),
            "EmptyLiquidFromObject": ("isFilledWithLiquid", True),
            "UseUpObject": ("isUsedUp", False),
        }
        state_check = state_checks.get(action)
        if state_check is not None:
            key, expected = state_check
            if bool(target.get(key)) is not expected:
                return (
                    f"{action} requires {key}={expected} for "
                    f"{target.get('objectId')}"
                )
        return None

    @staticmethod
    def _context_object(item: dict[str, Any]) -> dict[str, Any]:
        return {
            key: item.get(key)
            for key in CONTEXT_OBJECT_KEYS
            if key in item
        }

    @staticmethod
    def _normalize(value: str) -> str:
        return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", value.lower())
