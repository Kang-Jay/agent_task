from __future__ import annotations

import math
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
    "heldObject",
    "heldObjectId",
    "heldObjectType",
    "receptacle",
    "receptacleObjectId",
    "receptacleType",
}

STANDARD_TARGET_KEYS = (
    "object",
    "objectType",
    "target",
    "targetObject",
)

PUT_RECEPTACLE_KEYS = (
    "receptacle",
    "receptacleType",
    "target",
    "targetObject",
)

PUT_HELD_OBJECT_KEYS = (
    "object",
    "heldObject",
    "heldObjectId",
    "heldObjectType",
)

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
    "isPickedUp",
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
                self._distance_sort_key(item),
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
        if action == "PutObject":
            inventory_error = self._put_inventory_error(
                args=normalized_args,
                inventory=inventory,
            )
            if inventory_error:
                return InteractionBinding(
                    False,
                    action,
                    normalized_args,
                    errors=[inventory_error],
                )
        elif action == "PickupObject" and inventory:
            held_ids = sorted(
                str(item.get("objectId"))
                for item in inventory
                if item.get("objectId")
            )
            return InteractionBinding(
                False,
                action,
                normalized_args,
                errors=[
                    "PickupObject requires empty inventory; "
                    f"currently holding: {', '.join(held_ids) or 'unknown object'}"
                ],
            )

        explicit_id_error, object_id = self._explicit_target_id(
            action=action,
            args=normalized_args,
        )
        if explicit_id_error:
            return InteractionBinding(
                False,
                action,
                normalized_args,
                errors=[explicit_id_error],
            )
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
            selector_values = self._target_selector_values(
                action=action,
                args=normalized_args,
            )
            target = self._select_target(
                action=action,
                selector_values=selector_values,
                instruction=instruction,
                objects=objects,
            )
            if target is None:
                selector_detail = (
                    f" for selectors: {', '.join(selector_values)}"
                    if selector_values
                    else ""
                )
                return InteractionBinding(
                    False,
                    action,
                    normalized_args,
                    errors=[
                        "no uniquely grounded object satisfies the action affordance; "
                        f"observe or navigate before interacting{selector_detail}"
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
        interaction_error = self._interaction_state_error(
            action=action,
            target=target,
            force_action=bool(normalized_args.get("forceAction", False)),
        )
        if interaction_error:
            return InteractionBinding(
                False,
                action,
                normalized_args,
                target_object=self._context_object(target),
                errors=[interaction_error],
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
        selector_values: list[str],
        instruction: str,
        objects: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        target_text = " ".join(selector_values).strip()
        search_text = self._normalize(f"{target_text} {instruction}")
        candidates = [
            item
            for item in objects
            if item.get("objectId") and not self._requirement_error(action, item)
        ]
        if not candidates:
            return None
        if selector_values:
            candidates = [
                item
                for item in candidates
                if all(
                    self._matches_selector(item, selector)
                    for selector in selector_values
                )
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
            distance = self._distance_sort_key(item)
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
        requirement_error = AI2ThorInteractionResolver._requirement_error(
            action,
            target,
        )
        if requirement_error:
            return requirement_error
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
        if (
            action == "PutObject"
            and bool(target.get("openable"))
            and not bool(target.get("isOpen"))
        ):
            return (
                "PutObject requires an open receptacle for "
                f"{target.get('objectId')}"
            )
        return None

    @staticmethod
    def _requirement_error(
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
        return None

    def _put_inventory_error(
        self,
        *,
        args: dict[str, Any],
        inventory: list[dict[str, Any]],
    ) -> str | None:
        if not inventory:
            return "PutObject requires an object in inventory"
        if len(inventory) != 1:
            held_ids = sorted(
                str(item.get("objectId"))
                for item in inventory
                if item.get("objectId")
            )
            return (
                "PutObject requires exactly one unambiguous held object; "
                f"inventory contains: {', '.join(held_ids) or 'unknown objects'}"
            )

        held_object = inventory[0]
        held_selectors = [
            str(args.get(key) or "").strip()
            for key in PUT_HELD_OBJECT_KEYS
            if str(args.get(key) or "").strip()
        ]
        has_explicit_receptacle = any(
            str(args.get(key) or "").strip()
            for key in PUT_RECEPTACLE_KEYS
        ) or bool(str(args.get("receptacleObjectId") or "").strip())
        object_type = str(args.get("objectType") or "").strip()
        if has_explicit_receptacle and object_type:
            receptacle_selectors = [
                str(args.get(key) or "").strip()
                for key in PUT_RECEPTACLE_KEYS
                if str(args.get(key) or "").strip()
            ]
            if self._matches_selector(held_object, object_type):
                held_selectors.append(object_type)
            elif not any(
                self._selectors_overlap(object_type, selector)
                for selector in receptacle_selectors
            ):
                held_id = str(held_object.get("objectId") or "unknown")
                held_type = str(held_object.get("objectType") or "unknown")
                return (
                    "PutObject objectType matches neither the held object nor "
                    "the requested receptacle: "
                    f"selector={object_type}, held={held_type} ({held_id})"
                )

        for selector in held_selectors:
            if not self._matches_selector(held_object, selector):
                held_id = str(held_object.get("objectId") or "unknown")
                held_type = str(held_object.get("objectType") or "unknown")
                return (
                    "PutObject requested held object does not match inventory: "
                    f"selector={selector}, held={held_type} ({held_id})"
                )
        return None

    @staticmethod
    def _explicit_target_id(
        *,
        action: str,
        args: dict[str, Any],
    ) -> tuple[str | None, str]:
        object_id = str(args.get("objectId") or "").strip()
        if action != "PutObject":
            return None, object_id

        receptacle_id = str(args.get("receptacleObjectId") or "").strip()
        if object_id and receptacle_id and object_id != receptacle_id:
            return (
                "PutObject received conflicting receptacle identifiers: "
                f"objectId={object_id}, receptacleObjectId={receptacle_id}",
                "",
            )
        return None, object_id or receptacle_id

    @staticmethod
    def _target_selector_values(
        *,
        action: str,
        args: dict[str, Any],
    ) -> list[str]:
        if action != "PutObject":
            keys = STANDARD_TARGET_KEYS
        else:
            keys = PUT_RECEPTACLE_KEYS
            if not any(str(args.get(key) or "").strip() for key in keys):
                keys = ("objectType",)
        values: list[str] = []
        for key in keys:
            value = str(args.get(key) or "").strip()
            if value and value not in values:
                values.append(value)
        return values

    @classmethod
    def _matches_selector(
        cls,
        item: dict[str, Any],
        selector: str,
    ) -> bool:
        normalized_selector = cls._normalize(selector)
        if not normalized_selector:
            return False
        identities = (
            str(item.get("objectId") or ""),
            str(item.get("objectType") or ""),
            str(item.get("name") or ""),
        )
        return any(
            normalized_selector == cls._normalize(identity)
            for identity in identities
            if identity
        )

    @classmethod
    def _selectors_overlap(cls, left: str, right: str) -> bool:
        normalized_left = cls._normalize(left)
        normalized_right = cls._normalize(right)
        return bool(
            normalized_left
            and normalized_right
            and (
                normalized_left == normalized_right
                or normalized_left in normalized_right
                or normalized_right in normalized_left
            )
        )

    @staticmethod
    def _interaction_state_error(
        *,
        action: str,
        target: dict[str, Any],
        force_action: bool,
    ) -> str | None:
        if force_action:
            return None
        object_id = target.get("objectId")
        if not target.get("visible", False):
            return f"{action} target is not visible: {object_id}"
        distance = target.get("distance")
        try:
            numeric_distance = float(distance)
        except (TypeError, ValueError):
            return (
                f"{action} requires finite distance metadata for {object_id}; "
                f"received {distance!r}"
            )
        if not math.isfinite(numeric_distance) or numeric_distance < 0:
            return (
                f"{action} requires finite non-negative distance metadata for "
                f"{object_id}; received {distance!r}"
            )
        return None

    @staticmethod
    def _distance_sort_key(item: dict[str, Any]) -> float:
        try:
            distance = float(item.get("distance"))
        except (TypeError, ValueError):
            return math.inf
        if not math.isfinite(distance) or distance < 0:
            return math.inf
        return distance

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
