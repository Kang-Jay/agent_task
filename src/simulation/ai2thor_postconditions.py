from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class PostconditionResult:
    checked: bool
    passed: bool
    action: str
    reason: str
    evidence: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AI2ThorPostconditionVerifier:
    POSITION_EPSILON = 1e-4
    ANGLE_EPSILON = 1e-3

    def verify(
        self,
        *,
        action: str,
        args: dict[str, Any],
        before: dict[str, Any],
        after: dict[str, Any],
        runtime_success: bool,
    ) -> PostconditionResult:
        if not runtime_success:
            return PostconditionResult(
                checked=True,
                passed=False,
                action=action,
                reason="AI2-THOR reported lastActionSuccess=false",
                evidence={"errorMessage": after.get("errorMessage")},
            )

        if action in {"Pass", "Done"}:
            return self._result(action, True, "control action completed", {})
        if action in {"MoveAhead", "MoveBack", "MoveLeft", "MoveRight", "MoveRelative", "FlyAhead", "FlyBack", "FlyLeft", "FlyRight", "FlyUp", "FlyDown", "FlyTo"}:
            before_position = self._agent_position(before)
            after_position = self._agent_position(after)
            changed = self._distance(before_position, after_position) > self.POSITION_EPSILON
            return self._result(
                action,
                changed,
                "agent position changed" if changed else "agent position did not change",
                {"before": before_position, "after": after_position},
            )
        if action in {"RotateLeft", "RotateRight", "RotateAgent", "Rotate"}:
            before_yaw = self._agent_rotation(before).get("y", 0.0)
            after_yaw = self._agent_rotation(after).get("y", 0.0)
            changed = self._angle_delta(before_yaw, after_yaw) > self.ANGLE_EPSILON
            return self._result(
                action,
                changed,
                "agent yaw changed" if changed else "agent yaw did not change",
                {"before_yaw": before_yaw, "after_yaw": after_yaw},
            )
        if action in {"LookUp", "LookDown"}:
            before_horizon = float((before.get("agent") or {}).get("cameraHorizon", 0.0))
            after_horizon = float((after.get("agent") or {}).get("cameraHorizon", 0.0))
            changed = abs(before_horizon - after_horizon) > self.ANGLE_EPSILON
            return self._result(
                action,
                changed,
                "camera horizon changed" if changed else "camera horizon did not change",
                {"before_horizon": before_horizon, "after_horizon": after_horizon},
            )
        if action in {"Crouch", "Stand"}:
            expected = action == "Stand"
            actual = (after.get("agent") or {}).get("isStanding")
            if not isinstance(actual, bool):
                return self._result(
                    action,
                    False,
                    "agent isStanding is unavailable",
                    {
                        "expected_isStanding": expected,
                        "actual_isStanding": actual,
                    },
                )
            return self._result(
                action,
                actual == expected,
                f"agent isStanding={actual}",
                {"expected_isStanding": expected, "actual_isStanding": actual},
            )

        object_id = self._object_id(args)
        if action == "PickupObject":
            inventory = self._inventory_ids(after)
            passed = object_id in inventory if object_id else len(inventory) > len(self._inventory_ids(before))
            return self._result(
                action,
                passed,
                "target object entered inventory" if passed else "target object is not in inventory",
                {"objectId": object_id, "inventoryObjectIds": sorted(inventory)},
            )
        if action == "PutObject":
            before_inventory = self._inventory_ids(before)
            after_inventory = self._inventory_ids(after)
            released_ids = before_inventory - after_inventory
            receptacle_id = object_id
            target_receptacle = self._object(after, receptacle_id)
            receptacle_object_ids = {
                str(value)
                for value in (
                    (target_receptacle or {}).get("receptacleObjectIds") or []
                )
            }
            placed_ids = {
                released_id
                for released_id in released_ids
                if receptacle_id
                in (
                    (self._object(after, released_id) or {}).get(
                        "parentReceptacles"
                    )
                    or []
                )
                and released_id in receptacle_object_ids
            }
            passed = bool(released_ids) and placed_ids == released_ids
            return self._result(
                action,
                passed,
                (
                    "released object entered the requested receptacle"
                    if passed
                    else "released object is not registered in the requested receptacle"
                ),
                {
                    "receptacleObjectId": receptacle_id,
                    "beforeInventoryObjectIds": sorted(before_inventory),
                    "afterInventoryObjectIds": sorted(after_inventory),
                    "releasedObjectIds": sorted(released_ids),
                    "placedObjectIds": sorted(placed_ids),
                    "receptacleObjectIds": sorted(receptacle_object_ids),
                },
            )
        if action in {"DropHandObject", "ThrowObject", "ReleaseObject"}:
            before_inventory = self._inventory_ids(before)
            after_inventory = self._inventory_ids(after)
            passed = len(after_inventory) < len(before_inventory)
            return self._result(
                action,
                passed,
                (
                    "held object left inventory"
                    if passed
                    else "inventory did not release an object"
                ),
                {
                    "beforeInventoryObjectIds": sorted(before_inventory),
                    "afterInventoryObjectIds": sorted(after_inventory),
                },
            )

        object_expectations = {
            "OpenObject": ("isOpen", True),
            "CloseObject": ("isOpen", False),
            "ToggleObjectOn": ("isToggled", True),
            "ToggleObjectOff": ("isToggled", False),
            "DirtyObject": ("isDirty", True),
            "CleanObject": ("isDirty", False),
            "FillObjectWithLiquid": ("isFilledWithLiquid", True),
            "EmptyLiquidFromObject": ("isFilledWithLiquid", False),
            "SliceObject": ("isSliced", True),
            "BreakObject": ("isBroken", True),
            "UseUpObject": ("isUsedUp", True),
        }
        if action in object_expectations:
            field, expected = object_expectations[action]
            target = self._object(after, object_id)
            if target is None:
                return self._result(
                    action,
                    False,
                    "target object metadata is unavailable after execution",
                    {"objectId": object_id, "field": field, "expected": expected},
                )
            actual = target.get(field)
            return self._result(
                action,
                actual == expected,
                f"{field}={actual}",
                {
                    "objectId": object_id,
                    "field": field,
                    "expected": expected,
                    "actual": actual,
                },
            )

        return PostconditionResult(
            checked=False,
            passed=True,
            action=action,
            reason="no semantic postcondition verifier is registered for this action",
            evidence={},
        )

    def _result(
        self,
        action: str,
        passed: bool,
        reason: str,
        evidence: dict[str, Any],
    ) -> PostconditionResult:
        return PostconditionResult(
            checked=True,
            passed=passed,
            action=action,
            reason=reason,
            evidence=evidence,
        )

    @staticmethod
    def _object_id(args: dict[str, Any]) -> str | None:
        value = args.get("objectId")
        if value is None:
            candidates = args.get("objectIdCandidates")
            if isinstance(candidates, list) and candidates:
                value = candidates[0]
        return str(value) if value else None

    @staticmethod
    def _object(metadata: dict[str, Any], object_id: str | None) -> dict[str, Any] | None:
        if object_id is None:
            return None
        return next(
            (
                item
                for item in metadata.get("objects", [])
                if str(item.get("objectId")) == object_id
            ),
            None,
        )

    @staticmethod
    def _inventory_ids(metadata: dict[str, Any]) -> set[str]:
        return {
            str(item.get("objectId"))
            for item in metadata.get("inventoryObjects", [])
            if item.get("objectId")
        }

    @staticmethod
    def _agent_position(metadata: dict[str, Any]) -> dict[str, float]:
        position = (metadata.get("agent") or {}).get("position") or {}
        return {
            "x": float(position.get("x", 0.0)),
            "y": float(position.get("y", 0.0)),
            "z": float(position.get("z", 0.0)),
        }

    @staticmethod
    def _agent_rotation(metadata: dict[str, Any]) -> dict[str, float]:
        rotation = (metadata.get("agent") or {}).get("rotation") or {}
        return {
            "x": float(rotation.get("x", 0.0)),
            "y": float(rotation.get("y", 0.0)),
            "z": float(rotation.get("z", 0.0)),
        }

    @staticmethod
    def _distance(first: dict[str, float], second: dict[str, float]) -> float:
        return math.sqrt(
            sum((first[axis] - second[axis]) ** 2 for axis in ("x", "y", "z"))
        )

    @staticmethod
    def _angle_delta(first: float, second: float) -> float:
        return abs((second - first + 180.0) % 360.0 - 180.0)
