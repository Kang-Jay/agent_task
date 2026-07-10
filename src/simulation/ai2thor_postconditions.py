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
        if action in {"MoveAhead", "MoveBack", "MoveLeft", "MoveRight"}:
            return self._verify_planar_move(
                action=action,
                args=args,
                before=before,
                after=after,
            )
        if action in {"MoveRelative", "FlyAhead", "FlyBack", "FlyLeft", "FlyRight", "FlyUp", "FlyDown", "FlyTo"}:
            before_position = self._agent_position(before)
            after_position = self._agent_position(after)
            changed = self._distance(before_position, after_position) > self.POSITION_EPSILON
            return self._result(
                action,
                changed,
                "agent position changed" if changed else "agent position did not change",
                {"before": before_position, "after": after_position},
            )
        if action in {"RotateLeft", "RotateRight"}:
            return self._verify_cardinal_rotation(
                action=action,
                args=args,
                before=before,
                after=after,
            )
        if action in {"RotateAgent", "Rotate"}:
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
            before_horizon = self._finite_agent_horizon(before)
            after_horizon = self._finite_agent_horizon(after)
            if before_horizon is None or after_horizon is None:
                return self._result(
                    action,
                    False,
                    "agent cameraHorizon metadata is unavailable",
                    {
                        "before_horizon": before_horizon,
                        "after_horizon": after_horizon,
                    },
                )
            degrees = self._positive_finite_argument(args, "degrees")
            if degrees is None:
                if "degrees" not in args:
                    changed = (
                        abs(after_horizon - before_horizon)
                        > self.ANGLE_EPSILON
                    )
                    return self._result(
                        action,
                        changed,
                        (
                            "camera horizon changed using controller default look angle"
                            if changed
                            else "camera horizon did not change using controller default look angle"
                        ),
                        {
                            "before_horizon": before_horizon,
                            "after_horizon": after_horizon,
                            "used_controller_default": True,
                        },
                    )
                return self._result(
                    action,
                    False,
                    "look action requires a positive finite degrees argument",
                    {"args": dict(args)},
                )
            expected_horizon = (
                before_horizon + degrees
                if action == "LookDown"
                else before_horizon - degrees
            )
            matched = (
                abs(after_horizon - expected_horizon)
                <= self.ANGLE_EPSILON
            )
            return self._result(
                action,
                matched,
                (
                    "camera horizon matched requested look"
                    if matched
                    else "camera horizon did not match requested look"
                ),
                {
                    "before_horizon": before_horizon,
                    "after_horizon": after_horizon,
                    "expected_horizon": expected_horizon,
                    "degrees": degrees,
                },
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

    def _verify_planar_move(
        self,
        *,
        action: str,
        args: dict[str, Any],
        before: dict[str, Any],
        after: dict[str, Any],
    ) -> PostconditionResult:
        magnitude = self._positive_finite_argument(
            args,
            "moveMagnitude",
        )
        if magnitude is None:
            if "moveMagnitude" not in args:
                before_position = self._finite_agent_position(before)
                after_position = self._finite_agent_position(after)
                if (
                    before_position is None
                    or after_position is None
                ):
                    return self._result(
                        action,
                        False,
                        "agent pose metadata is unavailable for default move verification",
                        {
                            "before": before_position,
                            "after": after_position,
                        },
                    )
                changed = (
                    self._distance(
                        before_position,
                        after_position,
                    )
                    > self.POSITION_EPSILON
                )
                return self._result(
                    action,
                    changed,
                    (
                        "agent position changed using controller default move magnitude"
                        if changed
                        else "agent position did not change using controller default move magnitude"
                    ),
                    {
                        "before": before_position,
                        "after": after_position,
                        "used_controller_default": True,
                    },
                )
            return self._result(
                action,
                False,
                "move action requires a positive finite moveMagnitude",
                {"args": dict(args)},
            )
        before_position = self._finite_agent_position(before)
        after_position = self._finite_agent_position(after)
        before_yaw = self._finite_agent_yaw(before)
        if (
            before_position is None
            or after_position is None
            or before_yaw is None
        ):
            return self._result(
                action,
                False,
                "agent pose metadata is unavailable for move verification",
                {
                    "before": before_position,
                    "after": after_position,
                },
            )
        delta_x = after_position["x"] - before_position["x"]
        delta_z = after_position["z"] - before_position["z"]
        yaw = math.radians(before_yaw)
        forward_x = math.sin(yaw)
        forward_z = math.cos(yaw)
        right_x = math.cos(yaw)
        right_z = -math.sin(yaw)
        direction_by_action = {
            "MoveAhead": (forward_x, forward_z),
            "MoveBack": (-forward_x, -forward_z),
            "MoveLeft": (-right_x, -right_z),
            "MoveRight": (right_x, right_z),
        }
        expected_x, expected_z = direction_by_action[action]
        forward_progress = (
            delta_x * expected_x + delta_z * expected_z
        )
        lateral_error = abs(
            delta_x * expected_z - delta_z * expected_x
        )
        actual_distance = math.hypot(delta_x, delta_z)
        direction_matched = (
            forward_progress > self.POSITION_EPSILON
            and lateral_error <= self.POSITION_EPSILON
        )
        distance_matched = (
            abs(actual_distance - magnitude)
            <= self.POSITION_EPSILON
        )
        passed = direction_matched and distance_matched
        return self._result(
            action,
            passed,
            (
                "agent movement matched requested direction and distance"
                if passed
                else "agent movement did not match requested direction and distance"
            ),
            {
                "before": before_position,
                "after": after_position,
                "requested_distance": magnitude,
                "actual_distance": actual_distance,
                "forward_progress": forward_progress,
                "lateral_error": lateral_error,
            },
        )

    def _verify_cardinal_rotation(
        self,
        *,
        action: str,
        args: dict[str, Any],
        before: dict[str, Any],
        after: dict[str, Any],
    ) -> PostconditionResult:
        degrees = self._positive_finite_argument(args, "degrees")
        if degrees is None:
            if "degrees" not in args:
                before_yaw = self._finite_agent_yaw(before)
                after_yaw = self._finite_agent_yaw(after)
                if before_yaw is None or after_yaw is None:
                    return self._result(
                        action,
                        False,
                        "agent yaw metadata is unavailable for default rotation verification",
                        {
                            "before_yaw": before_yaw,
                            "after_yaw": after_yaw,
                        },
                    )
                changed = (
                    self._angle_delta(before_yaw, after_yaw)
                    > self.ANGLE_EPSILON
                )
                return self._result(
                    action,
                    changed,
                    (
                        "agent yaw changed using controller default rotation"
                        if changed
                        else "agent yaw did not change using controller default rotation"
                    ),
                    {
                        "before_yaw": before_yaw,
                        "after_yaw": after_yaw,
                        "used_controller_default": True,
                    },
                )
            return self._result(
                action,
                False,
                "rotation requires a positive finite degrees argument",
                {"args": dict(args)},
            )
        before_yaw = self._finite_agent_yaw(before)
        after_yaw = self._finite_agent_yaw(after)
        if before_yaw is None or after_yaw is None:
            return self._result(
                action,
                False,
                "agent yaw metadata is unavailable",
                {
                    "before_yaw": before_yaw,
                    "after_yaw": after_yaw,
                },
            )
        expected_yaw = (
            before_yaw + degrees
            if action == "RotateRight"
            else before_yaw - degrees
        ) % 360.0
        error = self._angle_delta(after_yaw, expected_yaw)
        passed = error <= self.ANGLE_EPSILON
        return self._result(
            action,
            passed,
            (
                "agent yaw matched requested rotation"
                if passed
                else "agent yaw did not match requested rotation"
            ),
            {
                "before_yaw": before_yaw,
                "after_yaw": after_yaw,
                "expected_yaw": expected_yaw,
                "degrees": degrees,
                "angle_error": error,
            },
        )

    @staticmethod
    def _positive_finite_argument(
        args: dict[str, Any],
        name: str,
    ) -> float | None:
        value = args.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
        ):
            return None
        number = float(value)
        if not math.isfinite(number) or number <= 0.0:
            return None
        return number

    @staticmethod
    def _finite_agent_position(
        metadata: dict[str, Any],
    ) -> dict[str, float] | None:
        position = (metadata.get("agent") or {}).get("position")
        if not isinstance(position, dict):
            return None
        try:
            result = {
                axis: float(position[axis])
                for axis in ("x", "y", "z")
            }
        except (KeyError, TypeError, ValueError):
            return None
        if not all(math.isfinite(value) for value in result.values()):
            return None
        return result

    @staticmethod
    def _finite_agent_yaw(
        metadata: dict[str, Any],
    ) -> float | None:
        rotation = (metadata.get("agent") or {}).get("rotation")
        if not isinstance(rotation, dict):
            return None
        try:
            yaw = float(rotation["y"])
        except (KeyError, TypeError, ValueError):
            return None
        return yaw if math.isfinite(yaw) else None

    @staticmethod
    def _finite_agent_horizon(
        metadata: dict[str, Any],
    ) -> float | None:
        agent = metadata.get("agent")
        if not isinstance(agent, dict):
            return None
        try:
            horizon = float(agent["cameraHorizon"])
        except (KeyError, TypeError, ValueError):
            return None
        return horizon if math.isfinite(horizon) else None

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
