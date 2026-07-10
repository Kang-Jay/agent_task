from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from src.simulation.ai2thor_actions import AI2ThorActionExecutor
from src.simulation.ai2thor_postconditions import (
    AI2ThorPostconditionVerifier,
)
from src.simulation.ai2thor_runtime import DEFAULT_GRID_SIZE_METERS


@dataclass(frozen=True)
class ApproachVerification:
    verified: bool
    objectId: str
    source: str
    reason: str
    candidate_count: int
    matched_pose: dict[str, Any] | None = None
    target_pose: dict[str, Any] | None = None
    path_status: str | None = None
    recommended_action: dict[str, Any] | None = None

    def to_context(self) -> dict[str, Any]:
        return asdict(self)


class AI2ThorApproachVerifier:
    SOURCE = "ai2thor_interactable_pose"
    PATH_COMPLETE = "PathComplete"

    def __init__(
        self,
        executor: AI2ThorActionExecutor | None = None,
    ) -> None:
        self.executor = executor or AI2ThorActionExecutor()

    def verify(
        self,
        controller: Any,
        *,
        mode: str,
        metadata: dict[str, Any],
        object_id: str,
    ) -> ApproachVerification:
        agent = metadata.get("agent") or {}
        standing = agent.get("isStanding")
        if not isinstance(standing, bool):
            return self._failure(
                object_id,
                "agent.isStanding is unavailable",
            )

        try:
            execution = self.executor.execute(
                controller,
                mode=mode,
                action="GetInteractablePoses",
                args={
                    "objectId": object_id,
                    "standings": [standing],
                    "maxPoses": 64,
                },
                actor="manual",
            )
        except (RuntimeError, ValueError) as exc:
            return self._failure(object_id, str(exc))

        poses = execution.action_return
        if not execution.success:
            return self._failure(
                object_id,
                execution.error_message
                or "GetInteractablePoses failed",
            )
        if not isinstance(poses, list) or not poses:
            return self._failure(
                object_id,
                "GetInteractablePoses returned no candidates",
            )

        valid_poses = [
            dict(pose)
            for pose in poses
            if isinstance(pose, dict) and self._valid_pose(pose)
        ]
        if not valid_poses:
            return self._failure(
                object_id,
                "GetInteractablePoses returned no valid candidates",
            )

        matched_pose = next(
            (
                pose
                for pose in valid_poses
                if self._matches_agent_pose(agent, pose)
            ),
            None,
        )
        if matched_pose is None:
            target_pose, path_status, recommended_action = (
                self._select_navigation_guidance(
                    controller,
                    mode=mode,
                    agent=agent,
                    poses=valid_poses,
                )
            )
            return ApproachVerification(
                verified=False,
                objectId=object_id,
                source=self.SOURCE,
                reason=(
                    "current agent pose is not an AI2-THOR "
                    "interactable pose for the target"
                ),
                candidate_count=len(valid_poses),
                target_pose=target_pose,
                path_status=path_status,
                recommended_action=recommended_action,
            )
        return ApproachVerification(
            verified=True,
            objectId=object_id,
            source=self.SOURCE,
            reason="current pose matches an AI2-THOR interactable pose",
            candidate_count=len(valid_poses),
            matched_pose=dict(matched_pose),
            target_pose=dict(matched_pose),
            path_status=self.PATH_COMPLETE,
        )

    def _select_navigation_guidance(
        self,
        controller: Any,
        *,
        mode: str,
        agent: dict[str, Any],
        poses: list[dict[str, Any]],
    ) -> tuple[
        dict[str, Any] | None,
        str | None,
        dict[str, Any] | None,
    ]:
        ordered_poses = sorted(
            enumerate(poses),
            key=lambda item: (
                self._distance_to_pose(agent, item[1]),
                item[0],
            ),
        )
        last_status: str | None = None
        for _, target_pose in ordered_poses:
            path_status, recommended_action = self._navigation_guidance(
                controller,
                mode=mode,
                agent=agent,
                target_pose=target_pose,
            )
            last_status = path_status or last_status
            if recommended_action is not None:
                return (
                    dict(target_pose),
                    path_status,
                    recommended_action,
                )
        return None, last_status, None

    def _failure(
        self,
        object_id: str,
        reason: str,
    ) -> ApproachVerification:
        return ApproachVerification(
            verified=False,
            objectId=object_id,
            source=self.SOURCE,
            reason=reason,
            candidate_count=0,
        )

    def _navigation_guidance(
        self,
        controller: Any,
        *,
        mode: str,
        agent: dict[str, Any],
        target_pose: dict[str, Any],
    ) -> tuple[str | None, dict[str, Any] | None]:
        direct_action = self._action_toward_pose(
            agent,
            target_pose,
            path_corners=[],
        )
        if direct_action is not None and self._same_position(
            agent,
            target_pose,
        ):
            return "PoseAlignment", direct_action

        try:
            execution = self.executor.execute(
                controller,
                mode=mode,
                action="GetShortestPathToPoint",
                args={
                    "target": {
                        "x": float(target_pose["x"]),
                        "y": float(target_pose["y"]),
                        "z": float(target_pose["z"]),
                    }
                },
                actor="manual",
            )
        except (KeyError, RuntimeError, TypeError, ValueError):
            return None, None
        payload = execution.action_return
        if not execution.success or not isinstance(payload, dict):
            return None, None
        status = str(payload.get("status") or "")
        if status != self.PATH_COMPLETE:
            return status or None, None
        corners = payload.get("corners") or []
        if not self._valid_path_corners(corners):
            return status, None
        return (
            status,
            self._action_toward_pose(
                agent,
                target_pose,
                path_corners=corners,
            ),
        )

    @classmethod
    def _action_toward_pose(
        cls,
        agent: dict[str, Any],
        target_pose: dict[str, Any],
        *,
        path_corners: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        position = agent.get("position") or {}
        rotation = agent.get("rotation") or {}
        try:
            current_x = float(position["x"])
            current_z = float(position["z"])
            current_yaw = float(rotation["y"])
        except (KeyError, TypeError, ValueError):
            return None

        if cls._same_position(agent, target_pose):
            try:
                target_yaw = float(target_pose["rotation"])
                target_horizon = float(target_pose["horizon"])
                current_horizon = float(agent["cameraHorizon"])
            except (KeyError, TypeError, ValueError):
                return None
            yaw_delta = cls._signed_angle_delta(
                current_yaw,
                target_yaw,
            )
            if (
                abs(yaw_delta)
                > AI2ThorPostconditionVerifier.ANGLE_EPSILON
            ):
                return {
                    "type": (
                        "TURN_RIGHT"
                        if yaw_delta > 0.0
                        else "TURN_LEFT"
                    ),
                    "args": {"angle": abs(yaw_delta)},
                }
            horizon_delta = target_horizon - current_horizon
            if (
                abs(horizon_delta)
                > AI2ThorPostconditionVerifier.ANGLE_EPSILON
            ):
                return {
                    "type": (
                        "LOOK_DOWN"
                        if horizon_delta > 0.0
                        else "LOOK_UP"
                    ),
                    "args": {"angle": abs(horizon_delta)},
                }
            return None

        next_corner = next(
            (
                corner
                for corner in path_corners
                if cls._planar_distance(
                    current_x,
                    current_z,
                    float(corner["x"]),
                    float(corner["z"]),
                )
                > AI2ThorPostconditionVerifier.POSITION_EPSILON
            ),
            None,
        )
        if next_corner is None:
            return None
        next_x = float(next_corner["x"])
        next_z = float(next_corner["z"])
        desired_yaw = math.degrees(
            math.atan2(next_x - current_x, next_z - current_z)
        ) % 360.0
        yaw_delta = cls._signed_angle_delta(
            current_yaw,
            desired_yaw,
        )
        if (
            abs(yaw_delta)
            > AI2ThorPostconditionVerifier.ANGLE_EPSILON
        ):
            return {
                "type": (
                    "TURN_RIGHT"
                    if yaw_delta > 0.0
                    else "TURN_LEFT"
                ),
                "args": {"angle": abs(yaw_delta)},
            }
        remaining_distance = cls._planar_distance(
            current_x,
            current_z,
            next_x,
            next_z,
        )
        return {
            "type": "MOVE_FORWARD",
            "args": {
                "distance": min(
                    remaining_distance,
                    DEFAULT_GRID_SIZE_METERS,
                )
            },
        }

    @classmethod
    def _valid_pose(cls, pose: dict[str, Any]) -> bool:
        required = ("x", "y", "z", "rotation", "horizon")
        try:
            return (
                all(math.isfinite(float(pose[key])) for key in required)
                and isinstance(pose.get("standing"), bool)
            )
        except (KeyError, TypeError, ValueError):
            return False

    @classmethod
    def _valid_path_corners(cls, corners: Any) -> bool:
        if not isinstance(corners, list) or not corners:
            return False
        try:
            return all(
                isinstance(corner, dict)
                and math.isfinite(float(corner["x"]))
                and math.isfinite(float(corner["z"]))
                for corner in corners
            )
        except (KeyError, TypeError, ValueError):
            return False

    @classmethod
    def _distance_to_pose(
        cls,
        agent: dict[str, Any],
        pose: dict[str, Any],
    ) -> float:
        position = agent.get("position") or {}
        try:
            distance = cls._planar_distance(
                float(position["x"]),
                float(position["z"]),
                float(pose["x"]),
                float(pose["z"]),
            )
        except (KeyError, TypeError, ValueError):
            return math.inf
        return distance if math.isfinite(distance) else math.inf

    @classmethod
    def _same_position(
        cls,
        agent: dict[str, Any],
        pose: dict[str, Any],
    ) -> bool:
        position = agent.get("position") or {}
        try:
            return (
                math.sqrt(
                    (
                        float(position["x"])
                        - float(pose["x"])
                    )
                    ** 2
                    + (
                        float(position["y"])
                        - float(pose["y"])
                    )
                    ** 2
                    + (
                        float(position["z"])
                        - float(pose["z"])
                    )
                    ** 2
                )
                <= AI2ThorPostconditionVerifier.POSITION_EPSILON
            )
        except (KeyError, TypeError, ValueError):
            return False

    @staticmethod
    def _planar_distance(
        left_x: float,
        left_z: float,
        right_x: float,
        right_z: float,
    ) -> float:
        return math.hypot(right_x - left_x, right_z - left_z)

    @staticmethod
    def _signed_angle_delta(current: float, target: float) -> float:
        return (target - current + 180.0) % 360.0 - 180.0

    @classmethod
    def _matches_agent_pose(
        cls,
        agent: dict[str, Any],
        pose: dict[str, Any],
    ) -> bool:
        position = agent.get("position") or {}
        rotation = agent.get("rotation") or {}
        try:
            position_delta = math.sqrt(
                (float(position["x"]) - float(pose["x"])) ** 2
                + (float(position["y"]) - float(pose["y"])) ** 2
                + (float(position["z"]) - float(pose["z"])) ** 2
            )
            yaw_delta = cls._angle_delta(
                float(rotation["y"]),
                float(pose["rotation"]),
            )
            horizon_delta = abs(
                float(agent["cameraHorizon"])
                - float(pose["horizon"])
            )
        except (KeyError, TypeError, ValueError):
            return False
        return (
            position_delta
            <= AI2ThorPostconditionVerifier.POSITION_EPSILON
            and yaw_delta
            <= AI2ThorPostconditionVerifier.ANGLE_EPSILON
            and horizon_delta
            <= AI2ThorPostconditionVerifier.ANGLE_EPSILON
            and agent.get("isStanding") is pose.get("standing")
        )

    @staticmethod
    def _angle_delta(left: float, right: float) -> float:
        return abs((left - right + 180.0) % 360.0 - 180.0)
