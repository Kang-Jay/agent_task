from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from src.simulation.ai2thor_actions import AI2ThorActionExecutor
from src.simulation.ai2thor_postconditions import (
    AI2ThorPostconditionVerifier,
)


@dataclass(frozen=True)
class ApproachVerification:
    verified: bool
    objectId: str
    source: str
    reason: str
    candidate_count: int
    matched_pose: dict[str, Any] | None = None

    def to_context(self) -> dict[str, Any]:
        return asdict(self)


class AI2ThorApproachVerifier:
    SOURCE = "ai2thor_interactable_pose"

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

        matched_pose = next(
            (
                pose
                for pose in poses
                if isinstance(pose, dict)
                and self._matches_agent_pose(agent, pose)
            ),
            None,
        )
        if matched_pose is None:
            return ApproachVerification(
                verified=False,
                objectId=object_id,
                source=self.SOURCE,
                reason=(
                    "current agent pose is not an AI2-THOR "
                    "interactable pose for the target"
                ),
                candidate_count=len(poses),
            )
        return ApproachVerification(
            verified=True,
            objectId=object_id,
            source=self.SOURCE,
            reason="current pose matches an AI2-THOR interactable pose",
            candidate_count=len(poses),
            matched_pose=dict(matched_pose),
        )

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
