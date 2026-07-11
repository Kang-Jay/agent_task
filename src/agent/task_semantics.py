from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from src.simulation.ai2thor_actions import AI2ThorActionCatalog


@dataclass(frozen=True)
class TaskPlan:
    instruction: str
    mode: str
    task_types: tuple[str, ...]
    required_actions: tuple[str, ...]
    unsupported_capabilities: tuple[str, ...]
    limitations: tuple[str, ...]
    completion_mode: str
    subgoals: tuple[dict[str, Any], ...]
    completion_rule: str
    clarification: str | None
    action_candidates: tuple[str, ...]
    action_specs: tuple[dict[str, Any], ...]

    @property
    def supported(self) -> bool:
        return not self.unsupported_capabilities

    @property
    def is_visual_search(self) -> bool:
        return self.supported and self.task_types == ("visual_search",)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["supported"] = self.supported
        payload["is_visual_search"] = self.is_visual_search
        return payload

    def completion_status(
        self,
        *,
        steps: list[dict[str, Any]],
        target_visible: bool,
        confidence: float,
        stop_confidence_threshold: float,
        environment_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        successful_actions = {
            str((step.get("executed_action") or step.get("action") or {}).get("type"))
            for step in steps
            if step.get("action_success") is True
        }
        missing_actions = [
            action for action in self.required_actions if action not in successful_actions
        ]
        context = environment_context or {}
        exit_evidence = context.get("exit") or context.get("door_crossing") or {}
        exit_verified = bool(
            exit_evidence.get("crossed_threshold")
            or exit_evidence.get("crossed")
            or exit_evidence.get("passed")
        )
        agent_state = context.get("agent") or {}
        matching_targets = [
            item
            for item in context.get("objects", [])
            if self._matches_instruction_target(item)
        ]
        visible_targets = [
            item for item in matching_targets if bool(item.get("visible"))
        ]
        target_visible_in_environment = bool(visible_targets)
        target_distances: list[float] = []
        for item in visible_targets:
            try:
                distance = float(item["distance"])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(distance) and distance >= 0.0:
                target_distances.append(distance)
        target_distance = min(target_distances, default=None)
        target_object_ids = self.matching_target_object_ids(context)
        approach_evidence = context.get("approach") or {}
        approach_object_id = str(approach_evidence.get("objectId") or "")
        approach_verified = (
            bool(approach_evidence.get("verified"))
            and bool(approach_object_id)
            and approach_object_id in target_object_ids
        )
        approach_source = str(
            approach_evidence.get("source")
            or "no verified approach evidence"
        )
        agent_is_standing = agent_state.get("isStanding")
        subgoal_progress: list[dict[str, Any]] = []

        if not self.supported:
            complete = False
            reason = "task requests unsupported embodied capabilities"
        elif self.completion_mode == "approximate_sit":
            located = (
                target_visible
                or target_visible_in_environment
                or approach_verified
            )
            approached = approach_verified
            crouch_succeeded = "Crouch" in successful_actions
            posture_verified = crouch_succeeded and agent_is_standing is False
            complete = approached and posture_verified
            if complete:
                reason = (
                    "verified crouch-near-target approximation completed; "
                    "AI2-THOR does not provide a native sit-on-furniture state"
                )
            elif not located:
                reason = "target furniture has not been located"
            elif not approached:
                reason = "target furniture is visible but approach proximity is not verified"
            elif not crouch_succeeded:
                reason = "target is approached but Crouch has not executed successfully"
            else:
                reason = "Crouch executed but the simulator posture change is not verified"
            subgoal_progress = [
                {
                    "id": "locate_target",
                    "complete": located,
                    "evidence": "visual or simulator target observation",
                },
                {
                    "id": "approach_target",
                    "complete": approached,
                    "evidence": (
                        f"{approach_source}; objectId={approach_object_id}"
                        if approached
                        else "AI2-THOR did not verify a target-aligned interaction pose"
                    ),
                },
                {
                    "id": "execute_crouch",
                    "complete": crouch_succeeded,
                    "evidence": "successful Crouch execution",
                },
                {
                    "id": "verify_posture",
                    "complete": posture_verified,
                    "evidence": f"agent.isStanding={agent_is_standing}",
                },
            ]
        elif self.is_visual_search:
            complete = (
                (target_visible or target_visible_in_environment)
                and confidence >= stop_confidence_threshold
            )
            reason = (
                "target is visually confirmed"
                if complete
                else "target confirmation threshold has not been met"
            )
        elif "exit_room" in self.task_types:
            located = (
                target_visible
                or target_visible_in_environment
                or approach_verified
                or bool(exit_evidence.get("doorObjectId"))
            )
            complete = located and exit_verified
            reason = (
                "doorway threshold crossing is verified by the environment"
                if complete
                else "exit task requires crossed-threshold evidence before termination"
            )
            subgoal_progress = [
                {
                    "id": "locate_exit",
                    "complete": located,
                    "evidence": (
                        exit_evidence.get("doorObjectId")
                        or "door or doorway visual/environment evidence"
                    ),
                },
                {
                    "id": "cross_threshold",
                    "complete": exit_verified,
                    "evidence": exit_evidence or "no threshold-crossing evidence",
                },
            ]
        elif "navigate_to" in self.task_types:
            complete = approach_verified and not missing_actions
            reason = (
                "target proximity and required interaction actions are verified"
                if complete
                else "navigation proximity must be verified by the environment before termination"
            )
        else:
            complete = bool(self.required_actions) and not missing_actions
            reason = (
                "all required task actions succeeded"
                if complete
                else "required task actions have not all succeeded"
            )
        return {
            "complete": complete,
            "outcome": (
                "approximate_success"
                if complete and self.completion_mode == "approximate_sit"
                else "exact_success"
                if complete
                else "in_progress"
            ),
            "reason": reason,
            "successful_actions": sorted(successful_actions),
            "missing_actions": missing_actions,
            "completion_mode": self.completion_mode,
            "approximate": self.completion_mode != "exact",
            "limitations": list(self.limitations),
            "target_located": target_visible or target_visible_in_environment,
            "target_visible_in_environment": target_visible_in_environment,
            "target_distance": target_distance,
            "approach_verified": approach_verified,
            "approach_object_id": approach_object_id or None,
            "approach_source": approach_source,
            "exit_verified": exit_verified,
            "exit_evidence": exit_evidence,
            "agent_is_standing": agent_is_standing,
            "subgoal_progress": subgoal_progress,
        }

    def matching_target_object_ids(
        self,
        environment_context: dict[str, Any] | None,
    ) -> set[str]:
        context = environment_context or {}
        return {
            str(item.get("objectId") or item.get("name") or "")
            for item in context.get("objects", [])
            if isinstance(item, dict)
            and self._matches_instruction_target(item)
            and (item.get("objectId") or item.get("name"))
        }

    def _matches_instruction_target(self, item: dict[str, Any]) -> bool:
        instruction = self.instruction.lower()
        object_type = str(item.get("objectType") or item.get("name") or "").lower()
        aliases = {
            "sofa": ("sofa", "couch", "沙发"),
            "armchair": ("armchair", "chair", "扶手椅", "椅子"),
            "television": ("television", "tv", "电视"),
            "door": ("door", "门", "房门", "右边的门"),
            "doorway": ("doorway", "door", "门", "门口", "出口"),
            "cup": ("cup", "杯子"),
            "mug": ("mug", "马克杯", "杯子"),
            "fridge": ("fridge", "refrigerator", "冰箱"),
            "cabinet": ("cabinet", "柜子"),
            "bowl": ("bowl", "碗"),
            "egg": ("egg", "鸡蛋"),
            "vase": ("vase", "花瓶"),
            "box": ("box", "纸箱", "箱子", "盒子"),
            "cardboardbox": ("cardboardbox", "cardboard box", "box", "纸箱", "箱子"),
        }
        terms = aliases.get(object_type, (object_type,))
        return bool(object_type) and any(term in instruction for term in terms)


class TaskSemantics:
    NAVIGATION_ACTIONS = {
        "MoveAhead",
        "MoveBack",
        "MoveLeft",
        "MoveRight",
        "MoveRelative",
        "RotateLeft",
        "RotateRight",
        "RotateAgent",
        "Rotate",
        "LookUp",
        "LookDown",
        "Pass",
    }
    INTERACTION_MARKERS = (
        (("pick up", "pickup", "grab", "拿起", "捡起", "拾取"), "pickup", "PickupObject"),
        (("put", "place into", "放入", "放到", "放在"), "put", "PutObject"),
        (("open", "打开"), "open", "OpenObject"),
        (("close", "关闭"), "close", "CloseObject"),
        (("turn on", "switch on", "开启", "打开电源"), "toggle_on", "ToggleObjectOn"),
        (("turn off", "switch off", "关掉", "关闭电源"), "toggle_off", "ToggleObjectOff"),
        (("slice", "切片", "切开"), "slice", "SliceObject"),
        (("break", "打碎", "破坏"), "break", "BreakObject"),
        (("clean", "清洁", "洗干净"), "clean", "CleanObject"),
        (("dirty", "弄脏"), "dirty", "DirtyObject"),
        (("fill", "装满", "注入"), "fill", "FillObjectWithLiquid"),
        (("empty", "倒空"), "empty", "EmptyLiquidFromObject"),
        (("use up", "用完"), "use_up", "UseUpObject"),
        (("drop", "丢下", "放手"), "drop", "DropHandObject"),
        (("throw", "扔", "投掷"), "throw", "ThrowObject"),
        (("crouch", "蹲下"), "crouch", "Crouch"),
        (("stand up", "站起", "站立"), "stand", "Stand"),
    )

    def __init__(self, catalog: AI2ThorActionCatalog | None = None):
        self.catalog = catalog or AI2ThorActionCatalog()

    def analyze(
        self,
        instruction: str,
        *,
        mode: str,
        legacy_actions: list[str] | None = None,
    ) -> TaskPlan:
        normalized = instruction.lower()
        task_types: list[str] = []
        required_actions: list[str] = []
        unsupported: list[str] = []
        limitations: list[str] = []

        if any(marker in normalized for marker in ("find", "locate", "search", "找到", "寻找", "查找", "搜索")):
            task_types.append("visual_search")
        if any(
            marker in normalized
            for marker in (
                "go to",
                "walk to",
                "walk out",
                "go out",
                "exit",
                "leave the room",
                "move to",
                "approach",
                "navigate to",
                "走到",
                "走出",
                "走出去",
                "出去",
                "离开房间",
                "移动到",
                "靠近",
                "导航到",
            )
        ):
            task_types.append("navigate_to")
        for markers, task_type, action in self.INTERACTION_MARKERS:
            if any(marker in normalized for marker in markers):
                task_types.append(task_type)
                required_actions.append(action)
        exit_requested = any(
            marker in normalized
            for marker in (
                "walk out",
                "go out",
                "exit",
                "leave the room",
                "走出去",
                "走出",
                "出去",
                "离开房间",
            )
        )
        if exit_requested:
            task_types.extend(("navigate_to", "exit_room"))
            limitations.append("exit_requires_crossed_threshold_verification")

        put_requested = "PutObject" in required_actions
        if put_requested:
            task_types.append("navigate_to")
            if "PickupObject" not in required_actions:
                required_actions.insert(0, "PickupObject")

        sit_requested = any(
            marker in normalized
            for marker in ("sit on", "sit down", "坐下", "坐到", "坐在")
        )
        if sit_requested:
            task_types.extend(("navigate_to", "sit_approximation"))
            required_actions.append("Crouch")
            limitations.append("native_sit_on_furniture_state_unavailable")
        if not task_types:
            task_types.append("visual_search")

        task_types = list(dict.fromkeys(task_types))
        required_actions = list(dict.fromkeys(required_actions))
        completion_mode = "approximate_sit" if sit_requested else "exact"
        if unsupported:
            clarification = (
                "Default AI2-THOR agents do not expose a verified sit-on-furniture state. "
                "Use 'approach the sofa and crouch' only if crouching is an acceptable substitute."
            )
            completion_rule = "unsupported task must not be reported as successful"
        elif sit_requested:
            clarification = (
                "AI2-THOR has no native SitOnObject state. The executable simulation plan "
                "uses a clearly labeled approach-and-Crouch approximation."
            )
            completion_rule = (
                "locate target, verify simulator proximity, execute Crouch, and verify "
                "agent.isStanding is false; report the result as an approximation"
            )
        elif task_types == ["visual_search"]:
            clarification = None
            completion_rule = "target must be visually grounded above the configured stop threshold"
        elif exit_requested:
            clarification = None
            completion_rule = (
                "ground the requested door or doorway, navigate through the right-side "
                "threshold, then terminate only after crossed-threshold evidence"
            )
        elif "navigate_to" in task_types and not required_actions:
            clarification = None
            completion_rule = "environment must verify target proximity before Done"
        else:
            clarification = None
            completion_rule = "every required interaction action must execute successfully before Done"

        candidate_names = set(legacy_actions or [])
        candidate_names.update(self.NAVIGATION_ACTIONS)
        candidate_names.update(required_actions)
        if mode == "drone":
            candidate_names.update(
                {"FlyAhead", "FlyBack", "FlyLeft", "FlyRight", "FlyUp", "FlyDown", "FlyTo"}
            )
        if mode in {"arm", "stretch", "stretchab"}:
            candidate_names.update(
                {
                    "MoveArm",
                    "MoveArmRelative",
                    "MoveArmBase",
                    "MoveArmBaseUp",
                    "MoveArmBaseDown",
                    "RotateWrist",
                    "RotateWristRelative",
                    "SetGripperOpenness",
                    "ReleaseObject",
                }
            )
        candidate_names.add("Done")
        if unsupported:
            candidate_names = {"ASK_CLARIFY"}

        subgoals: list[dict[str, Any]] = []
        if "visual_search" in task_types or "navigate_to" in task_types:
            subgoals.append(
                {
                    "id": "locate_target",
                    "description": "Locate and ground the requested target object",
                    "success_evidence": "target observation or AI2-THOR object visibility",
                }
            )
        if "navigate_to" in task_types:
            subgoals.append(
                {
                    "id": "approach_target",
                    "description": "Navigate until AI2-THOR verifies target proximity",
                    "success_evidence": "matching target is visible in environment metadata",
                }
            )
        for action in required_actions:
            subgoals.append(
                {
                    "id": f"execute_{action.lower()}",
                    "description": f"Execute {action}",
                    "success_evidence": f"{action} succeeds in AI2-THOR",
                }
            )
        if sit_requested:
            subgoals.append(
                {
                    "id": "verify_posture",
                    "description": "Verify the crouched posture near the target furniture",
                    "success_evidence": "agent.isStanding is false after successful Crouch",
                }
            )

        action_candidates: list[str] = []
        action_specs: list[dict[str, Any]] = []
        for name in sorted(candidate_names):
            if name == "ASK_CLARIFY":
                action_candidates.append(name)
                action_specs.append(
                    {
                        "name": name,
                        "native_action": None,
                        "category": "control",
                        "parameters": [{"name": "reason", "type": "string", "required": False}],
                    }
                )
                continue
            native_name = self.catalog.resolve_name(name)
            spec = self.catalog.get(native_name)
            if (
                spec is None
                or mode not in spec["modes"]
                or spec["exposure"] != "agent"
                or mode not in spec.get("planner_modes", [])
            ):
                continue
            overloads = spec.get("overloads_by_mode", {}).get(mode, [])
            preferred = min(
                overloads,
                key=lambda overload: (
                    len([p for p in overload["parameters"] if p["required"]]),
                    len(overload["parameters"]),
                ),
                default={},
            )
            action_candidates.append(name)
            action_specs.append(
                {
                    "name": name,
                    "native_action": native_name,
                    "category": spec["category"],
                    "parameters": preferred.get("parameters", []),
                    "legacy_server_action": bool(preferred.get("legacy_server_action")),
                }
            )

        return TaskPlan(
            instruction=instruction,
            mode=mode,
            task_types=tuple(task_types),
            required_actions=tuple(required_actions),
            unsupported_capabilities=tuple(unsupported),
            limitations=tuple(limitations),
            completion_mode=completion_mode,
            subgoals=tuple(subgoals),
            completion_rule=completion_rule,
            clarification=clarification,
            action_candidates=tuple(action_candidates),
            action_specs=tuple(action_specs),
        )
