from __future__ import annotations

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
    completion_rule: str
    clarification: str | None
    action_candidates: tuple[str, ...]
    action_specs: tuple[dict[str, Any], ...]

    @property
    def supported(self) -> bool:
        return not self.unsupported_capabilities

    @property
    def is_visual_search(self) -> bool:
        return self.task_types == ("visual_search",)

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
    ) -> dict[str, Any]:
        successful_actions = {
            str((step.get("executed_action") or step.get("action") or {}).get("type"))
            for step in steps
            if step.get("action_success") is True
        }
        missing_actions = [
            action for action in self.required_actions if action not in successful_actions
        ]
        if not self.supported:
            complete = False
            reason = "task requests unsupported embodied capabilities"
        elif self.is_visual_search:
            complete = target_visible and confidence >= stop_confidence_threshold
            reason = (
                "target is visually confirmed"
                if complete
                else "target confirmation threshold has not been met"
            )
        elif "navigate_to" in self.task_types:
            complete = not missing_actions and bool(self.required_actions)
            reason = (
                "required interaction actions succeeded"
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
            "reason": reason,
            "successful_actions": sorted(successful_actions),
            "missing_actions": missing_actions,
        }


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

        if any(marker in normalized for marker in ("find", "locate", "search", "找到", "寻找", "查找", "搜索")):
            task_types.append("visual_search")
        if any(
            marker in normalized
            for marker in (
                "go to",
                "walk to",
                "move to",
                "approach",
                "navigate to",
                "走到",
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

        if any(marker in normalized for marker in ("sit on", "sit down", "坐下", "坐到", "坐在")):
            unsupported.append("human_or_robot_sitting_pose")
        if not task_types:
            task_types.append("visual_search")

        task_types = list(dict.fromkeys(task_types))
        required_actions = list(dict.fromkeys(required_actions))
        if unsupported:
            clarification = (
                "Default AI2-THOR agents do not expose a verified sit-on-furniture state. "
                "Use 'approach the sofa and crouch' only if crouching is an acceptable substitute."
            )
            completion_rule = "unsupported task must not be reported as successful"
        elif task_types == ["visual_search"]:
            clarification = None
            completion_rule = "target must be visually grounded above the configured stop threshold"
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
            completion_rule=completion_rule,
            clarification=clarification,
            action_candidates=tuple(action_candidates),
            action_specs=tuple(action_specs),
        )
