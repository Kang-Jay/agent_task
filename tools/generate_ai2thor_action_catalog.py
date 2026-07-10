from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = ROOT / "research" / "codebases" / "ai2thor" / "source"
DEFAULT_OUTPUT = ROOT / "configs" / "ai2thor_actions_v5.json"

POSSIBLE_DISPATCH_RETURN_TYPES = {"void", "ActionFinished", "IEnumerator"}
MODE_CONTROLLER_CANDIDATES = {
    "default": ("PhysicsRemoteFPSAgentController",),
    "locobot": ("LocobotFPSAgentController",),
    "drone": ("DroneFPSAgentController",),
    "arm": ("KinovaArmAgentController", "ArmAgentController"),
    "stretch": ("StretchAgentController",),
    "stretchab": ("ArticulatedAgentController",),
    "fpin": ("FpinAgentController",),
}

AGENT_ACTIONS = {
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
    "Crouch",
    "Stand",
    "PickupObject",
    "PutObject",
    "DropHandObject",
    "ThrowObject",
    "ReleaseObject",
    "OpenObject",
    "CloseObject",
    "ToggleObjectOn",
    "ToggleObjectOff",
    "SliceObject",
    "BreakObject",
    "DirtyObject",
    "CleanObject",
    "FillObjectWithLiquid",
    "EmptyLiquidFromObject",
    "UseUpObject",
    "PushObject",
    "PullObject",
    "TouchThenApplyForce",
    "MoveHeldObject",
    "MoveHeldObjectAhead",
    "MoveHeldObjectBack",
    "MoveHeldObjectLeft",
    "MoveHeldObjectRight",
    "MoveHeldObjectUp",
    "MoveHeldObjectDown",
    "RotateHeldObject",
    "MoveArm",
    "MoveArmRelative",
    "MoveArmBase",
    "MoveArmBaseUp",
    "MoveArmBaseDown",
    "RotateWrist",
    "RotateWristRelative",
    "SetGripperOpenness",
    "Pass",
    "Done",
}

QUERY_PREFIXES = (
    "Get",
    "ObjectTypeTo",
    "Check",
    "VisibleRange",
    "ApproxPercent",
    "BBoxDistance",
    "SimObjPhysicsType",
)
SYSTEM_PREFIXES = (
    "Reset",
    "Initialize",
    "Teleport",
    "Random",
    "Create",
    "Destroy",
    "Remove",
    "Spawn",
    "Place",
    "Parent",
    "Unparent",
    "Set",
    "Change",
    "ToggleMap",
    "Bake",
    "ReBake",
    "Overwrite",
    "Pause",
    "Unpause",
    "AdvancePhysics",
    "MakeAll",
    "Visualize",
    "HideVisualized",
    "AddThirdParty",
    "UpdateMainCamera",
    "UpdateThirdPartyCamera",
)
INTERNAL_NAMES = {
    "Start",
    "Update",
    "FixedUpdate",
    "LateUpdate",
    "Complete",
    "ProcessControlCommand",
    "InitializeBody",
    "EmitFrame",
    "ResetCoroutine",
    "WaitOnResolutionChange",
    "createPayload",
    "registerAsThirdPartyCamera",
    "updateImageSynthesis",
    "updateThirdPartyCameraImageSynthesis",
    "updateAntiAliasing",
    "actionFinished",
    "unrollSimulatePhysics",
    "print",
}
INTERNAL_PREFIXES = (
    "Test",
    "Debug",
    "On",
    "checkInitialize",
    "spawnAgent",
    "destroyAgent",
    "DeleteMe",
)


@dataclass(frozen=True)
class ClassBlock:
    name: str
    base: str | None
    source: Path
    body: str
    body_offset: int


def _mask_non_code(text: str) -> str:
    chars = list(text)
    index = 0
    state = "code"
    while index < len(chars):
        char = chars[index]
        next_char = chars[index + 1] if index + 1 < len(chars) else ""
        if state == "code":
            if char == "/" and next_char == "/":
                chars[index] = chars[index + 1] = " "
                index += 2
                state = "line_comment"
                continue
            if char == "/" and next_char == "*":
                chars[index] = chars[index + 1] = " "
                index += 2
                state = "block_comment"
                continue
            if char == '"':
                chars[index] = " "
                state = "string"
            elif char == "'":
                chars[index] = " "
                state = "char"
        elif state == "line_comment":
            if char == "\n":
                state = "code"
            else:
                chars[index] = " "
        elif state == "block_comment":
            chars[index] = " "
            if char == "*" and next_char == "/":
                chars[index + 1] = " "
                index += 2
                state = "code"
                continue
        elif state in {"string", "char"}:
            chars[index] = " "
            if char == "\\":
                if index + 1 < len(chars):
                    chars[index + 1] = " "
                    index += 2
                    continue
            if (state == "string" and char == '"') or (state == "char" and char == "'"):
                state = "code"
        index += 1
    return "".join(chars)


def _matching_delimiter(masked: str, start: int, opening: str, closing: str) -> int:
    depth = 0
    for index in range(start, len(masked)):
        char = masked[index]
        if char == opening:
            depth += 1
        elif char == closing:
            depth -= 1
            if depth == 0:
                return index
    raise ValueError(f"Unbalanced {opening}{closing} delimiter at offset {start}")


def _class_blocks(source_root: Path) -> dict[str, list[ClassBlock]]:
    scripts_root = source_root / "unity" / "Assets" / "Scripts"
    class_pattern = re.compile(
        r"public\s+(?:(?:abstract|partial|sealed)\s+)*class\s+"
        r"(?P<name>[A-Za-z_]\w*)"
        r"(?:\s*:\s*(?P<bases>[A-Za-z0-9_<>,.\s]+))?\s*\{",
        re.MULTILINE,
    )
    blocks: dict[str, list[ClassBlock]] = {}
    for path in sorted(scripts_root.rglob("*.cs")):
        text = path.read_text(encoding="utf-8", errors="replace")
        masked = _mask_non_code(text)
        for match in class_pattern.finditer(masked):
            open_brace = masked.find("{", match.start(), match.end())
            close_brace = _matching_delimiter(masked, open_brace, "{", "}")
            bases = (match.group("bases") or "").strip()
            base = bases.split(",", 1)[0].strip() or None
            name = match.group("name")
            blocks.setdefault(name, []).append(
                ClassBlock(
                    name=name,
                    base=base,
                    source=path,
                    body=text[open_brace + 1 : close_brace],
                    body_offset=open_brace + 1,
                )
            )
    return blocks


def _split_top_level(text: str, delimiter: str = ",") -> list[str]:
    parts: list[str] = []
    start = 0
    depth = 0
    pairs = {"(": ")", "[": "]", "{": "}", "<": ">"}
    openings = set(pairs)
    closings = set(pairs.values())
    for index, char in enumerate(text):
        if char in openings:
            depth += 1
        elif char in closings:
            depth = max(0, depth - 1)
        elif char == delimiter and depth == 0:
            parts.append(text[start:index].strip())
            start = index + 1
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def _parse_default(raw: str) -> Any:
    value = raw.strip()
    if value in {"null", "default"}:
        return None
    if value == "true":
        return True
    if value == "false":
        return False
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    numeric = re.fullmatch(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?[fFdDmM]?", value)
    if numeric:
        normalized = value.rstrip("fFdDmM")
        return float(normalized) if any(token in normalized for token in (".", "e", "E")) else int(normalized)
    return value


def _parse_parameters(raw: str) -> tuple[list[dict[str, Any]], bool]:
    raw = re.sub(r"//[^\n]*(?=\n|$)", "", raw)
    raw = re.sub(r"/\*.*?\*/", "", raw, flags=re.DOTALL)
    parameters: list[dict[str, Any]] = []
    legacy_server_action = False
    for item in _split_top_level(raw):
        item = re.sub(r"^\[[^\]]+\]\s*", "", item.strip())
        if not item:
            continue
        default_raw: str | None = None
        default_parts = _split_top_level(item, "=")
        if len(default_parts) > 1:
            item = default_parts[0]
            default_raw = "=".join(default_parts[1:]).strip()
        tokens = item.split()
        while tokens and tokens[0] in {"ref", "out", "in", "params", "this"}:
            tokens.pop(0)
        if len(tokens) < 2:
            continue
        name = tokens[-1]
        type_name = " ".join(tokens[:-1])
        if type_name == "ServerAction":
            legacy_server_action = True
        parameter: dict[str, Any] = {
            "name": name,
            "type": type_name,
            "required": default_raw is None,
        }
        if default_raw is not None:
            parameter["default"] = _parse_default(default_raw)
            parameter["default_source"] = default_raw
        parameters.append(parameter)
    return parameters, legacy_server_action


def _methods_for_block(
    block: ClassBlock,
    source_root: Path,
    dispatch_return_types: set[str],
) -> list[dict[str, Any]]:
    method_pattern = re.compile(
        r"public\s+"
        r"(?P<modifiers>(?:(?:virtual|override|new|async|sealed|extern|static)\s+)*)"
        r"(?P<return>void|ActionFinished|IEnumerator)\s+"
        r"(?P<name>[A-Za-z_]\w*)\s*\(",
        re.MULTILINE,
    )
    body_masked = _mask_non_code(block.body)
    methods: list[dict[str, Any]] = []
    for match in method_pattern.finditer(body_masked):
        if body_masked[: match.start()].count("{") != body_masked[: match.start()].count("}"):
            continue
        modifiers = match.group("modifiers").split()
        if "static" in modifiers:
            continue
        return_type = match.group("return")
        if return_type not in dispatch_return_types:
            continue
        open_paren = body_masked.find("(", match.start(), match.end())
        close_paren = _matching_delimiter(body_masked, open_paren, "(", ")")
        raw_parameters = block.body[open_paren + 1 : close_paren]
        parameters, legacy_server_action = _parse_parameters(raw_parameters)
        absolute_offset = block.body_offset + match.start()
        source_text = block.source.read_text(encoding="utf-8", errors="replace")
        line = source_text.count("\n", 0, absolute_offset) + 1
        methods.append(
            {
                "name": match.group("name"),
                "declaring_class": block.name,
                "return_type": return_type,
                "modifiers": modifiers,
                "parameters": parameters,
                "legacy_server_action": legacy_server_action,
                "source": str(block.source.relative_to(source_root)).replace("\\", "/"),
                "line": line,
            }
        )
    return methods


def _inheritance_chain(class_name: str, blocks: dict[str, list[ClassBlock]]) -> list[str]:
    chain: list[str] = []
    current: str | None = class_name
    while current and current not in chain:
        chain.append(current)
        candidates = blocks.get(current, [])
        current = next((block.base for block in candidates if block.base), None)
    return chain


def _classify(name: str, manager_action: bool = False) -> tuple[str, str]:
    if name in INTERNAL_NAMES or name.startswith(INTERNAL_PREFIXES) or (name and name[0].islower()):
        return "internal", "internal"
    if manager_action:
        return "manager", "system"
    if name in AGENT_ACTIONS:
        if name.startswith(("MoveArm", "RotateWrist", "SetGripper", "ReleaseObject")):
            return "arm", "agent"
        if name.startswith("Fly"):
            return "drone", "agent"
        if name.startswith(("Move", "Rotate", "Look", "Crouch", "Stand")):
            return "navigation", "agent"
        if name in {"Pass", "Done"}:
            return "control", "agent"
        return "object_interaction", "agent"
    if name.startswith(QUERY_PREFIXES):
        return "query", "manual"
    if name.startswith(SYSTEM_PREFIXES):
        return "scene_admin", "system"
    return "advanced", "manual"


def _manager_actions(source_root: Path) -> list[str]:
    path = source_root / "unity" / "Assets" / "Scripts" / "AgentManager.cs"
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(
        r"agentManagerActions\s*=\s*new\s+HashSet<string>\s*\{(?P<body>.*?)\};",
        text,
        re.DOTALL,
    )
    if not match:
        raise ValueError("Could not locate AgentManager.agentManagerActions")
    return re.findall(r'"([^"]+)"', match.group("body"))


def _dispatch_return_types(source_root: Path) -> set[str]:
    path = source_root / "unity" / "Assets" / "Scripts" / "ActionDispatcher.cs"
    text = path.read_text(encoding="utf-8", errors="replace")
    allowed = {"void"}
    if "typeof(ActionFinished)" in text:
        allowed.add("ActionFinished")
    if "typeof(IEnumerator)" in text:
        allowed.add("IEnumerator")
    return allowed & POSSIBLE_DISPATCH_RETURN_TYPES


def _git_value(source_root: Path, *args: str) -> str:
    if not (source_root / ".git").exists():
        return "unknown"
    completed = subprocess.run(
        ["git", "-C", str(source_root), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def _dedupe_overloads(overloads: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for overload in overloads:
        key = json.dumps(
            {
                "parameters": overload["parameters"],
                "legacy_server_action": overload["legacy_server_action"],
            },
            ensure_ascii=True,
            sort_keys=True,
        )
        existing = unique.get(key)
        if existing is None:
            unique[key] = overload
            continue
        if overload["declaring_class"] != existing["declaring_class"]:
            existing.setdefault("also_declared_in", []).append(
                {
                    "class": overload["declaring_class"],
                    "source": overload["source"],
                    "line": overload["line"],
                }
            )
    return sorted(
        unique.values(),
        key=lambda item: (
            len([parameter for parameter in item["parameters"] if parameter["required"]]),
            len(item["parameters"]),
            item["declaring_class"],
            item["line"],
        ),
    )


def build_catalog(
    source_root: Path,
    ai2thor_version: str,
    *,
    source_commit_override: str | None = None,
    source_date_override: str | None = None,
) -> dict[str, Any]:
    blocks = _class_blocks(source_root)
    dispatch_return_types = _dispatch_return_types(source_root)
    mode_controllers = {
        mode: next(controller for controller in candidates if controller in blocks)
        for mode, candidates in MODE_CONTROLLER_CANDIDATES.items()
        if any(controller in blocks for controller in candidates)
    }
    method_cache: dict[str, list[dict[str, Any]]] = {}
    for class_name, class_blocks in blocks.items():
        method_cache[class_name] = [
            method
            for block in class_blocks
            for method in _methods_for_block(
                block,
                source_root,
                dispatch_return_types,
            )
        ]

    actions: dict[str, dict[str, Any]] = {}
    mode_chains: dict[str, list[str]] = {}
    for mode, controller_class in mode_controllers.items():
        chain = _inheritance_chain(controller_class, blocks)
        mode_chains[mode] = chain
        for class_name in reversed(chain):
            for overload in method_cache.get(class_name, []):
                action = actions.setdefault(
                    overload["name"],
                    {
                        "name": overload["name"],
                        "modes": [],
                        "manager_action": False,
                        "runtime_available": True,
                        "overloads_by_mode": {},
                    },
                )
                if mode not in action["modes"]:
                    action["modes"].append(mode)
                action["overloads_by_mode"].setdefault(mode, []).append(overload)

    manager_actions = _manager_actions(source_root)
    for name in manager_actions:
        manager_overloads = [
            overload
            for overload in method_cache.get("AgentManager", [])
            if overload["name"] == name
        ]
        runtime_available = bool(manager_overloads)
        action = actions.setdefault(
            name,
            {
                "name": name,
                "modes": list(mode_controllers),
                "manager_action": True,
                "runtime_available": runtime_available,
                "overloads_by_mode": {},
            },
        )
        action["manager_action"] = True
        action["runtime_available"] = runtime_available
        action["modes"] = sorted(set(action["modes"]) | set(mode_controllers))
        if manager_overloads:
            for mode in mode_controllers:
                action["overloads_by_mode"].setdefault(mode, []).extend(
                    manager_overloads
                )
        else:
            action["availability_reason"] = (
                "listed in AgentManager.agentManagerActions but no dispatchable "
                "method exists in this source snapshot"
            )

    for action in actions.values():
        category, exposure = _classify(action["name"], action["manager_action"])
        action["category"] = category
        action["exposure"] = exposure
        action["modes"] = sorted(action["modes"])
        planner_modes = list(action["modes"]) if exposure == "agent" else []
        if action["name"] in {"MoveLeft", "MoveRight"}:
            planner_modes = [mode for mode in planner_modes if mode != "locobot"]
        action["planner_modes"] = sorted(planner_modes)
        for mode, overloads in list(action["overloads_by_mode"].items()):
            action["overloads_by_mode"][mode] = _dedupe_overloads(overloads)

    source_commit = source_commit_override or _git_value(source_root, "rev-parse", "HEAD")
    source_date = source_date_override or _git_value(
        source_root,
        "show",
        "-s",
        "--format=%cI",
        "HEAD",
    )
    catalog_actions = sorted(actions.values(), key=lambda item: item["name"])
    return {
        "schema_version": 1,
        "ai2thor_version": ai2thor_version,
        "source": {
            "repository": "https://github.com/allenai/ai2thor",
            "commit": source_commit,
            "commit_date": source_date,
            "dispatch_contract": (
                "public instance methods returning "
                + ", ".join(sorted(dispatch_return_types))
            ),
            "manager_whitelist_source": "unity/Assets/Scripts/AgentManager.cs",
            "dispatcher_source": "unity/Assets/Scripts/ActionDispatcher.cs",
        },
        "dispatch_return_types": sorted(dispatch_return_types),
        "mode_controllers": mode_controllers,
        "mode_inheritance": mode_chains,
        "manager_actions": sorted(manager_actions),
        "counts": {
            "actions": len(catalog_actions),
            "runtime_available": sum(
                action.get("runtime_available", True) for action in catalog_actions
            ),
            "agent_exposed": sum(action["exposure"] == "agent" for action in catalog_actions),
            "manual_exposed": sum(action["exposure"] == "manual" for action in catalog_actions),
            "system_exposed": sum(action["exposure"] == "system" for action in catalog_actions),
            "internal": sum(action["exposure"] == "internal" for action in catalog_actions),
            "by_mode": {
                mode: sum(mode in action["modes"] for action in catalog_actions)
                for mode in mode_controllers
            },
        },
        "actions": catalog_actions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the AI2-THOR action catalog from official C# sources.")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ai2thor-version", default="5.0.0")
    parser.add_argument("--source-commit")
    parser.add_argument("--source-date")
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    output = args.output.resolve()
    catalog = build_catalog(
        source_root,
        args.ai2thor_version,
        source_commit_override=args.source_commit,
        source_date_override=args.source_date,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(catalog["counts"], ensure_ascii=False, indent=2))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
