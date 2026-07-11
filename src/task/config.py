from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT / "configs" / "agent_config.json"


@dataclass(frozen=True)
class AgentConfig:
    raw: dict[str, Any]
    path: Path

    @property
    def allowed_actions(self) -> list[str]:
        return list(self.raw["action_space"]["allowed_actions"])

    @property
    def terminal_actions(self) -> set[str]:
        return set(self.raw["action_space"]["terminal_actions"])

    @property
    def max_steps(self) -> int:
        return int(self.raw["agent"]["max_steps"])

    @property
    def stop_confidence_threshold(self) -> float:
        return float(self.raw["agent"]["stop_confidence_threshold"])

    @property
    def target_visible_threshold(self) -> float:
        return float(self.raw["agent"]["target_visible_threshold"])

    @property
    def visual_search_authority(self) -> str:
        """Who decides visual-search target confirmation: 'vlm' or 'simulator'.

        'vlm' (default): the VLM's own visual confirmation drives INSPECT/STOP,
        while AI2-THOR instance segmentation is retained only as cross-validation
        evidence. 'simulator': legacy behavior where segmentation ground truth
        overrides the VLM decision.
        """
        value = str(
            self.raw["agent"].get("visual_search_authority", "vlm")
        ).strip().lower()
        return value if value in {"vlm", "simulator"} else "vlm"

    @property
    def history_window(self) -> int:
        return int(self.raw["agent"]["history_window"])

    @property
    def image_size(self) -> tuple[int, int]:
        width, height = self.raw["vision"]["image_size"]
        return int(width), int(height)

    @property
    def data_root(self) -> Path:
        return ROOT / self.raw["data"]["dataset_root"]

    @property
    def annotation_file(self) -> Path:
        return ROOT / self.raw["data"]["annotation_file"]

    @property
    def trajectory_dir(self) -> Path:
        return ROOT / self.raw["data"]["trajectory_dir"]

    @property
    def image_dir(self) -> Path:
        return ROOT / self.raw["data"]["image_dir"]


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> AgentConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    config = AgentConfig(raw=raw, path=config_path)
    validate_config(config)
    return config


def validate_config(config: AgentConfig) -> None:
    raw = config.raw
    required_top_level = {"project", "pipeline", "agent", "vision", "memory", "action_space", "data", "evaluation"}
    missing = required_top_level.difference(raw)
    if missing:
        raise ValueError(f"Missing config sections: {sorted(missing)}")

    allowed = raw["action_space"]["allowed_actions"]
    if not allowed or len(allowed) != len(set(allowed)):
        raise ValueError("allowed_actions must be non-empty and unique")

    terminal = set(raw["action_space"]["terminal_actions"])
    if not terminal.issubset(set(allowed)):
        raise ValueError("terminal_actions must be a subset of allowed_actions")

    if "STOP" not in allowed:
        raise ValueError("STOP action is required for visual search completion")

    if raw["evaluation"]["max_episode_steps"] != raw["agent"]["max_steps"]:
        raise ValueError("evaluation.max_episode_steps must equal agent.max_steps")

    stop_threshold = float(raw["agent"]["stop_confidence_threshold"])
    eval_threshold = float(raw["evaluation"]["min_success_confidence"])
    if stop_threshold != eval_threshold:
        raise ValueError("agent stop threshold and evaluation success threshold must match")

    min_success_iou = float(raw["evaluation"]["min_success_iou"])
    if not 0.0 <= min_success_iou <= 1.0:
        raise ValueError("evaluation.min_success_iou must be between 0 and 1")

    weights = [
        float(raw["vision"]["color_match_weight"]),
        float(raw["vision"]["text_match_weight"]),
        float(raw["vision"]["center_prior_weight"]),
    ]
    if abs(sum(weights) - 1.0) > 1e-6:
        raise ValueError("vision match weights must sum to 1.0")

    memory_capacity = int(raw["memory"]["long_term_capacity"])
    negative_capacity = int(raw["memory"]["negative_memory_capacity"])
    retrieval_top_k = int(raw["memory"]["retrieval_top_k"])
    if memory_capacity <= 0:
        raise ValueError("memory.long_term_capacity must be positive")
    if negative_capacity <= 0:
        raise ValueError("memory.negative_memory_capacity must be positive")
    if retrieval_top_k <= 0 or retrieval_top_k > memory_capacity:
        raise ValueError(
            "memory.retrieval_top_k must be positive and no greater than long_term_capacity"
        )

    stages = raw["pipeline"]["stages"]
    expected = [
        "validate_request",
        "decode_observation",
        "load_memory",
        "analyze_vision",
        "retrieve_hints",
        "plan_action",
        "validate_action",
        "update_memory",
        "emit_response",
    ]
    if stages != expected:
        raise ValueError("pipeline stages differ from the documented project pipeline")
