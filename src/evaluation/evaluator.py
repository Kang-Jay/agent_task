from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.agent.controller import EmbodiedSearchAgent
from src.data.generate_demo_dataset import build_dataset
from src.task.config import AgentConfig, load_config
from src.types.schema import AgentRequest


@dataclass(frozen=True)
class EvalResult:
    episodes: int
    successes: int
    illegal_actions: int
    average_confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "episodes": self.episodes,
            "successes": self.successes,
            "success_rate": self.successes / self.episodes if self.episodes else 0.0,
            "illegal_actions": self.illegal_actions,
            "average_confidence": self.average_confidence,
        }


def load_episodes(config: AgentConfig) -> list[dict[str, Any]]:
    if not config.annotation_file.exists():
        build_dataset()
    episodes = []
    with config.annotation_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                episodes.append(json.loads(line))
    return episodes


def validate_dataset(config: AgentConfig) -> None:
    episodes = load_episodes(config)
    allowed = set(config.allowed_actions)
    for episode in episodes:
        image_path = config.data_root / episode["image"]
        if not image_path.exists():
            raise ValueError(f"Missing image: {image_path}")
        target = episode["target"]
        bbox = target["bbox"]
        if len(bbox) != 4 or bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
            raise ValueError(f"Invalid bbox for {episode['episode_id']}")
        if not any(step.get("done") for step in episode["steps"]):
            raise ValueError(f"Episode lacks stop condition: {episode['episode_id']}")
        for step in episode["steps"]:
            if step["action"] not in allowed:
                raise ValueError(f"Illegal annotated action {step['action']} in {episode['episode_id']}")


def evaluate(config: AgentConfig | None = None) -> EvalResult:
    config = config or load_config()
    validate_dataset(config)
    agent = EmbodiedSearchAgent(config)
    episodes = load_episodes(config)
    successes = 0
    illegal_actions = 0
    confidences: list[float] = []
    for episode in episodes:
        agent.reset(episode["episode_id"])
        image_path = config.data_root / episode["image"]
        request = AgentRequest(
            session_id=episode["episode_id"],
            instruction=episode["instruction"],
            observation_image=str(image_path),
            step_id=0,
        )
        response = agent.step(request)
        confidences.append(response.confidence)
        if response.action.type not in config.allowed_actions:
            illegal_actions += 1
        expected = episode["expected_action"]
        if response.action.type == expected and response.confidence >= config.stop_confidence_threshold:
            successes += 1
    return EvalResult(
        episodes=len(episodes),
        successes=successes,
        illegal_actions=illegal_actions,
        average_confidence=sum(confidences) / len(confidences) if confidences else 0.0,
    )


def main() -> None:
    result = evaluate()
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    if result.successes != result.episodes or result.illegal_actions:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

