from __future__ import annotations

from copy import deepcopy
import json
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from src.agent.controller import EmbodiedSearchAgent
from src.agent.model_adapter import ModelAdapter
from src.data.generate_demo_dataset import build_dataset
from src.evaluation.metrics import aggregate_metrics, evaluate_episode
from src.task.config import AgentConfig, load_config
from src.types.schema import AgentRequest


@dataclass(frozen=True)
class EvalResult:
    episodes: int
    successes: int
    illegal_actions: int
    average_confidence: float
    average_iou: float
    spl: float | None
    spl_coverage: float
    model_planner_usage_rate: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "episodes": self.episodes,
            "successes": self.successes,
            "success_rate": self.successes / self.episodes if self.episodes else 0.0,
            "illegal_actions": self.illegal_actions,
            "average_confidence": self.average_confidence,
            "average_iou": self.average_iou,
            "spl": self.spl,
            "spl_coverage": self.spl_coverage,
            "model_planner_usage_rate": self.model_planner_usage_rate,
        }


def load_episodes(config: AgentConfig) -> list[dict[str, Any]]:
    if not config.annotation_file.exists():
        build_dataset(config)
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


def evaluate(
    config: AgentConfig | None = None,
    agent: EmbodiedSearchAgent | None = None,
) -> EvalResult:
    config = config or load_config()
    validate_dataset(config)
    temporary_directory: TemporaryDirectory[str] | None = None
    if agent is None:
        temporary_directory = TemporaryDirectory()
        evaluation_raw = deepcopy(config.raw)
        evaluation_raw["memory"]["persist_traces"] = False
        evaluation_raw["data"]["trajectory_dir"] = str(
            Path(temporary_directory.name) / "trajectories"
        )
        evaluation_config = AgentConfig(raw=evaluation_raw, path=config.path)
        agent = EmbodiedSearchAgent(
            evaluation_config,
            model_adapter=ModelAdapter(credentials=[]),
        )
    episodes = load_episodes(config)
    successes = 0
    illegal_actions = 0
    confidences: list[float] = []
    episode_metrics = []
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
        response_dict = response.to_dict()
        confidences.append(response.confidence)
        if response.action.type not in config.allowed_actions:
            illegal_actions += 1
        metrics = evaluate_episode(
            episode,
            {
                "steps": [response_dict],
                "execution_time": 0.0,
            },
            config,
        )
        episode_metrics.append(metrics)
        if metrics.success:
            successes += 1
    dataset_metrics = aggregate_metrics(episode_metrics)
    result = EvalResult(
        episodes=len(episodes),
        successes=successes,
        illegal_actions=illegal_actions,
        average_confidence=sum(confidences) / len(confidences) if confidences else 0.0,
        average_iou=dataset_metrics.average_iou,
        spl=dataset_metrics.spl,
        spl_coverage=dataset_metrics.spl_coverage,
        model_planner_usage_rate=dataset_metrics.model_planner_usage_rate,
    )
    if temporary_directory is not None:
        temporary_directory.cleanup()
    return result


def main() -> None:
    result = evaluate()
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    if result.successes != result.episodes or result.illegal_actions:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
