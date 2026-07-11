from __future__ import annotations

from copy import deepcopy
import json
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from src.agent.controller import EmbodiedSearchAgent
from src.agent.model_adapter import ModelAdapter
from src.evaluation.manifest import BenchmarkManifest, EpisodeSpec, load_manifest
from src.data.generate_demo_dataset import build_dataset
from src.evaluation.metrics import (
    EpisodeMetrics,
    aggregate_metrics,
    dataset_metrics_to_dict,
    evaluate_episode,
)
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


def episode_spec_to_data(episode: EpisodeSpec) -> dict[str, Any]:
    """Convert a frozen manifest episode into the metrics input shape."""

    data = episode.to_dict()
    data["task_type"] = episode.task.task_type
    data["target"] = dict(episode.task.target)
    data["required_actions"] = list(episode.task.required_actions)
    data["allows_approximate_success"] = episode.task.allows_approximate_success
    data["optimal_path_length_meters"] = episode.reference.optimal_path_length_meters
    return data


def _safe_result_path(output_dir: Path, result_file: str) -> Path:
    root = output_dir.resolve()
    candidate = (root / result_file).resolve()
    if root != candidate and root not in candidate.parents:
        raise ValueError(f"result path escapes output directory: {result_file}")
    return candidate


def _normalise_step(step: dict[str, Any]) -> dict[str, Any]:
    if "action" in step and isinstance(step["action"], dict):
        action = dict(step["action"])
    else:
        action_value = step.get("action") or step.get("action_type") or ""
        action = {"type": action_value}
    observation = step.get("observation")
    if not isinstance(observation, dict):
        observation = {}
    if "best_candidate" not in observation and "best_candidate" in step:
        observation["best_candidate"] = step["best_candidate"]
    return {
        **step,
        "action": action,
        "done": bool(step.get("done", False)),
        "confidence": float(step.get("confidence", 0.0) or 0.0),
        "planner_source": step.get("planner_source", step.get("source", "unknown")),
        "observation": observation,
    }


def load_result_trajectory(path: Path) -> dict[str, Any]:
    """Load a stored episode result or AI2-THOR summary for metric aggregation."""

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"missing result file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid result JSON {path}: line {exc.lineno}") from exc

    if not isinstance(raw, dict):
        raise ValueError(f"result file must contain a JSON object: {path}")
    steps_raw = raw.get("steps")
    if not isinstance(steps_raw, list):
        raise ValueError(f"result file lacks steps array: {path}")
    steps: list[dict[str, Any]] = []
    for index, step in enumerate(steps_raw):
        if not isinstance(step, dict):
            raise ValueError(
                f"result file step {index} must be a JSON object: {path}"
            )
        steps.append(_normalise_step(step))
    return {**raw, "steps": steps}


def filter_manifest_episodes(
    manifest: BenchmarkManifest,
    *,
    split: str | None = None,
    group: str | None = None,
) -> list[EpisodeSpec]:
    episodes = list(manifest.ordered_episodes())
    if split:
        episodes = [episode for episode in episodes if episode.split == split]
    if group:
        episodes = [episode for episode in episodes if episode.group == group]
    return episodes


def build_manifest_summary(
    manifest: BenchmarkManifest,
    episode_metrics: list[EpisodeMetrics],
    *,
    manifest_path: Path,
    output_dir: Path,
    dry_run: bool,
) -> dict[str, Any]:
    dataset_metrics = aggregate_metrics(episode_metrics)
    coverage = manifest.coverage()
    return {
        "benchmark_id": manifest.benchmark_id,
        "dataset_version": manifest.dataset_version,
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest.sha256(),
        "output_dir": str(output_dir),
        "dry_run": dry_run,
        "inference_only": manifest.inference_only,
        "episode_count": coverage["episode_count"],
        "pair_count": coverage["pair_count"],
        "scenes": coverage["scenes"],
        "splits": coverage["splits"],
        "groups": coverage["groups"],
        "task_types": coverage["task_types"],
        "evaluated_episodes": len(episode_metrics),
        "metrics": dataset_metrics_to_dict(dataset_metrics),
        "episodes": [
            {
                "episode_id": metric.episode_id,
                "success": metric.success,
                "strict_success": metric.strict_success,
                "approximate_success": metric.approximate_success,
                "interaction_success": metric.interaction_success,
                "misstop": metric.misstop,
                "collision_count": metric.collision_count,
                "path_length": metric.path_length,
                "spl_eligible": metric.spl_eligible,
                "group": metric.group,
                "task_type": metric.task_type,
                "scene": metric.scene,
                "split": metric.split,
            }
            for metric in episode_metrics
        ],
    }


def evaluate_manifest_results(
    manifest_path: Path | str,
    output_dir: Path | str,
    *,
    split: str | None = None,
    group: str | None = None,
    dry_run: bool = False,
    config: AgentConfig | None = None,
) -> dict[str, Any]:
    """Validate a Plan2 manifest and optionally aggregate existing episode results."""

    config = config or load_config()
    manifest_path = Path(manifest_path)
    output_dir = Path(output_dir)
    manifest = load_manifest(manifest_path)
    selected = filter_manifest_episodes(manifest, split=split, group=group)
    episode_metrics: list[EpisodeMetrics] = []
    if not dry_run:
        for episode in selected:
            result_path = _safe_result_path(output_dir, episode.result_file)
            trajectory = load_result_trajectory(result_path)
            episode_metrics.append(
                evaluate_episode(episode_spec_to_data(episode), trajectory, config)
            )
    return build_manifest_summary(
        manifest,
        episode_metrics,
        manifest_path=manifest_path,
        output_dir=output_dir,
        dry_run=dry_run,
    )


def write_manifest_summary(summary: dict[str, Any], output_dir: Path | str) -> Path:
    output_path = Path(output_dir) / "summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output_path


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
