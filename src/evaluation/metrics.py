"""Plan2 evaluation metrics for embodied visual search and interaction tasks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.task.config import AgentConfig


@dataclass
class EpisodeMetrics:
    """Metrics for a single evaluated episode."""

    episode_id: str
    success: bool
    path_length: float
    optimal_path_length: float
    final_iou: float
    confidence_at_stop: float
    illegal_actions: int
    planner_source_counts: dict[str, int]
    execution_time_seconds: float
    spl_eligible: bool = False
    category: str = "unknown"
    difficulty: str = "unknown"
    task_type: str = "unknown"
    group: str = "unknown"
    scene: str = "unknown"
    split: str = "unknown"
    step_count: int = 0
    path_length_meters: float | None = None
    optimal_path_length_meters: float | None = None
    collision_count: int = 0
    misstop: bool = False
    interaction_success: bool = False
    approximate_success: bool = False
    strict_success: bool = False


@dataclass
class DatasetMetrics:
    """Aggregated metrics for an evaluation run."""

    total_episodes: int
    success_rate: float
    spl: float | None
    spl_coverage: float
    average_path_length: float
    average_optimal_path_length: float
    average_iou: float
    illegal_action_rate: float
    model_planner_usage_rate: float
    average_confidence_at_success: float
    per_category_success: dict[str, float]
    per_difficulty_success: dict[str, float]
    collision_count: int = 0
    collision_rate: float = 0.0
    misstop_count: int = 0
    misstop_rate: float = 0.0
    interaction_success_rate: float = 0.0
    approximate_success_rate: float = 0.0
    strict_success_rate: float = 0.0
    by_group: dict[str, dict[str, float | int | None]] | None = None
    by_task_type: dict[str, dict[str, float | int | None]] | None = None
    by_scene: dict[str, dict[str, float | int | None]] | None = None
    by_split: dict[str, dict[str, float | int | None]] | None = None


def compute_iou(pred_bbox: list[int] | None, gt_bbox: list[int] | None) -> float:
    """Compute intersection over union for [x1, y1, x2, y2] boxes."""

    if not pred_bbox or not gt_bbox:
        return 0.0
    if len(pred_bbox) != 4 or len(gt_bbox) != 4:
        return 0.0

    x1_inter = max(pred_bbox[0], gt_bbox[0])
    y1_inter = max(pred_bbox[1], gt_bbox[1])
    x2_inter = min(pred_bbox[2], gt_bbox[2])
    y2_inter = min(pred_bbox[3], gt_bbox[3])

    if x2_inter < x1_inter or y2_inter < y1_inter:
        return 0.0

    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    pred_area = (pred_bbox[2] - pred_bbox[0]) * (pred_bbox[3] - pred_bbox[1])
    gt_area = (gt_bbox[2] - gt_bbox[0]) * (gt_bbox[3] - gt_bbox[1])
    union_area = pred_area + gt_area - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def compute_spl(success: bool, path_length: float, optimal_path_length: float) -> float:
    """Compute Success weighted by Path Length."""

    if not success:
        return 0.0
    if optimal_path_length == 0:
        return 1.0 if path_length == 0 else 0.0
    return optimal_path_length / max(optimal_path_length, path_length)


def _action_type(step: dict[str, Any]) -> str:
    action = step.get("action")
    if isinstance(action, dict):
        value = action.get("type") or action.get("action")
    else:
        value = action
    return str(value or "")


def _step_succeeded(step: dict[str, Any]) -> bool:
    if "success" in step:
        return bool(step["success"])
    if "lastActionSuccess" in step:
        return bool(step["lastActionSuccess"])
    if isinstance(step.get("environment_feedback"), dict):
        feedback = step["environment_feedback"]
        if "lastActionSuccess" in feedback:
            return bool(feedback["lastActionSuccess"])
        if "success" in feedback:
            return bool(feedback["success"])
    if isinstance(step.get("metadata"), dict) and "lastActionSuccess" in step["metadata"]:
        return bool(step["metadata"]["lastActionSuccess"])
    return True


def _collision_count(steps: list[dict[str, Any]]) -> int:
    collisions = 0
    for step in steps:
        action = _action_type(step)
        if action not in {"MoveAhead", "MoveBack", "MoveLeft", "MoveRight"}:
            continue
        if not _step_succeeded(step):
            collisions += 1
    return collisions


def _planner_source_counts(steps: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for step in steps:
        source = str(step.get("planner_source") or step.get("source") or "unknown")
        counts[source] = counts.get(source, 0) + 1
    return counts


def _candidate_bbox(step: dict[str, Any]) -> list[int] | None:
    best = step.get("observation", {}).get("best_candidate")
    if isinstance(best, dict):
        bbox = best.get("bbox")
        if isinstance(bbox, list):
            return bbox
    bbox = step.get("best_candidate_bbox") or step.get("bbox")
    return bbox if isinstance(bbox, list) else None


def _target_bbox(episode_data: dict[str, Any]) -> list[int] | None:
    target = episode_data.get("target", {})
    if isinstance(target, dict):
        bbox = target.get("bbox")
        if isinstance(bbox, list):
            return bbox
    task_target = episode_data.get("task", {}).get("target", {})
    if isinstance(task_target, dict):
        bbox = task_target.get("bbox")
        if isinstance(bbox, list):
            return bbox
    return None


def _reference_optimal_path(episode_data: dict[str, Any]) -> float | None:
    direct = episode_data.get("optimal_path_length_meters")
    if isinstance(direct, (int, float)) and not isinstance(direct, bool) and direct >= 0:
        return float(direct)
    reference = episode_data.get("reference", {})
    if isinstance(reference, dict):
        nested = reference.get("optimal_path_length_meters")
        if isinstance(nested, (int, float)) and not isinstance(nested, bool) and nested >= 0:
            return float(nested)
    return None


def _trajectory_path_length(trajectory_data: dict[str, Any]) -> float | None:
    value = trajectory_data.get("path_length_meters")
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
        return float(value)
    return None


def _episode_task(episode_data: dict[str, Any]) -> dict[str, Any]:
    task = episode_data.get("task")
    if isinstance(task, dict):
        return task
    return {
        "task_type": episode_data.get("task_type", "visual_search"),
        "target": episode_data.get("target", {}),
        "required_actions": episode_data.get("required_actions", []),
        "allows_approximate_success": episode_data.get(
            "allows_approximate_success", False
        ),
    }


def _has_required_interactions(steps: list[dict[str, Any]], required_actions: list[str]) -> bool:
    if not required_actions:
        return False
    index = 0
    for step in steps:
        if index >= len(required_actions):
            break
        if _action_type(step) == required_actions[index] and _step_succeeded(step):
            index += 1
    return index == len(required_actions)


def _postcondition_passed(step: dict[str, Any]) -> bool:
    postcondition = step.get("postcondition")
    if isinstance(postcondition, dict):
        return postcondition.get("passed") is True
    execution = step.get("execution")
    if isinstance(execution, dict):
        postcondition = execution.get("postcondition")
        if isinstance(postcondition, dict):
            return postcondition.get("passed") is True
    return False


def _has_strict_interaction_evidence(
    steps: list[dict[str, Any]],
    required_actions: list[str],
) -> bool:
    if not required_actions:
        return False
    index = 0
    for step in steps:
        if index >= len(required_actions):
            break
        if (
            _action_type(step) == required_actions[index]
            and _step_succeeded(step)
            and _postcondition_passed(step)
        ):
            index += 1
    return index == len(required_actions)


def _exit_crossing_verified(trajectory_data: dict[str, Any]) -> bool:
    for key in ("exit", "door_crossing", "completion_evidence"):
        evidence = trajectory_data.get(key)
        if isinstance(evidence, dict) and (
            evidence.get("crossed_threshold") is True
            or evidence.get("crossed") is True
            or evidence.get("passed") is True
        ):
            return True
    for step in trajectory_data.get("steps", []):
        completion = step.get("completion_status")
        if isinstance(completion, dict):
            if completion.get("exit_verified") is True:
                return True
            evidence = completion.get("exit_evidence")
            if isinstance(evidence, dict) and (
                evidence.get("crossed_threshold") is True
                or evidence.get("crossed") is True
                or evidence.get("passed") is True
            ):
                return True
        for key in ("exit", "door_crossing"):
            evidence = step.get(key)
            if isinstance(evidence, dict) and (
                evidence.get("crossed_threshold") is True
                or evidence.get("crossed") is True
                or evidence.get("passed") is True
            ):
                return True
    return False


def _approximate_success_allowed(episode_data: dict[str, Any]) -> bool:
    task = _episode_task(episode_data)
    return bool(task.get("allows_approximate_success", False))


def _is_approximate_success(trajectory_data: dict[str, Any]) -> bool:
    if trajectory_data.get("approximate_success") is True:
        return True
    outcome = str(trajectory_data.get("outcome", "")).lower()
    if outcome in {"approximate_success", "approximate"}:
        return True
    for step in trajectory_data.get("steps", []):
        if step.get("approximate_success") is True:
            return True
        feedback = step.get("environment_feedback")
        if isinstance(feedback, dict) and feedback.get("approximate_success") is True:
            return True
        if str(step.get("outcome", "")).lower() in {"approximate_success", "approximate"}:
            return True
    return False


def evaluate_episode(
    episode_data: dict[str, Any],
    trajectory_data: dict[str, Any],
    config: AgentConfig,
) -> EpisodeMetrics:
    """Evaluate one episode against Plan2 metrics."""

    steps = list(trajectory_data.get("steps", []))
    step_count = len(steps)
    task = _episode_task(episode_data)
    task_type = str(task.get("task_type", "unknown"))
    required_actions = list(task.get("required_actions", []))

    path_length_meters = _trajectory_path_length(trajectory_data)
    optimal_path_length_meters = _reference_optimal_path(episode_data)
    spl_eligible = path_length_meters is not None and optimal_path_length_meters is not None
    path_length = path_length_meters if path_length_meters is not None else float(step_count)
    optimal_path_length = (
        optimal_path_length_meters if optimal_path_length_meters is not None else 0.0
    )

    final_iou = 0.0
    confidence_at_stop = 0.0
    strict_success = False
    interaction_success = False
    misstop = False
    approximate_success = _is_approximate_success(trajectory_data)

    if steps:
        last_step = steps[-1]
        action = _action_type(last_step)
        confidence = float(last_step.get("confidence", 0.0) or 0.0)
        done = bool(last_step.get("done", False))
        if action == "STOP":
            confidence_at_stop = confidence

        pred_bbox = _candidate_bbox(last_step)
        gt_bbox = _target_bbox(episode_data)
        final_iou = compute_iou(pred_bbox, gt_bbox) if pred_bbox and gt_bbox else 0.0
        min_confidence = float(config.raw["evaluation"]["min_success_confidence"])
        min_iou = float(config.raw["evaluation"]["min_success_iou"])

        if task_type == "interaction":
            interaction_success = _has_strict_interaction_evidence(
                steps,
                required_actions,
            )
            strict_success = interaction_success and done
        elif task_type == "navigation":
            target = task.get("target", {})
            requires_exit = (
                isinstance(target, dict)
                and str(target.get("object_type", "")).lower()
                in {"door", "doorway", "exit"}
            )
            if requires_exit:
                strict_success = (
                    done
                    and not trajectory_data.get("failed", False)
                    and _exit_crossing_verified(trajectory_data)
                )
            else:
                strict_success = done and not trajectory_data.get("failed", False)
        else:
            strict_success = action == "STOP" and done and confidence >= min_confidence
            if gt_bbox and final_iou < min_iou:
                strict_success = False
        misstop = action == "STOP" and not strict_success

    success = strict_success or (_approximate_success_allowed(episode_data) and approximate_success)
    collision_count = _collision_count(steps)
    illegal_actions = sum(
        1 for step in steps if _action_type(step) not in config.allowed_actions
    )

    return EpisodeMetrics(
        episode_id=str(episode_data.get("episode_id", "unknown")),
        success=success,
        path_length=path_length,
        optimal_path_length=optimal_path_length,
        final_iou=final_iou,
        confidence_at_stop=confidence_at_stop,
        illegal_actions=illegal_actions,
        planner_source_counts=_planner_source_counts(steps),
        execution_time_seconds=float(trajectory_data.get("execution_time", 0.0) or 0.0),
        spl_eligible=spl_eligible,
        category=str(episode_data.get("category", task_type or "unknown")),
        difficulty=str(episode_data.get("difficulty", "unknown")),
        task_type=task_type,
        group=str(episode_data.get("group", "unknown")),
        scene=str(episode_data.get("scene", "unknown")),
        split=str(episode_data.get("split", "unknown")),
        step_count=step_count,
        path_length_meters=path_length_meters,
        optimal_path_length_meters=optimal_path_length_meters,
        collision_count=collision_count,
        misstop=misstop,
        interaction_success=interaction_success,
        approximate_success=approximate_success,
        strict_success=strict_success,
    )


def _summary(metrics: list[EpisodeMetrics]) -> dict[str, float | int | None]:
    if not metrics:
        return {
            "episodes": 0,
            "success_rate": 0.0,
            "spl": None,
            "spl_coverage": 0.0,
            "average_path_length": 0.0,
            "collision_rate": 0.0,
            "misstop_rate": 0.0,
            "interaction_success_rate": 0.0,
            "approximate_success_rate": 0.0,
        }
    total = len(metrics)
    spl_eligible = [metric for metric in metrics if metric.spl_eligible]
    spl_scores = [
        compute_spl(metric.success, metric.path_length, metric.optimal_path_length)
        for metric in spl_eligible
    ]
    interaction = [metric for metric in metrics if metric.task_type == "interaction"]
    total_steps = sum(_metric_step_count(metric) for metric in metrics)
    return {
        "episodes": total,
        "success_rate": sum(metric.success for metric in metrics) / total,
        "spl": sum(spl_scores) / len(spl_scores) if spl_scores else None,
        "spl_coverage": len(spl_eligible) / total,
        "average_path_length": sum(metric.path_length for metric in metrics) / total,
        "collision_rate": (
            sum(metric.collision_count for metric in metrics) / total_steps
            if total_steps
            else 0.0
        ),
        "misstop_rate": sum(metric.misstop for metric in metrics) / total,
        "interaction_success_rate": (
            sum(metric.interaction_success for metric in interaction) / len(interaction)
            if interaction
            else 0.0
        ),
        "approximate_success_rate": sum(metric.approximate_success for metric in metrics)
        / total,
    }


def _group_by(
    episode_metrics: list[EpisodeMetrics], field: str
) -> dict[str, dict[str, float | int | None]]:
    grouped: dict[str, list[EpisodeMetrics]] = {}
    for metric in episode_metrics:
        grouped.setdefault(str(getattr(metric, field)), []).append(metric)
    return {key: _summary(values) for key, values in sorted(grouped.items())}


def _metric_step_count(metric: EpisodeMetrics) -> int:
    """Return action count without confusing meters with steps."""

    if metric.step_count > 0:
        return metric.step_count
    planner_steps = sum(metric.planner_source_counts.values())
    if planner_steps > 0:
        return planner_steps
    return int(metric.path_length) if metric.path_length > 0 else 0


def aggregate_metrics(episode_metrics: list[EpisodeMetrics]) -> DatasetMetrics:
    """Aggregate episode metrics into dataset-level metrics."""

    if not episode_metrics:
        return DatasetMetrics(
            total_episodes=0,
            success_rate=0.0,
            spl=None,
            spl_coverage=0.0,
            average_path_length=0.0,
            average_optimal_path_length=0.0,
            average_iou=0.0,
            illegal_action_rate=0.0,
            model_planner_usage_rate=0.0,
            average_confidence_at_success=0.0,
            per_category_success={},
            per_difficulty_success={},
            by_group={},
            by_task_type={},
            by_scene={},
            by_split={},
        )

    total = len(episode_metrics)
    successes = sum(metric.success for metric in episode_metrics)
    spl_eligible = [metric for metric in episode_metrics if metric.spl_eligible]
    spl_scores = [
        compute_spl(metric.success, metric.path_length, metric.optimal_path_length)
        for metric in spl_eligible
    ]
    avg_spl = sum(spl_scores) / len(spl_scores) if spl_scores else None
    avg_path = sum(metric.path_length for metric in episode_metrics) / total
    avg_optimal = (
        sum(metric.optimal_path_length for metric in spl_eligible) / len(spl_eligible)
        if spl_eligible
        else 0.0
    )
    avg_iou = sum(metric.final_iou for metric in episode_metrics) / total
    total_steps = sum(_metric_step_count(metric) for metric in episode_metrics)
    total_illegal = sum(metric.illegal_actions for metric in episode_metrics)
    total_collisions = sum(metric.collision_count for metric in episode_metrics)
    illegal_rate = total_illegal / total_steps if total_steps else 0.0
    collision_rate = total_collisions / total_steps if total_steps else 0.0
    model_steps = sum(
        metric.planner_source_counts.get("model_planner", 0)
        for metric in episode_metrics
    )
    model_usage_rate = model_steps / total_steps if total_steps else 0.0
    successful_confidences = [
        metric.confidence_at_stop
        for metric in episode_metrics
        if metric.success and metric.confidence_at_stop > 0
    ]
    avg_confidence = (
        sum(successful_confidences) / len(successful_confidences)
        if successful_confidences
        else 0.0
    )

    category_groups: dict[str, list[bool]] = {}
    difficulty_groups: dict[str, list[bool]] = {}
    for metric in episode_metrics:
        category_groups.setdefault(metric.category, []).append(metric.success)
        difficulty_groups.setdefault(metric.difficulty, []).append(metric.success)
    interaction = [
        metric for metric in episode_metrics if metric.task_type == "interaction"
    ]

    return DatasetMetrics(
        total_episodes=total,
        success_rate=successes / total,
        spl=avg_spl,
        spl_coverage=len(spl_eligible) / total,
        average_path_length=avg_path,
        average_optimal_path_length=avg_optimal,
        average_iou=avg_iou,
        illegal_action_rate=illegal_rate,
        model_planner_usage_rate=model_usage_rate,
        average_confidence_at_success=avg_confidence,
        per_category_success={
            name: sum(values) / len(values) for name, values in category_groups.items()
        },
        per_difficulty_success={
            name: sum(values) / len(values) for name, values in difficulty_groups.items()
        },
        collision_count=total_collisions,
        collision_rate=collision_rate,
        misstop_count=sum(metric.misstop for metric in episode_metrics),
        misstop_rate=sum(metric.misstop for metric in episode_metrics) / total,
        interaction_success_rate=(
            sum(metric.interaction_success for metric in interaction) / len(interaction)
            if interaction
            else 0.0
        ),
        approximate_success_rate=sum(
            metric.approximate_success for metric in episode_metrics
        )
        / total,
        strict_success_rate=sum(metric.strict_success for metric in episode_metrics)
        / total,
        by_group=_group_by(episode_metrics, "group"),
        by_task_type=_group_by(episode_metrics, "task_type"),
        by_scene=_group_by(episode_metrics, "scene"),
        by_split=_group_by(episode_metrics, "split"),
    )


def dataset_metrics_to_dict(metrics: DatasetMetrics) -> dict[str, Any]:
    """Serialize dataset metrics to a stable JSON-compatible dictionary."""

    return {
        "total_episodes": metrics.total_episodes,
        "success_rate": metrics.success_rate,
        "spl": metrics.spl,
        "spl_coverage": metrics.spl_coverage,
        "average_path_length": metrics.average_path_length,
        "average_optimal_path_length": metrics.average_optimal_path_length,
        "average_iou": metrics.average_iou,
        "illegal_action_rate": metrics.illegal_action_rate,
        "model_planner_usage_rate": metrics.model_planner_usage_rate,
        "average_confidence_at_success": metrics.average_confidence_at_success,
        "collision_count": metrics.collision_count,
        "collision_rate": metrics.collision_rate,
        "misstop_count": metrics.misstop_count,
        "misstop_rate": metrics.misstop_rate,
        "interaction_success_rate": metrics.interaction_success_rate,
        "approximate_success_rate": metrics.approximate_success_rate,
        "strict_success_rate": metrics.strict_success_rate,
        "per_category_success": metrics.per_category_success,
        "per_difficulty_success": metrics.per_difficulty_success,
        "by_group": metrics.by_group or {},
        "by_task_type": metrics.by_task_type or {},
        "by_scene": metrics.by_scene or {},
        "by_split": metrics.by_split or {},
    }


def print_metrics(metrics: DatasetMetrics) -> None:
    """Print metrics in readable format."""

    print("=" * 60)
    print("Evaluation Metrics")
    print("=" * 60)
    print(f"Total Episodes: {metrics.total_episodes}")
    print(f"Success Rate (SR): {metrics.success_rate:.1%}")
    if metrics.spl is None:
        print("Success weighted by Path Length (SPL): unavailable")
    else:
        print(f"Success weighted by Path Length (SPL): {metrics.spl:.3f}")
    print(f"SPL Coverage: {metrics.spl_coverage:.1%}")
    print(f"Average Path Length: {metrics.average_path_length:.1f}")
    print(f"Average IoU: {metrics.average_iou:.3f}")
    print(f"Collision Rate: {metrics.collision_rate:.1%}")
    print(f"Misstop Rate: {metrics.misstop_rate:.1%}")
    print(f"Interaction Success Rate: {metrics.interaction_success_rate:.1%}")
    print(f"Approximate Success Rate: {metrics.approximate_success_rate:.1%}")
    print("=" * 60)
