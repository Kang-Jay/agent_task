"""Enhanced evaluation metrics for Phase 5.

According to Plan_1_agent_demo_repair.md Phase 5 requirements.

Implements standard embodied AI metrics:
- Success Rate (SR)
- Success weighted by Path Length (SPL)
- Intersection over Union (IoU) for localization
- Navigation Efficiency
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.task.config import AgentConfig


@dataclass
class EpisodeMetrics:
    """Metrics for a single episode."""
    episode_id: str
    success: bool
    path_length: int
    optimal_path_length: int
    final_iou: float
    confidence_at_stop: float
    illegal_actions: int
    planner_source_counts: dict[str, int]
    execution_time_seconds: float
    spl_eligible: bool = False
    category: str = "unknown"
    difficulty: str = "unknown"


@dataclass
class DatasetMetrics:
    """Aggregated metrics for entire dataset."""
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


def compute_iou(pred_bbox: list[int], gt_bbox: list[int]) -> float:
    """Compute Intersection over Union for bounding boxes.

    Args:
        pred_bbox: [x1, y1, x2, y2] predicted box
        gt_bbox: [x1, y1, x2, y2] ground truth box

    Returns:
        IoU score between 0 and 1
    """
    if not pred_bbox or not gt_bbox:
        return 0.0

    # Compute intersection
    x1_inter = max(pred_bbox[0], gt_bbox[0])
    y1_inter = max(pred_bbox[1], gt_bbox[1])
    x2_inter = min(pred_bbox[2], gt_bbox[2])
    y2_inter = min(pred_bbox[3], gt_bbox[3])

    if x2_inter < x1_inter or y2_inter < y1_inter:
        return 0.0

    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)

    # Compute union
    pred_area = (pred_bbox[2] - pred_bbox[0]) * (pred_bbox[3] - pred_bbox[1])
    gt_area = (gt_bbox[2] - gt_bbox[0]) * (gt_bbox[3] - gt_bbox[1])
    union_area = pred_area + gt_area - inter_area

    if union_area == 0:
        return 0.0

    return inter_area / union_area


def compute_spl(success: bool, path_length: float, optimal_path_length: float) -> float:
    """Compute Success weighted by Path Length (SPL).

    SPL = success * (optimal_length / max(optimal_length, actual_length))

    Args:
        success: Whether episode succeeded
        path_length: Actual number of steps taken
        optimal_path_length: Shortest possible path length

    Returns:
        SPL score between 0 and 1
    """
    if not success:
        return 0.0

    if optimal_path_length == 0:
        return 1.0 if path_length == 0 else 0.0

    return optimal_path_length / max(optimal_path_length, path_length)


def evaluate_episode(episode_data: dict[str, Any], trajectory_data: dict[str, Any], config: AgentConfig) -> EpisodeMetrics:
    """Evaluate a single episode.

    Args:
        episode_data: Episode annotation from dataset
        trajectory_data: Agent execution trajectory
        config: Agent configuration

    Returns:
        Episode metrics
    """
    steps = trajectory_data.get("steps", [])
    path_length_meters = trajectory_data.get("path_length_meters")
    optimal_path_length_meters = episode_data.get("optimal_path_length_meters")
    spl_eligible = (
        isinstance(path_length_meters, (int, float))
        and isinstance(optimal_path_length_meters, (int, float))
        and float(path_length_meters) >= 0.0
        and float(optimal_path_length_meters) >= 0.0
    )
    path_length = float(path_length_meters) if spl_eligible else float(len(steps))
    optimal_path_length = (
        float(optimal_path_length_meters) if spl_eligible else 0.0
    )

    # Determine success
    if not steps:
        success = False
        final_iou = 0.0
        confidence_at_stop = 0.0
    else:
        last_step = steps[-1]
        action = last_step.get("action", {}).get("type")
        done = last_step.get("done", False)
        confidence = last_step.get("confidence", 0.0)

        # Success requires:
        # 1. STOP action
        # 2. High confidence (>= min_success_confidence)
        # 3. Target correctly localized (if bbox available)
        success = (
            action == "STOP" and
            done and
            confidence >= config.raw["evaluation"]["min_success_confidence"]
        )

        confidence_at_stop = confidence if action == "STOP" else 0.0

        # Compute IoU if bounding boxes available
        pred_bbox = last_step.get("observation", {}).get("best_candidate", {})
        if pred_bbox and isinstance(pred_bbox, dict):
            pred_bbox = pred_bbox.get("bbox")

        gt_bbox = episode_data.get("target", {}).get("bbox")
        final_iou = compute_iou(pred_bbox, gt_bbox) if pred_bbox and gt_bbox else 0.0

        # Check for bbox match if success claimed
        min_iou = float(config.raw["evaluation"]["min_success_iou"])
        if success and gt_bbox and final_iou < min_iou:
            success = False  # Localization too poor

    # Count illegal actions
    illegal_actions = sum(
        1 for step in steps
        if step.get("action", {}).get("type") not in config.allowed_actions
    )

    # Count planner sources
    planner_source_counts: dict[str, int] = {}
    for step in steps:
        source = step.get("planner_source", "unknown")
        planner_source_counts[source] = planner_source_counts.get(source, 0) + 1

    return EpisodeMetrics(
        episode_id=episode_data.get("episode_id", "unknown"),
        success=success,
        path_length=path_length,
        optimal_path_length=optimal_path_length,
        final_iou=final_iou,
        confidence_at_stop=confidence_at_stop,
        illegal_actions=illegal_actions,
        planner_source_counts=planner_source_counts,
        execution_time_seconds=trajectory_data.get("execution_time", 0.0),
        spl_eligible=spl_eligible,
        category=episode_data.get("category", "unknown"),
        difficulty=episode_data.get("difficulty", "unknown"),
    )


def aggregate_metrics(episode_metrics: list[EpisodeMetrics]) -> DatasetMetrics:
    """Aggregate episode metrics into dataset-level metrics.

    Args:
        episode_metrics: List of per-episode metrics

    Returns:
        Aggregated dataset metrics
    """
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
            per_difficulty_success={}
        )

    total = len(episode_metrics)
    successes = sum(1 for m in episode_metrics if m.success)

    # Compute SPL
    spl_eligible = [metric for metric in episode_metrics if metric.spl_eligible]
    spl_scores = [
        compute_spl(m.success, m.path_length, m.optimal_path_length)
        for m in spl_eligible
    ]
    avg_spl = sum(spl_scores) / len(spl_scores) if spl_scores else None

    # Average path lengths
    avg_path = sum(m.path_length for m in episode_metrics) / total
    avg_optimal = (
        sum(m.optimal_path_length for m in spl_eligible) / len(spl_eligible)
        if spl_eligible else 0.0
    )

    # Average IoU
    avg_iou = sum(m.final_iou for m in episode_metrics) / total

    # Illegal action rate
    total_actions = sum(m.path_length for m in episode_metrics)
    total_illegal = sum(m.illegal_actions for m in episode_metrics)
    illegal_rate = total_illegal / total_actions if total_actions > 0 else 0.0

    # Model planner usage
    total_steps = sum(m.path_length for m in episode_metrics)
    model_steps = sum(
        m.planner_source_counts.get("model_planner", 0)
        for m in episode_metrics
    )
    model_usage_rate = model_steps / total_steps if total_steps > 0 else 0.0

    # Average confidence at success
    successful_confidences = [
        m.confidence_at_stop for m in episode_metrics
        if m.success and m.confidence_at_stop > 0
    ]
    avg_confidence = (
        sum(successful_confidences) / len(successful_confidences)
        if successful_confidences else 0.0
    )

    category_groups: dict[str, list[bool]] = {}
    difficulty_groups: dict[str, list[bool]] = {}
    for metric in episode_metrics:
        category_groups.setdefault(metric.category, []).append(metric.success)
        difficulty_groups.setdefault(metric.difficulty, []).append(metric.success)

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
            name: sum(values) / len(values)
            for name, values in category_groups.items()
        },
        per_difficulty_success={
            name: sum(values) / len(values)
            for name, values in difficulty_groups.items()
        },
    )


def print_metrics(metrics: DatasetMetrics) -> None:
    """Print metrics in readable format."""
    print("=" * 60)
    print("Evaluation Metrics")
    print("=" * 60)
    print(f"Total Episodes: {metrics.total_episodes}")
    print(f"\nSuccess Rate (SR): {metrics.success_rate:.1%}")
    if metrics.spl is None:
        print("Success weighted by Path Length (SPL): unavailable (missing geodesic distances)")
    else:
        print(f"Success weighted by Path Length (SPL): {metrics.spl:.3f}")
    print(f"SPL Coverage: {metrics.spl_coverage:.1%}")
    print(f"\nAverage Path Length: {metrics.average_path_length:.1f}")
    print(f"Average Optimal Path Length: {metrics.average_optimal_path_length:.1f}")
    print(f"Navigation Efficiency: {metrics.average_optimal_path_length / metrics.average_path_length:.1%}" if metrics.average_path_length > 0 else "N/A")
    print(f"\nAverage IoU (Localization): {metrics.average_iou:.3f}")
    print(f"Average Confidence at Success: {metrics.average_confidence_at_success:.3f}")
    print(f"\nIllegal Action Rate: {metrics.illegal_action_rate:.1%}")
    print(f"Model Planner Usage Rate: {metrics.model_planner_usage_rate:.1%}")
    print("=" * 60)
