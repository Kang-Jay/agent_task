"""Tests for Phase 5 evaluation metrics.

According to Plan_1_agent_demo_repair.md Phase 5 requirements.
"""
from __future__ import annotations

import unittest

from src.evaluation.metrics import (
    compute_iou,
    compute_spl,
    evaluate_episode,
    aggregate_metrics,
    EpisodeMetrics
)
from src.task.config import load_config


class MetricsTests(unittest.TestCase):
    """Test evaluation metrics implementation."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config()

    def test_iou_perfect_match(self) -> None:
        """Test IoU with identical boxes."""
        bbox1 = [10, 10, 50, 50]
        bbox2 = [10, 10, 50, 50]
        iou = compute_iou(bbox1, bbox2)
        self.assertAlmostEqual(iou, 1.0)

    def test_iou_no_overlap(self) -> None:
        """Test IoU with no overlap."""
        bbox1 = [10, 10, 30, 30]
        bbox2 = [50, 50, 70, 70]
        iou = compute_iou(bbox1, bbox2)
        self.assertEqual(iou, 0.0)

    def test_iou_partial_overlap(self) -> None:
        """Test IoU with partial overlap."""
        bbox1 = [10, 10, 50, 50]
        bbox2 = [30, 30, 70, 70]
        # Intersection: 20x20 = 400
        # Union: 1600 + 1600 - 400 = 2800
        # IoU = 400/2800 = 0.142857
        iou = compute_iou(bbox1, bbox2)
        self.assertAlmostEqual(iou, 0.142857, places=5)

    def test_iou_with_none(self) -> None:
        """Test IoU handles None gracefully."""
        iou = compute_iou(None, [10, 10, 50, 50])
        self.assertEqual(iou, 0.0)

        iou = compute_iou([10, 10, 50, 50], None)
        self.assertEqual(iou, 0.0)

    def test_spl_perfect_path(self) -> None:
        """Test SPL with optimal path."""
        spl = compute_spl(success=True, path_length=5, optimal_path_length=5)
        self.assertEqual(spl, 1.0)

    def test_spl_longer_path(self) -> None:
        """Test SPL with suboptimal path."""
        spl = compute_spl(success=True, path_length=10, optimal_path_length=5)
        self.assertEqual(spl, 0.5)

    def test_spl_failure(self) -> None:
        """Test SPL for failed episode."""
        spl = compute_spl(success=False, path_length=10, optimal_path_length=5)
        self.assertEqual(spl, 0.0)

    def test_evaluate_episode_success(self) -> None:
        """Test episode evaluation with successful trajectory."""
        episode_data = {
            "episode_id": "test_success",
            "instruction": "Find red cup",
            "target": {"bbox": [100, 100, 150, 150]},
            "steps": [{"action": "STOP"}],
            "optimal_path_length_meters": 1.0,
        }

        trajectory_data = {
            "steps": [
                {
                    "action": {"type": "TURN_RIGHT"},
                    "done": False,
                    "confidence": 0.4,
                    "planner_source": "model_planner",
                    "observation": {"best_candidate": None}
                },
                {
                    "action": {"type": "STOP"},
                    "done": True,
                    "confidence": 0.85,
                    "planner_source": "model_planner",
                    "observation": {
                        "best_candidate": {
                            "bbox": [95, 95, 155, 155]  # Good overlap with target
                        }
                    }
                }
            ],
            "execution_time": 2.5,
            "path_length_meters": 2.0,
        }

        metrics = evaluate_episode(episode_data, trajectory_data, self.config)

        self.assertTrue(metrics.success)
        self.assertEqual(metrics.path_length, 2.0)
        self.assertEqual(metrics.optimal_path_length, 1.0)
        self.assertTrue(metrics.spl_eligible)
        self.assertGreater(metrics.final_iou, 0.5)
        self.assertEqual(metrics.confidence_at_stop, 0.85)
        self.assertEqual(metrics.illegal_actions, 0)
        self.assertTrue(metrics.spl_eligible)

    def test_evaluate_episode_failure_low_confidence(self) -> None:
        """Test episode evaluation with low confidence stop."""
        episode_data = {
            "episode_id": "test_low_conf",
            "target": {"bbox": [100, 100, 150, 150]},
            "steps": [{"action": "STOP"}]
        }

        trajectory_data = {
            "steps": [
                {
                    "action": {"type": "STOP"},
                    "done": True,
                    "confidence": 0.50,  # Below threshold (0.78)
                    "planner_source": "rule_fallback",
                    "observation": {"best_candidate": {"bbox": [100, 100, 150, 150]}}
                }
            ]
        }

        metrics = evaluate_episode(episode_data, trajectory_data, self.config)

        # Should fail due to low confidence
        self.assertFalse(metrics.success)

    def test_evaluate_episode_failure_poor_localization(self) -> None:
        """Test episode fails with poor bbox match."""
        episode_data = {
            "episode_id": "test_poor_loc",
            "target": {"bbox": [100, 100, 150, 150]},
            "steps": [{"action": "STOP"}]
        }

        trajectory_data = {
            "steps": [
                {
                    "action": {"type": "STOP"},
                    "done": True,
                    "confidence": 0.85,
                    "planner_source": "model_planner",
                    "observation": {
                        "best_candidate": {"bbox": [300, 300, 350, 350]}  # Wrong location
                    }
                }
            ]
        }

        metrics = evaluate_episode(episode_data, trajectory_data, self.config)

        # Should fail due to IoU < 0.3
        self.assertFalse(metrics.success)
        self.assertLess(metrics.final_iou, 0.3)

    def test_aggregate_metrics(self) -> None:
        """Test aggregation of episode metrics."""
        episodes = [
            EpisodeMetrics(
                episode_id="ep1",
                success=True,
                path_length=5,
                optimal_path_length=3,
                final_iou=0.8,
                confidence_at_stop=0.85,
                illegal_actions=0,
                planner_source_counts={"model_planner": 5},
                execution_time_seconds=1.0,
                spl_eligible=True,
                category="positive",
                difficulty="easy",
            ),
            EpisodeMetrics(
                episode_id="ep2",
                success=False,
                path_length=10,
                optimal_path_length=4,
                final_iou=0.2,
                confidence_at_stop=0.0,
                illegal_actions=1,
                planner_source_counts={"rule_fallback": 10},
                execution_time_seconds=2.0,
                spl_eligible=True,
                category="negative",
                difficulty="hard",
            ),
            EpisodeMetrics(
                episode_id="ep3",
                success=True,
                path_length=4,
                optimal_path_length=4,
                final_iou=0.9,
                confidence_at_stop=0.92,
                illegal_actions=0,
                planner_source_counts={"model_planner": 4},
                execution_time_seconds=0.8,
                spl_eligible=True,
                category="positive",
                difficulty="easy",
            )
        ]

        metrics = aggregate_metrics(episodes)

        self.assertEqual(metrics.total_episodes, 3)
        self.assertAlmostEqual(metrics.success_rate, 2/3)
        self.assertGreater(metrics.spl, 0.0)
        self.assertAlmostEqual(metrics.average_path_length, (5+10+4)/3)
        self.assertAlmostEqual(metrics.average_iou, (0.8+0.2+0.9)/3)
        self.assertGreater(metrics.model_planner_usage_rate, 0.4)
        self.assertAlmostEqual(metrics.average_confidence_at_success, (0.85+0.92)/2)
        self.assertEqual(metrics.spl_coverage, 1.0)
        self.assertAlmostEqual(metrics.per_category_success["positive"], 1.0)

    def test_aggregate_empty_list(self) -> None:
        """Test aggregation handles empty list."""
        metrics = aggregate_metrics([])

        self.assertEqual(metrics.total_episodes, 0)
        self.assertEqual(metrics.success_rate, 0.0)
        self.assertIsNone(metrics.spl)
        self.assertEqual(metrics.spl_coverage, 0.0)

    def test_interaction_requires_passed_postconditions(self) -> None:
        episode_data = {
            "episode_id": "put_wrong_receptacle",
            "task": {
                "task_type": "interaction",
                "target": {
                    "object_type": "Vase",
                    "source_object_type": "Vase",
                    "destination_object_type": "Box",
                },
                "required_actions": ["PickupObject", "PutObject"],
                "allows_approximate_success": False,
            },
        }
        trajectory_data = {
            "steps": [
                {
                    "action": {"type": "PickupObject"},
                    "done": False,
                    "success": True,
                    "postcondition": {"passed": True},
                },
                {
                    "action": {"type": "PutObject"},
                    "done": True,
                    "success": True,
                    "postcondition": {
                        "passed": False,
                        "evidence": {"receptacleObjectId": "WrongBox|1"},
                    },
                },
            ]
        }

        metrics = evaluate_episode(episode_data, trajectory_data, self.config)

        self.assertFalse(metrics.success)
        self.assertFalse(metrics.strict_success)
        self.assertFalse(metrics.interaction_success)

    def test_exit_navigation_requires_crossing_evidence(self) -> None:
        episode_data = {
            "episode_id": "right_door_exit",
            "task": {
                "task_type": "navigation",
                "target": {"object_type": "Door"},
                "required_actions": [],
                "allows_approximate_success": False,
            },
        }
        trajectory_data = {
            "steps": [
                {
                    "action": {"type": "Done"},
                    "done": True,
                    "success": True,
                }
            ]
        }

        metrics = evaluate_episode(episode_data, trajectory_data, self.config)
        self.assertFalse(metrics.success)

        trajectory_data["door_crossing"] = {"crossed_threshold": True}
        metrics = evaluate_episode(episode_data, trajectory_data, self.config)
        self.assertTrue(metrics.success)


if __name__ == "__main__":
    unittest.main()
