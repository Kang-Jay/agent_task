import unittest

from src.mapping.observations import CameraIntrinsics, CameraPose, RGBDObservation
from src.mapping.occupancy_grid import GridSpec, OccupancyGrid
from src.planning.exploration_planner import (
    ExplorationPlanner,
    ExplorationPlannerInput,
)
from src.planning.grid_planner import GridPose
from src.exploration.frontier_policy import OracleLeakageError


class ExplorationPlannerTests(unittest.TestCase):
    def test_decision_updates_rgbd_map_and_returns_frontier_action(self):
        grid = OccupancyGrid(GridSpec(1.0, 9, 9, -4.0, -4.0))
        planner = ExplorationPlanner(grid, depth_sample_stride=1)
        observation = RGBDObservation(
            depth_meters=[[3.0]],
            intrinsics=CameraIntrinsics(1, 1, fx=1.0, fy=1.0, cx=0.0, cy=0.0),
            camera_pose=CameraPose(x=0.0, y=1.0, z=0.0, yaw_degrees=0.0),
        )

        decision = planner.decide(
            ExplorationPlannerInput(
                observation=observation,
                pose=GridPose(row=4, col=4, heading_degrees=0.0),
                target_terms=("television",),
            )
        )

        self.assertEqual(decision.planner_source, "non_oracle_frontier")
        self.assertGreaterEqual(decision.map_update.projected_depth_pixels, 1)
        self.assertGreater(len(decision.frontiers), 0)
        self.assertIn(decision.action.type, {"MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "INSPECT"})
        self.assertIsNotNone(decision.selected_frontier)

    def test_semantic_candidate_can_prioritize_farther_frontier(self):
        grid = OccupancyGrid(GridSpec(1.0, 8, 8, 0.0, 0.0))
        grid.apply_updates(
            free_cells=[(2, 2), (2, 3), (2, 4), (2, 5), (3, 5), (4, 5), (5, 5)]
        )
        planner = ExplorationPlanner(grid)

        decision = planner.decide(
            ExplorationPlannerInput(
                observation=None,
                pose=GridPose(row=2, col=2, heading_degrees=90.0),
                target_terms=("vase",),
                rgb_candidates=(
                    {
                        "label": "vase",
                        "confidence": 0.95,
                        "world_x": 5.2,
                        "world_z": 5.2,
                    },
                ),
            )
        )

        self.assertEqual(decision.selected_frontier.cell, (5, 5))
        self.assertGreater(decision.ranked_frontiers[0].semantic_score, 0.0)

    def test_failed_forward_cell_is_blocked_before_planning(self):
        grid = OccupancyGrid(GridSpec(1.0, 5, 5, 0.0, 0.0))
        grid.apply_updates(free_cells=[(2, 2), (2, 3)])
        planner = ExplorationPlanner(grid)

        decision = planner.decide(
            ExplorationPlannerInput(
                observation=None,
                pose=GridPose(row=2, col=2, heading_degrees=90.0),
                failed_forward_cell=(2, 3),
            )
        )

        self.assertFalse(grid.is_traversable(2, 3))
        self.assertNotEqual(decision.path.goal, (2, 3))

    def test_rejects_oracle_context(self):
        cases = [
            {"objects": [{"objectId": "Door|1", "distance": 1.0}]},
            {"target_pose": {"x": 1.0, "z": 2.0}},
            {"matched_pose": {"x": 1.0, "z": 2.0}},
            {"recommended_action": "TeleportFull"},
        ]
        for context in cases:
            with self.subTest(context=context):
                with self.assertRaises(OracleLeakageError):
                    ExplorationPlanner.validate_planner_context(context)


if __name__ == "__main__":
    unittest.main()
