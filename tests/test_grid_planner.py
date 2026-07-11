import unittest

from src.exploration.frontier_policy import rank_frontiers
from src.mapping.frontier import Frontier
from src.mapping.occupancy_grid import GridSpec, OccupancyGrid
from src.planning.grid_planner import GridPlanner, GridPose


class GridPlannerTests(unittest.TestCase):
    def test_path_avoids_occupied_cells(self):
        grid = OccupancyGrid(GridSpec(1.0, 5, 5, 0.0, 0.0))
        free_cells = [(2, 0), (2, 1), (1, 1), (1, 2), (1, 3), (2, 3), (2, 4)]
        grid.apply_updates(free_cells=free_cells, occupied_cells=[(2, 2)])
        planner = GridPlanner(grid)

        path = planner.plan_path((2, 0), (2, 4))

        self.assertTrue(path.ok)
        self.assertNotIn((2, 2), path.path)
        self.assertEqual(path.path[0], (2, 0))
        self.assertEqual(path.path[-1], (2, 4))

    def test_no_path_when_goal_is_sealed(self):
        grid = OccupancyGrid(GridSpec(1.0, 3, 3, 0.0, 0.0))
        grid.apply_updates(
            free_cells=[(0, 0), (1, 1)],
            occupied_cells=[(0, 1), (1, 0), (1, 2), (2, 1)],
        )
        planner = GridPlanner(grid)

        path = planner.plan_path((0, 0), (1, 1))

        self.assertEqual(path.status, "no_path")

    def test_next_action_turns_right_for_positive_ai2thor_yaw(self):
        grid = OccupancyGrid(GridSpec(1.0, 3, 3, 0.0, 0.0))
        grid.apply_updates(free_cells=[(1, 1), (1, 2)])
        planner = GridPlanner(grid)
        path = planner.plan_path((1, 1), (1, 2))

        action = planner.next_navigation_action(GridPose(1, 1, heading_degrees=0.0), path)

        self.assertEqual(action.type, "TURN_RIGHT")

    def test_next_action_moves_forward_when_heading_matches(self):
        grid = OccupancyGrid(GridSpec(1.0, 3, 3, 0.0, 0.0))
        grid.apply_updates(free_cells=[(1, 1), (2, 1)])
        planner = GridPlanner(grid)
        path = planner.plan_path((1, 1), (2, 1))

        action = planner.next_navigation_action(GridPose(1, 1, heading_degrees=0.0), path)

        self.assertEqual(action.type, "MOVE_FORWARD")

    def test_plan_to_ranked_frontier_uses_reachable_rank_order(self):
        grid = OccupancyGrid(GridSpec(1.0, 5, 5, 0.0, 0.0))
        grid.apply_updates(free_cells=[(2, 1), (2, 2), (2, 3)])
        frontiers = [
            Frontier(row=2, col=1, unknown_neighbors=1, free_neighbors=1, distance_cells=1),
            Frontier(row=2, col=3, unknown_neighbors=3, free_neighbors=1, distance_cells=3),
        ]
        planner = GridPlanner(grid)

        result = planner.plan_to_frontier(
            (2, 2),
            frontiers,
            ranked_frontiers=rank_frontiers(frontiers, distance_weight=0.0),
        )

        self.assertEqual(result.goal, (2, 3))


if __name__ == "__main__":
    unittest.main()
