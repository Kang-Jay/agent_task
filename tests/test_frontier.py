import unittest

from src.mapping.frontier import extract_frontiers
from src.mapping.occupancy_grid import GridSpec, OccupancyGrid


class FrontierTests(unittest.TestCase):
    def test_frontiers_are_free_cells_adjacent_to_unknown(self):
        grid = OccupancyGrid(GridSpec(1.0, 5, 5, 0.0, 0.0))
        grid.apply_updates(free_cells=[(2, 2), (2, 3)], occupied_cells=[(1, 2)])

        frontiers = extract_frontiers(grid, start=(2, 2))

        self.assertIn((2, 2), [frontier.cell for frontier in frontiers])
        self.assertIn((2, 3), [frontier.cell for frontier in frontiers])
        self.assertNotIn((1, 2), [frontier.cell for frontier in frontiers])

    def test_reachable_filter_removes_unreachable_frontiers(self):
        grid = OccupancyGrid(GridSpec(1.0, 5, 5, 0.0, 0.0))
        grid.apply_updates(free_cells=[(2, 2), (2, 3)])

        frontiers = extract_frontiers(
            grid,
            start=(2, 2),
            reachable_cells={(2, 2)},
        )

        self.assertEqual([frontier.cell for frontier in frontiers], [(2, 2)])


if __name__ == "__main__":
    unittest.main()
