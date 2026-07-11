import unittest

from src.mapping.occupancy_grid import GridCellState, GridSpec, OccupancyGrid


class OccupancyGridTests(unittest.TestCase):
    def test_world_grid_roundtrip_uses_cell_centers(self):
        grid = OccupancyGrid(GridSpec(0.5, 10, 8, origin_x=-1.0, origin_z=-2.0))

        cell = grid.world_to_grid(-0.76, -1.76)

        self.assertEqual(cell, (0, 0))
        self.assertEqual(grid.grid_to_world(*cell), (-0.75, -1.75))

    def test_updates_mark_free_occupied_visited_and_blocked(self):
        grid = OccupancyGrid(GridSpec(1.0, 4, 4, 0.0, 0.0))

        result = grid.apply_updates(
            free_cells=[(1, 1), (1, 2)],
            occupied_cells=[(2, 2)],
            visited_cells=[(1, 1)],
            blocked_cells=[(3, 3)],
        )

        self.assertEqual(result.free_cells, 2)
        self.assertEqual(result.occupied_cells, 1)
        self.assertEqual(result.visited_cells, 1)
        self.assertEqual(result.blocked_cells, 1)
        self.assertEqual(grid.get(1, 1), GridCellState.FREE)
        self.assertEqual(grid.get(2, 2), GridCellState.OCCUPIED)
        self.assertTrue(grid.visited[1][1])
        self.assertFalse(grid.is_traversable(3, 3))

    def test_mark_free_does_not_clear_blocked_cell(self):
        grid = OccupancyGrid(GridSpec(1.0, 3, 3, 0.0, 0.0))
        grid.mark_blocked(1, 1)

        changed = grid.mark_free(1, 1)

        self.assertFalse(changed)
        self.assertEqual(grid.get(1, 1), GridCellState.OCCUPIED)


if __name__ == "__main__":
    unittest.main()
