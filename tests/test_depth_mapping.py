import unittest

from src.mapping.depth_projector import project_depth_to_grid
from src.mapping.observations import CameraIntrinsics, CameraPose, RGBDObservation
from src.mapping.occupancy_grid import GridCellState, GridSpec, OccupancyGrid


class DepthProjectorTests(unittest.TestCase):
    def test_depth_ray_marks_free_and_terminal_obstacle(self):
        grid = OccupancyGrid(GridSpec(1.0, 7, 7, -3.0, -3.0))
        observation = RGBDObservation(
            depth_meters=[[2.0]],
            intrinsics=CameraIntrinsics(1, 1, fx=1.0, fy=1.0, cx=0.0, cy=0.0),
            camera_pose=CameraPose(x=0.0, y=1.0, z=0.0, yaw_degrees=0.0),
        )

        result = project_depth_to_grid(grid, observation, max_depth_m=5.0, sample_stride=1)

        self.assertEqual(result.projected_depth_pixels, 1)
        self.assertEqual(grid.get(3, 3), GridCellState.FREE)
        self.assertEqual(grid.get(5, 3), GridCellState.OCCUPIED)

    def test_invalid_depth_is_ignored(self):
        grid = OccupancyGrid(GridSpec(1.0, 5, 5, -2.0, -2.0))
        observation = RGBDObservation(
            depth_meters=[[0.0, float("nan")]],
            intrinsics=CameraIntrinsics(2, 1, fx=1.0, fy=1.0, cx=0.5, cy=0.0),
            camera_pose=CameraPose(x=0.0, y=1.0, z=0.0, yaw_degrees=0.0),
        )

        result = project_depth_to_grid(grid, observation, sample_stride=1)

        self.assertEqual(result.projected_depth_pixels, 0)
        self.assertEqual(result.ignored_depth_pixels, 2)
        self.assertEqual(grid.summary()["occupied_cells"], 0)

    def test_yaw_rotates_projected_terminal_cell(self):
        grid = OccupancyGrid(GridSpec(1.0, 7, 7, -3.0, -3.0))
        observation = RGBDObservation(
            depth_meters=[[2.0]],
            intrinsics=CameraIntrinsics(1, 1, fx=1.0, fy=1.0, cx=0.0, cy=0.0),
            camera_pose=CameraPose(x=0.0, y=1.0, z=0.0, yaw_degrees=90.0),
        )

        project_depth_to_grid(grid, observation, max_depth_m=5.0, sample_stride=1)

        self.assertEqual(grid.get(3, 5), GridCellState.OCCUPIED)


if __name__ == "__main__":
    unittest.main()
