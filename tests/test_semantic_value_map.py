import unittest

from src.exploration.frontier_policy import (
    OracleLeakageError,
    SemanticValueMap,
    rank_frontiers,
)
from src.mapping.frontier import Frontier
from src.mapping.occupancy_grid import GridSpec, OccupancyGrid


class SemanticValueMapTests(unittest.TestCase):
    def test_updates_target_matching_candidates_without_oracle_fields(self):
        grid = OccupancyGrid(GridSpec(1.0, 6, 6, 0.0, 0.0))
        value_map = SemanticValueMap(grid)

        updated = value_map.update_from_candidates(
            [{"label": "sofa", "confidence": 0.8, "world_x": 2.2, "world_z": 3.1}],
            target_terms=["sofa"],
        )

        self.assertEqual(updated, 1)
        self.assertGreater(value_map.value_at(3, 2), 0.7)

    def test_rejects_oracle_object_metadata(self):
        grid = OccupancyGrid(GridSpec(1.0, 3, 3, 0.0, 0.0))
        value_map = SemanticValueMap(grid)

        with self.assertRaises(OracleLeakageError):
            value_map.update_from_candidates(
                [{"label": "vase", "confidence": 0.9, "objectId": "Vase|1"}],
                target_terms=["vase"],
            )

    def test_rejects_underscore_oracle_keys_after_normalization(self):
        grid = OccupancyGrid(GridSpec(1.0, 3, 3, 0.0, 0.0))
        value_map = SemanticValueMap(grid)

        for key in ("target_pose", "matched_pose", "recommended_action"):
            with self.subTest(key=key):
                with self.assertRaises(OracleLeakageError):
                    value_map.update_from_candidates(
                        [{"label": "door", "confidence": 0.9, key: {"x": 1.0}}],
                        target_terms=["door"],
                    )

    def test_ranking_prefers_semantic_evidence_over_nearer_empty_frontier(self):
        grid = OccupancyGrid(GridSpec(1.0, 8, 8, 0.0, 0.0))
        value_map = SemanticValueMap(grid)
        value_map.update_from_candidates(
            [{"label": "door", "confidence": 0.95, "world_x": 6.2, "world_z": 6.2}],
            target_terms=["door"],
        )
        frontiers = [
            Frontier(row=1, col=1, unknown_neighbors=2, free_neighbors=1, distance_cells=1),
            Frontier(row=6, col=6, unknown_neighbors=2, free_neighbors=1, distance_cells=6),
        ]

        ranked = rank_frontiers(frontiers, semantic_map=value_map, distance_weight=0.1)

        self.assertEqual(ranked[0].frontier.cell, (6, 6))


if __name__ == "__main__":
    unittest.main()
