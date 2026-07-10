from __future__ import annotations

import unittest
from pathlib import Path


class UIStreamContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.html = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "ui"
            / "static"
            / "index.html"
        ).read_text(encoding="utf-8")

    def test_environment_feedback_switches_to_post_action_pov(
        self,
    ) -> None:
        self.assertIn(
            'payload.observation_phase === "after_action"',
            self.html,
        )
        self.assertIn(
            'asset(payload.observation_path)',
            self.html,
        )

    def test_post_action_phase_is_visible_in_ui_state(self) -> None:
        self.assertIn(
            'textContent = "observing after action"',
            self.html,
        )


if __name__ == "__main__":
    unittest.main()
