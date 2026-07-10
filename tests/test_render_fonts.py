from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from src.simulation import room_simulator


class RenderFontTests(unittest.TestCase):
    def test_cjk_font_candidates_precede_generic_fallback(self) -> None:
        candidate_names = [path.name for path in room_simulator.FONT_CANDIDATES]

        self.assertEqual(
            candidate_names[:4],
            [
                "msyh.ttc",
                "simhei.ttf",
                "NotoSansCJK-Regular.ttc",
                "wqy-zenhei.ttc",
            ],
        )
        self.assertEqual(candidate_names[-1], "DejaVuSans.ttf")

    def test_load_render_font_uses_first_available_candidate(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            candidates = (root / "first-cjk.ttc", root / "second-cjk.ttf")
            for candidate in candidates:
                candidate.touch()
            expected_font = object()

            with (
                patch.object(room_simulator, "FONT_CANDIDATES", candidates),
                patch.object(
                    room_simulator.ImageFont,
                    "truetype",
                    return_value=expected_font,
                ) as truetype,
            ):
                loaded_font = room_simulator.load_render_font(19)

        self.assertIs(loaded_font, expected_font)
        truetype.assert_called_once_with(str(candidates[0]), size=19)

    def test_load_render_font_skips_unreadable_candidate(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            candidates = (root / "broken-cjk.ttc", root / "valid-cjk.ttf")
            for candidate in candidates:
                candidate.touch()
            expected_font = object()

            with (
                patch.object(room_simulator, "FONT_CANDIDATES", candidates),
                patch.object(
                    room_simulator.ImageFont,
                    "truetype",
                    side_effect=(OSError("invalid font"), expected_font),
                ) as truetype,
            ):
                loaded_font = room_simulator.load_render_font(17)

        self.assertIs(loaded_font, expected_font)
        self.assertEqual(
            truetype.call_args_list,
            [
                unittest.mock.call(str(candidates[0]), size=17),
                unittest.mock.call(str(candidates[1]), size=17),
            ],
        )

    def test_load_render_font_falls_back_when_candidates_are_missing(self) -> None:
        missing_candidates = (
            Path("missing-cjk-font.ttc"),
            Path("missing-generic-font.ttf"),
        )
        expected_font = object()

        with (
            patch.object(room_simulator, "FONT_CANDIDATES", missing_candidates),
            patch.object(
                room_simulator.ImageFont,
                "truetype",
            ) as truetype,
            patch.object(
                room_simulator.ImageFont,
                "load_default",
                return_value=expected_font,
            ) as load_default,
        ):
            loaded_font = room_simulator.load_render_font(15)

        self.assertIs(loaded_font, expected_font)
        truetype.assert_not_called()
        load_default.assert_called_once_with()

    def test_room_simulator_legacy_loader_delegates_to_shared_function(self) -> None:
        expected_font = object()

        with patch.object(
            room_simulator,
            "load_render_font",
            return_value=expected_font,
        ) as loader:
            loaded_font = room_simulator.RoomSimulator._load_font(22)

        self.assertIs(loaded_font, expected_font)
        loader.assert_called_once_with(22)

    def test_available_cjk_font_renders_distinct_chinese_glyphs(
        self,
    ) -> None:
        font = room_simulator.load_render_font(18)
        selected_path = Path(str(getattr(font, "path", "")))
        cjk_font_names = {
            "msyh.ttc",
            "simhei.ttf",
            "NotoSansCJK-Regular.ttc",
            "wqy-zenhei.ttc",
            "DroidSansFallbackFull.ttf",
            "NotoSansSC-VF.ttf",
        }
        if selected_path.name not in cjk_font_names:
            self.skipTest("no configured CJK font is installed")

        def fingerprint(character: str) -> tuple[tuple[int, int], bytes]:
            mask = font.getmask(character)
            return mask.size, bytes(mask)

        chinese_fingerprints = {
            fingerprint(character)
            for character in ("\u4e2d", "\u6587", "\u6c99", "\u53d1")
        }
        missing_glyph_fingerprints = {
            fingerprint("?"),
            fingerprint("\u25a1"),
        }

        self.assertEqual(len(chinese_fingerprints), 4)
        self.assertTrue(
            chinese_fingerprints.isdisjoint(
                missing_glyph_fingerprints
            )
        )


if __name__ == "__main__":
    unittest.main()
