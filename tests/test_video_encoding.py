from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import subprocess
import unittest

import cv2
from PIL import Image

from src.simulation.video_encoding import find_ffmpeg, write_browser_compatible_mp4


class BrowserVideoEncodingTests(unittest.TestCase):
    def test_h264_video_is_browser_compatible_and_decodable(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            frames = []
            for index, color in enumerate(((220, 40, 40), (40, 120, 220))):
                frame_path = root / f"frame_{index}.png"
                Image.new("RGB", (96, 64), color).save(frame_path)
                frames.append(frame_path)

            output_path = root / "demo.mp4"
            metadata = write_browser_compatible_mp4(
                frames,
                output_path,
                fps=2.0,
                hold_frames=2,
            )

            self.assertEqual(metadata["codec"], "h264")
            self.assertEqual(metadata["pixel_format"], "yuv420p")
            self.assertEqual(metadata["frame_count"], 4)
            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)
            self.assertFalse(list(root.glob("*.source.mp4")))

            capture = cv2.VideoCapture(str(output_path))
            self.assertTrue(capture.isOpened())
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 4)
            readable = 0
            while True:
                ok, _ = capture.read()
                if not ok:
                    break
                readable += 1
            capture.release()
            self.assertEqual(readable, 4)

            probe = subprocess.run(
                [find_ffmpeg(), "-hide_banner", "-i", str(output_path)],
                capture_output=True,
                text=True,
            )
            self.assertIn("Video: h264", probe.stderr)

    def test_empty_frame_list_is_rejected(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            with self.assertRaises(ValueError):
                write_browser_compatible_mp4(
                    [],
                    Path(temporary_directory) / "demo.mp4",
                    fps=2.0,
                )


if __name__ == "__main__":
    unittest.main()
