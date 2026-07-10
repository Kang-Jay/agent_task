from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any
from uuid import uuid4

import cv2


def find_ffmpeg() -> str:
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg
    try:
        import imageio_ffmpeg

        bundled_ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError(
            "H.264 video encoding requires system ffmpeg or imageio-ffmpeg"
        ) from exc
    if not Path(bundled_ffmpeg).exists():
        raise RuntimeError(f"ffmpeg executable does not exist: {bundled_ffmpeg}")
    return bundled_ffmpeg


def write_browser_compatible_mp4(
    frames: list[Path],
    path: Path,
    *,
    fps: float,
    hold_frames: int = 1,
) -> dict[str, Any]:
    if not frames:
        raise ValueError("at least one frame is required")
    if fps <= 0:
        raise ValueError("fps must be positive")
    if hold_frames <= 0:
        raise ValueError("hold_frames must be positive")

    path.parent.mkdir(parents=True, exist_ok=True)
    first = cv2.imread(str(frames[0]))
    if first is None:
        raise RuntimeError(f"failed to read video frame: {frames[0]}")
    height, width = first.shape[:2]
    source_path = path.with_name(
        f".{path.stem}.{uuid4().hex}.source.mp4"
    )

    writer = cv2.VideoWriter(
        str(source_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"failed to initialize temporary video writer: {source_path}")

    encoded_frames = 0
    try:
        for frame_path in frames:
            frame = cv2.imread(str(frame_path))
            if frame is None:
                raise RuntimeError(f"failed to read video frame: {frame_path}")
            if frame.shape[:2] != (height, width):
                raise ValueError(
                    f"frame size mismatch for {frame_path}: "
                    f"{frame.shape[1]}x{frame.shape[0]} != {width}x{height}"
                )
            for _ in range(hold_frames):
                writer.write(frame)
                encoded_frames += 1
    finally:
        writer.release()

    ffmpeg = find_ffmpeg()
    command = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(source_path),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(path),
    ]
    try:
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        path.unlink(missing_ok=True)
        raise RuntimeError(
            f"ffmpeg H.264 encoding failed: {exc.stderr.strip()}"
        ) from exc
    finally:
        source_path.unlink(missing_ok=True)

    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"H.264 video was not created: {path}")
    return {
        "path": str(path),
        "codec": "h264",
        "pixel_format": "yuv420p",
        "width": width,
        "height": height,
        "fps": float(fps),
        "frame_count": encoded_frames,
        "duration_seconds": encoded_frames / float(fps),
        "bytes": path.stat().st_size,
    }
