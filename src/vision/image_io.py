from __future__ import annotations

import base64
import io
from pathlib import Path

from PIL import Image


def load_image_from_any(value: str, root: Path | None = None) -> Image.Image:
    if not value:
        raise ValueError("observation_image is required")

    if value.startswith("data:image"):
        _, encoded = value.split(",", 1)
        return Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")

    possible = Path(value)
    if not possible.is_absolute() and root is not None:
        possible = root / possible
    if possible.exists():
        return Image.open(possible).convert("RGB")

    try:
        return Image.open(io.BytesIO(base64.b64decode(value))).convert("RGB")
    except Exception as exc:
        raise ValueError("observation_image must be a path, base64 image, or data URL") from exc


def image_to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def crop_from_point(image: Image.Image, point: list[int], patch_size: int) -> Image.Image:
    x, y = int(point[0]), int(point[1])
    half = patch_size // 2
    left = max(0, x - half)
    top = max(0, y - half)
    right = min(image.width, x + half)
    bottom = min(image.height, y + half)
    if right <= left or bottom <= top:
        raise ValueError("clicked_point is outside the image")
    return image.crop((left, top, right, bottom))

