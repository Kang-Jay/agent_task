from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from src.task.config import load_config


OBJECTS = {
    "red cup": {"color": (210, 55, 55), "shape": "ellipse"},
    "blue book": {"color": (55, 95, 205), "shape": "rectangle"},
    "green plant": {"color": (60, 155, 80), "shape": "ellipse"},
}


def _draw_room(draw: ImageDraw.ImageDraw, width: int, height: int) -> None:
    draw.rectangle([0, 0, width, height], fill=(226, 224, 216))
    draw.rectangle([0, int(height * 0.68), width, height], fill=(184, 168, 142))
    draw.rectangle([35, 245, 210, 310], fill=(124, 92, 62))
    draw.rectangle([45, 180, 195, 245], fill=(178, 142, 96))
    draw.rectangle([260, 210, 395, 290], fill=(116, 126, 145))
    draw.rectangle([285, 170, 370, 210], fill=(138, 150, 172))
    draw.rectangle([40, 65, 145, 150], fill=(188, 205, 218))
    draw.rectangle([320, 55, 405, 165], fill=(170, 130, 84))


def _draw_object(draw: ImageDraw.ImageDraw, name: str, bbox: list[int]) -> None:
    meta = OBJECTS[name]
    if meta["shape"] == "ellipse":
        draw.ellipse(bbox, fill=meta["color"], outline=(40, 40, 40), width=3)
    else:
        draw.rectangle(bbox, fill=meta["color"], outline=(40, 40, 40), width=3)
    draw.text((bbox[0], max(0, bbox[1] - 18)), name, fill=(35, 35, 35))


def _make_image(path: Path, target_name: str, bbox: list[int], distractors: list[tuple[str, list[int]]]) -> None:
    image = Image.new("RGB", (448, 448), (230, 230, 225))
    draw = ImageDraw.Draw(image)
    _draw_room(draw, 448, 448)
    for name, dbbox in distractors:
        _draw_object(draw, name, dbbox)
    _draw_object(draw, target_name, bbox)
    draw.text((14, 14), "first-person observation", fill=(20, 20, 20))
    image.save(path)


def build_dataset() -> list[dict[str, object]]:
    config = load_config()
    config.image_dir.mkdir(parents=True, exist_ok=True)
    config.annotation_file.parent.mkdir(parents=True, exist_ok=True)
    config.trajectory_dir.mkdir(parents=True, exist_ok=True)

    episodes = [
        {
            "episode_id": "ep_red_cup_visible",
            "instruction": "Find the red cup on the table",
            "target": {"name": "red cup", "type": "cup", "bbox": [92, 168, 144, 225]},
            "image": "images/ep_red_cup_visible_000.png",
            "expected_action": "STOP",
        },
        {
            "episode_id": "ep_blue_book_visible",
            "instruction": "找到蓝色书本",
            "target": {"name": "blue book", "type": "book", "bbox": [300, 192, 374, 228]},
            "image": "images/ep_blue_book_visible_000.png",
            "expected_action": "STOP",
        },
        {
            "episode_id": "ep_green_plant_visible",
            "instruction": "Find the green plant near the window",
            "target": {"name": "green plant", "type": "plant", "bbox": [72, 82, 126, 146]},
            "image": "images/ep_green_plant_visible_000.png",
            "expected_action": "STOP",
        },
    ]

    for episode in episodes:
        target = episode["target"]
        image_path = config.image_dir / Path(episode["image"]).name
        distractors = [
            ("blue book", [310, 248, 380, 282]),
            ("green plant", [345, 82, 392, 150]),
        ]
        if target["name"] == "blue book":
            distractors = [("red cup", [90, 172, 138, 224]), ("green plant", [345, 82, 392, 150])]
        if target["name"] == "green plant":
            distractors = [("red cup", [90, 172, 138, 224]), ("blue book", [310, 248, 380, 282])]
        _make_image(image_path, target["name"], target["bbox"], distractors)
        episode["steps"] = [
            {
                "image": episode["image"],
                "thought": "Target-like evidence should be visible in the current first-person observation.",
                "action": episode["expected_action"],
                "done": True,
            }
        ]

    with config.annotation_file.open("w", encoding="utf-8") as handle:
        for episode in episodes:
            handle.write(json.dumps(episode, ensure_ascii=False) + "\n")

    return episodes


def main() -> None:
    episodes = build_dataset()
    print(f"generated {len(episodes)} demo episodes")


if __name__ == "__main__":
    main()

