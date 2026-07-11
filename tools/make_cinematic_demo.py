from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.simulation.room_simulator import load_render_font
from src.simulation.video_encoding import write_browser_compatible_mp4


FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080

CYAN = (48, 224, 210)
BLUE = (88, 166, 255)
GREEN = (90, 242, 150)
RED = (255, 92, 82)
AMBER = (255, 199, 87)
WHITE = (238, 246, 255)
MUTED = (150, 170, 196)
PANEL = (13, 24, 40)
TERMINAL_ACTIONS = {"STOP", "Done", "ASK_CLARIFY"}


@dataclass(frozen=True)
class CinematicSettings:
    fps: int = 24
    hold_frames: int = 28
    intro_frames: int = 48
    outro_frames: int = 48
    strict_ai2thor: bool = True
    allow_legacy_pre_action: bool = False


@dataclass(frozen=True)
class CinematicFonts:
    title: ImageFont.ImageFont
    heading: ImageFont.ImageFont
    body: ImageFont.ImageFont
    small: ImageFont.ImageFont
    mono: ImageFont.ImageFont


def load_fonts() -> CinematicFonts:
    return CinematicFonts(
        title=load_render_font(46),
        heading=load_render_font(30),
        body=load_render_font(24),
        small=load_render_font(18),
        mono=load_render_font(20),
    )


def rgba(color: tuple[int, int, int], alpha: int = 255) -> tuple[int, int, int, int]:
    return (*color, alpha)


def resolve_project_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def relative_or_absolute(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def load_demo_summary(
    summary_path: Path,
    *,
    project_root: Path,
    settings: CinematicSettings,
) -> dict[str, Any]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    steps = summary.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("summary must contain at least one demo step")

    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            raise ValueError(f"step {index} is not an object")
        missing = [
            name
            for name in (
                "action",
                "observation_path",
                "topdown_path",
                "thought",
                "confidence",
            )
            if name not in step
        ]
        if missing:
            raise ValueError(
                f"step {index} is missing required fields: {', '.join(missing)}"
            )
        if settings.strict_ai2thor and step.get("backend") != "ai2thor":
            raise ValueError(
                f"step {index} backend is not strict AI2-THOR: "
                f"{step.get('backend')!r}"
            )
        phase = infer_observation_phase(step)
        if phase == "before_action" and not settings.allow_legacy_pre_action:
            raise ValueError(
                f"step {index} uses a legacy pre-action observation. "
                "Use a post-action summary or pass --allow-legacy-pre-action "
                "for archival rendering."
            )
        step["_cinematic_observation_phase"] = phase
        for field in ("observation_path", "topdown_path"):
            asset_path = resolve_project_path(project_root, step[field])
            if not asset_path.is_file():
                raise FileNotFoundError(
                    f"step {index} {field} does not exist: {asset_path}"
                )
    return summary


def infer_observation_phase(step: dict[str, Any]) -> str:
    explicit = str(step.get("observation_phase") or "").strip().lower()
    if explicit in {"after_action", "before_action"}:
        return explicit
    filename = Path(str(step.get("observation_path", ""))).name.lower()
    if "obs_after_" in filename or "observation_after_" in filename:
        return "after_action"
    return "before_action"


def observation_badge(step: dict[str, Any]) -> str:
    action = str(step.get("action", "UNKNOWN"))
    phase = step.get("_cinematic_observation_phase") or infer_observation_phase(step)
    if phase == "after_action":
        if action in TERMINAL_ACTIONS:
            return f"VERIFIED OBSERVATION AT TERMINAL DECISION: {action}"
        return f"OBSERVATION AFTER EXECUTED ACTION: {action}"
    return f"LEGACY OBSERVATION BEFORE NEXT ACTION: {action}"


def action_badge(step: dict[str, Any]) -> str:
    action = str(step.get("action", "UNKNOWN"))
    phase = step.get("_cinematic_observation_phase") or infer_observation_phase(step)
    if phase == "after_action":
        return f"EXECUTED ACTION: {action}"
    return f"NEXT ACTION IN LEGACY TRACE: {action}"


def phase_description(step: dict[str, Any]) -> str:
    phase = step.get("_cinematic_observation_phase") or infer_observation_phase(step)
    if phase == "after_action":
        if str(step.get("action")) in TERMINAL_ACTIONS:
            return "Terminal decision shown with the final verified simulator observation"
        return "Robot POV, pose and map are synchronized after the executed action"
    return "Archival compatibility mode: frame precedes the displayed action"


def read_image(project_root: Path, value: str | Path) -> Image.Image:
    return Image.open(resolve_project_path(project_root, value)).convert("RGB")


def cover_resize(image: Image.Image, box: tuple[int, int]) -> Image.Image:
    box_width, box_height = box
    image_width, image_height = image.size
    scale = max(box_width / image_width, box_height / image_height)
    new_width = max(1, int(round(image_width * scale)))
    new_height = max(1, int(round(image_height * scale)))
    resized = image.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS,
    )
    left = (new_width - box_width) // 2
    top = (new_height - box_height) // 2
    return resized.crop(
        (left, top, left + box_width, top + box_height)
    )


def round_rect(
    draw: ImageDraw.ImageDraw,
    xy: Sequence[int],
    radius: int,
    fill: tuple[int, ...],
    outline: tuple[int, ...] | None = None,
    width: int = 1,
) -> None:
    draw.rounded_rectangle(
        xy,
        radius=radius,
        fill=fill,
        outline=outline,
        width=width,
    )


def alpha_paste(
    base: Image.Image,
    overlay: Image.Image,
    xy: tuple[int, int] = (0, 0),
) -> None:
    rgba_overlay = overlay if overlay.mode == "RGBA" else overlay.convert("RGBA")
    base.paste(rgba_overlay, xy, rgba_overlay)


def wrap_text_pixels(
    draw: ImageDraw.ImageDraw,
    text: str,
    *,
    font: ImageFont.ImageFont,
    max_width: int,
    max_lines: int,
) -> list[str]:
    source = str(text or "").strip()
    if not source:
        return []
    tokens = source.split()
    if len(tokens) <= 1:
        tokens = list(source)

    lines: list[str] = []
    current = ""
    separator = " " if len(source.split()) > 1 else ""
    for token in tokens:
        probe = f"{current}{separator if current else ''}{token}"
        width = draw.textbbox((0, 0), probe, font=font)[2]
        if current and width > max_width:
            lines.append(current)
            current = token
        else:
            current = probe
    if current:
        lines.append(current)

    if len(lines) > max_lines:
        lines = lines[:max_lines]
        ellipsis = "..."
        candidate = lines[-1]
        while candidate:
            probe = f"{candidate}{ellipsis}"
            if draw.textbbox((0, 0), probe, font=font)[2] <= max_width:
                lines[-1] = probe
                break
            candidate = candidate[:-1]
        if not candidate:
            lines[-1] = ellipsis
    return lines


def draw_wrapped(
    draw: ImageDraw.ImageDraw,
    text: str,
    xy: tuple[int, int],
    *,
    max_width: int,
    font: ImageFont.ImageFont,
    fill: tuple[int, ...],
    line_gap: int = 8,
    max_lines: int = 4,
) -> None:
    x, y = xy
    lines = wrap_text_pixels(
        draw,
        text,
        font=font,
        max_width=max_width,
        max_lines=max_lines,
    )
    bbox = draw.textbbox((0, 0), "Ag", font=font)
    line_height = max(1, bbox[3] - bbox[1])
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        y += line_height + line_gap


def scale_bbox(
    candidate: dict[str, Any] | None,
    source_size: tuple[int, int],
    destination: tuple[int, int, int, int],
) -> list[int] | None:
    if not candidate:
        return None
    bbox = candidate.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    source_width, source_height = source_size
    dx, dy, destination_width, destination_height = destination
    x0, y0, x1, y1 = [float(value) for value in bbox]
    return [
        dx + int(x0 / source_width * destination_width),
        dy + int(y0 / source_height * destination_height),
        dx + int(x1 / source_width * destination_width),
        dy + int(y1 / source_height * destination_height),
    ]


def gradient_background(progress: float) -> Image.Image:
    array = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    xs = np.linspace(0, 1, FRAME_WIDTH, dtype=np.float32)
    ys = np.linspace(0, 1, FRAME_HEIGHT, dtype=np.float32)
    nx, ny = np.meshgrid(xs, ys)
    pulse = 0.5 + 0.5 * np.sin(progress * 2 * math.pi + nx * 5.0)
    array[:, :, 0] = np.clip(5 + 15 * nx + 10 * pulse, 0, 255)
    array[:, :, 1] = np.clip(10 + 20 * ny, 0, 255)
    array[:, :, 2] = np.clip(18 + 35 * (1 - nx) + 8 * pulse, 0, 255)
    return Image.fromarray(array, "RGB")


def summary_instruction(summary: dict[str, Any]) -> str:
    if summary.get("instruction"):
        return str(summary["instruction"])
    for step in summary["steps"]:
        task_plan = step.get("task_plan") or {}
        for key in ("instruction", "task_summary"):
            if task_plan.get(key):
                return str(task_plan[key])
    return "Embodied AI2-THOR task execution"


def summary_scene(summary: dict[str, Any]) -> str:
    if summary.get("scene"):
        return str(summary["scene"])
    return str(summary["steps"][0].get("scene") or "AI2-THOR scene")


def summary_target(summary: dict[str, Any]) -> str:
    for step in reversed(summary["steps"]):
        candidate = step.get("best_candidate") or {}
        if candidate.get("label"):
            return str(candidate["label"])
    return "task target"


def visible_timeline(
    steps: list[dict[str, Any]],
    active_index: int,
    *,
    window_size: int = 7,
) -> list[tuple[int, dict[str, Any]]]:
    if len(steps) <= window_size:
        return list(enumerate(steps))
    half = window_size // 2
    start = max(0, min(active_index - half, len(steps) - window_size))
    return list(enumerate(steps[start : start + window_size], start=start))


def draw_hud(
    *,
    summary: dict[str, Any],
    step: dict[str, Any],
    step_index: int,
    local_progress: float,
    global_progress: float,
    project_root: Path,
    fonts: CinematicFonts,
) -> Image.Image:
    steps = summary["steps"]
    base = gradient_background(global_progress)
    draw = ImageDraw.Draw(base, "RGBA")

    source_observation = read_image(project_root, step["observation_path"])
    observation_source_size = source_observation.size
    observation = cover_resize(source_observation, (1120, 720))
    topdown = cover_resize(
        read_image(project_root, step["topdown_path"]),
        (470, 470),
    )

    obs_x, obs_y, obs_width, obs_height = 70, 205, 1120, 720
    map_x, map_y, map_width, map_height = 1375, 185, 470, 470

    shadow = Image.new(
        "RGBA",
        (obs_width + 36, obs_height + 36),
        (0, 0, 0, 0),
    )
    shadow_draw = ImageDraw.Draw(shadow, "RGBA")
    shadow_draw.rounded_rectangle(
        [18, 18, obs_width + 18, obs_height + 18],
        radius=22,
        fill=(0, 0, 0, 110),
    )
    shadow = shadow.filter(ImageFilter.GaussianBlur(16))
    alpha_paste(base, shadow, (obs_x - 18, obs_y - 18))
    base.paste(observation, (obs_x, obs_y))
    draw.rounded_rectangle(
        [obs_x, obs_y, obs_x + obs_width, obs_y + obs_height],
        radius=20,
        outline=rgba(CYAN, 230),
        width=4,
    )

    sweep_x = obs_x + int((local_progress % 1.0) * obs_width)
    draw.rectangle(
        [sweep_x - 5, obs_y, sweep_x + 5, obs_y + obs_height],
        fill=rgba(CYAN, 95),
    )
    draw.line(
        [(sweep_x, obs_y), (sweep_x, obs_y + obs_height)],
        fill=rgba(WHITE, 180),
        width=2,
    )

    candidate = step.get("best_candidate")
    scaled = scale_bbox(
        candidate,
        observation_source_size,
        (obs_x, obs_y, obs_width, obs_height),
    )
    if scaled and candidate:
        pulse = 0.65 + 0.35 * math.sin(global_progress * 2 * math.pi * 8)
        color = (int(255 * pulse), 70, 70)
        for grow, alpha in ((18, 50), (10, 95), (3, 235)):
            draw.rectangle(
                [
                    scaled[0] - grow,
                    scaled[1] - grow,
                    scaled[2] + grow,
                    scaled[3] + grow,
                ],
                outline=rgba(color, alpha),
                width=4,
            )
        confidence = float(candidate.get("confidence", 0.0))
        draw.text(
            (scaled[0], max(obs_y, scaled[1] - 36)),
            f"TARGET: {candidate.get('label', 'unknown')}  {confidence:.3f}",
            font=fonts.mono,
            fill=rgba(RED, 255),
        )

    base.paste(topdown, (map_x, map_y))
    draw.rounded_rectangle(
        [map_x, map_y, map_x + map_width, map_y + map_height],
        radius=16,
        outline=rgba(BLUE, 235),
        width=4,
    )

    draw.text(
        (70, 54),
        "EMBODIED AI2-THOR AGENT",
        font=fonts.title,
        fill=rgba(WHITE, 255),
    )
    subtitle = f"{summary_scene(summary)} | target: {summary_target(summary)}"
    draw.text(
        (72, 116),
        subtitle,
        font=fonts.heading,
        fill=rgba(CYAN, 255),
    )
    draw.text(
        (70, 160),
        phase_description(step),
        font=fonts.body,
        fill=rgba(MUTED, 255),
    )

    round_rect(
        draw,
        [obs_x + 20, obs_y + 20, obs_x + 765, obs_y + 62],
        12,
        (6, 18, 30, 210),
        rgba(CYAN, 210),
        2,
    )
    draw.text(
        (obs_x + 38, obs_y + 30),
        observation_badge(step),
        font=fonts.small,
        fill=rgba(WHITE, 255),
    )

    panel_x, panel_y, panel_width, panel_height = 1235, 690, 610, 260
    round_rect(
        draw,
        [
            panel_x,
            panel_y,
            panel_x + panel_width,
            panel_y + panel_height,
        ],
        18,
        rgba(PANEL, 235),
        rgba(BLUE, 95),
        2,
    )
    draw.text(
        (panel_x + 28, panel_y + 24),
        f"STEP {step_index:02d}",
        font=fonts.heading,
        fill=rgba(MUTED, 255),
    )
    draw.text(
        (panel_x + 28, panel_y + 60),
        action_badge(step),
        font=fonts.small,
        fill=rgba(MUTED, 255),
    )
    action = str(step["action"])
    action_color = (
        GREEN
        if action in TERMINAL_ACTIONS
        else CYAN
        if action == "INSPECT"
        else BLUE
    )
    draw.text(
        (panel_x + 28, panel_y + 90),
        action,
        font=fonts.heading,
        fill=rgba(action_color, 255),
    )
    draw.text(
        (panel_x + 330, panel_y + 34),
        "CONFIDENCE",
        font=fonts.small,
        fill=rgba(MUTED, 255),
    )
    confidence = min(1.0, max(0.0, float(step["confidence"])))
    draw.rounded_rectangle(
        [panel_x + 330, panel_y + 72, panel_x + 555, panel_y + 96],
        radius=12,
        fill=(25, 38, 58, 255),
    )
    draw.rounded_rectangle(
        [
            panel_x + 330,
            panel_y + 72,
            panel_x + 330 + int(225 * confidence),
            panel_y + 96,
        ],
        radius=12,
        fill=rgba(RED, 255),
    )
    draw.text(
        (panel_x + 330, panel_y + 108),
        f"{confidence:.3f}",
        font=fonts.heading,
        fill=rgba(RED, 255),
    )
    draw_wrapped(
        draw,
        str(step.get("thought") or ""),
        (panel_x + 28, panel_y + 150),
        max_width=550,
        font=fonts.small,
        fill=rgba(WHITE, 255),
        max_lines=4,
    )

    timeline_x, timeline_y = 70, 955
    draw.text(
        (timeline_x, timeline_y - 42),
        "Executed trajectory",
        font=fonts.body,
        fill=rgba(WHITE, 255),
    )
    timeline_steps = visible_timeline(steps, step_index)
    gap = 15
    step_width = int(
        (FRAME_WIDTH - 2 * timeline_x - gap * (len(timeline_steps) - 1))
        / max(1, len(timeline_steps))
    )
    for timeline_position, (absolute_index, timeline_step) in enumerate(
        timeline_steps
    ):
        x = timeline_x + timeline_position * (step_width + gap)
        active = absolute_index == step_index
        fill = (30, 56, 80, 235) if active else (16, 30, 48, 210)
        outline = CYAN if active else (68, 88, 110)
        round_rect(
            draw,
            [x, timeline_y, x + step_width, timeline_y + 78],
            12,
            fill,
            rgba(outline, 220),
            2,
        )
        prefix = (
            "executed"
            if timeline_step.get("_cinematic_observation_phase")
            == "after_action"
            else "next"
        )
        draw.text(
            (x + 14, timeline_y + 12),
            f"{absolute_index}: {prefix} {timeline_step['action']}",
            font=fonts.small,
            fill=rgba(WHITE if active else MUTED, 255),
        )
        bar_width = int(
            (step_width - 28)
            * min(1.0, max(0.0, float(timeline_step["confidence"])))
        )
        draw.rounded_rectangle(
            [
                x + 14,
                timeline_y + 49,
                x + step_width - 14,
                timeline_y + 61,
            ],
            radius=6,
            fill=(28, 38, 56, 255),
        )
        draw.rounded_rectangle(
            [x + 14, timeline_y + 49, x + 14 + bar_width, timeline_y + 61],
            radius=6,
            fill=rgba(
                RED
                if str(timeline_step["action"]) in TERMINAL_ACTIONS
                else CYAN,
                255,
            ),
        )

    badges = [
        ("REAL AI2-THOR", GREEN),
        ("POST-ACTION SYNC", CYAN),
        ("NO SYNTHETIC FRAMES", AMBER),
    ]
    badge_x = 1260
    for text, color in badges:
        text_width = draw.textbbox((0, 0), text, font=fonts.small)[2]
        width = text_width + 38
        round_rect(
            draw,
            [badge_x, 54, badge_x + width, 96],
            14,
            (20, 35, 50, 230),
            rgba(color, 230),
            2,
        )
        draw.text(
            (badge_x + 18, 64),
            text,
            font=fonts.small,
            fill=rgba(color, 255),
        )
        badge_x += width + 16

    return base


def intro_frame(
    *,
    summary: dict[str, Any],
    progress: float,
    fonts: CinematicFonts,
) -> Image.Image:
    frame = gradient_background(progress)
    draw = ImageDraw.Draw(frame, "RGBA")
    draw.text(
        (110, 290),
        "Embodied AI2-THOR Agent",
        font=fonts.title,
        fill=rgba(WHITE, 255),
    )
    draw.text(
        (112, 358),
        f"Scene: {summary_scene(summary)}",
        font=fonts.heading,
        fill=rgba(CYAN, 255),
    )
    draw_wrapped(
        draw,
        f"Instruction: {summary_instruction(summary)}",
        (112, 420),
        max_width=1250,
        font=fonts.body,
        fill=rgba(MUTED, 255),
        max_lines=3,
    )
    labels = [
        "Language and visual observation",
        "Model decision summary and structured action",
        "AI2-THOR execution and postcondition verification",
        "Post-action robot POV and map synchronization",
    ]
    x0, y0 = 112, 565
    for index, label in enumerate(labels):
        alpha = int(255 * min(1, max(0, progress * 5 - index)))
        round_rect(
            draw,
            [x0, y0 + index * 70, x0 + 850, y0 + index * 70 + 48],
            12,
            (16, 30, 48, alpha),
            rgba(CYAN, alpha),
            2,
        )
        draw.text(
            (x0 + 20, y0 + index * 70 + 11),
            label,
            font=fonts.body,
            fill=rgba(WHITE, alpha),
        )
    return frame


def outro_frame(
    *,
    summary: dict[str, Any],
    progress: float,
    project_root: Path,
    fonts: CinematicFonts,
) -> Image.Image:
    last_index = len(summary["steps"]) - 1
    last_step = summary["steps"][last_index]
    frame = draw_hud(
        summary=summary,
        step=last_step,
        step_index=last_index,
        local_progress=progress,
        global_progress=progress,
        project_root=project_root,
        fonts=fonts,
    )
    overlay = Image.new(
        "RGBA",
        (FRAME_WIDTH, FRAME_HEIGHT),
        (0, 0, 0, int(92 * progress)),
    )
    overlay_draw = ImageDraw.Draw(overlay, "RGBA")
    alpha = int(235 * min(1, progress * 1.8))
    overlay_draw.rounded_rectangle(
        [118, 118, 1040, 290],
        radius=22,
        fill=(6, 18, 30, int(218 * min(1, progress * 1.8))),
        outline=rgba(GREEN, alpha),
        width=3,
    )
    overlay_draw.text(
        (152, 144),
        "DEMO TRACE COMPLETE",
        font=fonts.title,
        fill=rgba(GREEN, 255),
    )
    final_action = str(last_step["action"])
    overlay_draw.text(
        (154, 215),
        (
            f"{len(summary['steps'])}-step trace | final action {final_action} | "
            f"backend {last_step.get('backend', 'unknown')}"
        ),
        font=fonts.body,
        fill=rgba(WHITE, 255),
    )
    alpha_paste(frame, overlay)
    return frame


def render_source_frames(
    *,
    summary: dict[str, Any],
    frame_dir: Path,
    project_root: Path,
    settings: CinematicSettings,
) -> list[Path]:
    frame_dir.mkdir(parents=True, exist_ok=True)
    for old_frame in frame_dir.glob("cinematic_*.png"):
        old_frame.unlink()

    fonts = load_fonts()
    frames: list[Image.Image] = []
    for index in range(settings.intro_frames):
        progress = index / max(settings.intro_frames - 1, 1)
        frames.append(
            intro_frame(
                summary=summary,
                progress=progress,
                fonts=fonts,
            )
        )

    step_count = len(summary["steps"])
    for step_index, step in enumerate(summary["steps"]):
        for hold_index in range(settings.hold_frames):
            frames.append(
                draw_hud(
                    summary=summary,
                    step=step,
                    step_index=step_index,
                    local_progress=hold_index / max(settings.hold_frames, 1),
                    global_progress=(
                        step_index
                        + hold_index / max(settings.hold_frames, 1)
                    )
                    / step_count,
                    project_root=project_root,
                    fonts=fonts,
                )
            )

    for index in range(settings.outro_frames):
        progress = index / max(settings.outro_frames - 1, 1)
        frames.append(
            outro_frame(
                summary=summary,
                progress=progress,
                project_root=project_root,
                fonts=fonts,
            )
        )

    paths: list[Path] = []
    for index, frame in enumerate(frames):
        path = frame_dir / f"cinematic_{index:04d}.png"
        frame.save(path, format="PNG")
        paths.append(path)
    return paths


def probe_video(path: Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"failed to open generated video: {path}")
        return {
            "frame_count": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            "fps": float(capture.get(cv2.CAP_PROP_FPS)),
            "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
    finally:
        capture.release()


def sample_frame_stats(paths: list[Path]) -> list[dict[str, Any]]:
    if not paths:
        return []
    selected = [paths[0], paths[len(paths) // 2], paths[-1]]
    stats: list[dict[str, Any]] = []
    for path in selected:
        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError(f"failed to inspect rendered frame: {path}")
        stats.append(
            {
                "name": path.name,
                "mean": round(float(image.mean()), 2),
                "std": round(float(image.std()), 2),
            }
        )
    return stats


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        temporary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def generate_cinematic_demo(
    *,
    summary_path: Path,
    output_video: Path,
    verification_path: Path,
    project_root: Path = PROJECT_ROOT,
    keep_frames_dir: Path | None = None,
    settings: CinematicSettings = CinematicSettings(),
    encoder: Callable[..., dict[str, Any]] = write_browser_compatible_mp4,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    summary_path = resolve_project_path(project_root, summary_path).resolve()
    output_video = resolve_project_path(project_root, output_video).resolve()
    verification_path = resolve_project_path(
        project_root,
        verification_path,
    ).resolve()
    summary = load_demo_summary(
        summary_path,
        project_root=project_root,
        settings=settings,
    )

    output_video.parent.mkdir(parents=True, exist_ok=True)
    output_video.unlink(missing_ok=True)
    verification_path.unlink(missing_ok=True)

    if keep_frames_dir is None:
        frame_context: Any = tempfile.TemporaryDirectory(
            prefix="ai2thor_cinematic_"
        )
        frame_dir = Path(frame_context.name)
    else:
        frame_context = None
        frame_dir = resolve_project_path(
            project_root,
            keep_frames_dir,
        ).resolve()

    try:
        frame_paths = render_source_frames(
            summary=summary,
            frame_dir=frame_dir,
            project_root=project_root,
            settings=settings,
        )
        frame_stats = sample_frame_stats(frame_paths)
        encoding = encoder(
            frame_paths,
            output_video,
            fps=settings.fps,
        )
        probe = probe_video(output_video)
        if probe["frame_count"] != len(frame_paths):
            raise RuntimeError(
                "generated video frame count does not match rendered frames: "
                f"{probe['frame_count']} != {len(frame_paths)}"
            )
        if probe["width"] != FRAME_WIDTH or probe["height"] != FRAME_HEIGHT:
            raise RuntimeError(
                "generated video resolution is incorrect: "
                f"{probe['width']}x{probe['height']}"
            )

        steps = summary["steps"]
        verification = {
            "summary_path": relative_or_absolute(summary_path, project_root),
            "video_path": relative_or_absolute(output_video, project_root),
            "verification_path": relative_or_absolute(
                verification_path,
                project_root,
            ),
            "exists": output_video.is_file(),
            "bytes": output_video.stat().st_size,
            "sha256": hashlib.sha256(output_video.read_bytes()).hexdigest(),
            "frame_count": probe["frame_count"],
            "fps": probe["fps"],
            "width": probe["width"],
            "height": probe["height"],
            "duration_seconds": round(
                probe["frame_count"] / float(settings.fps),
                3,
            ),
            "codec": encoding.get("codec"),
            "pixel_format": encoding.get("pixel_format"),
            "source_steps": len(steps),
            "all_steps_ai2thor": all(
                step.get("backend") == "ai2thor" for step in steps
            ),
            "observation_phases": [
                step["_cinematic_observation_phase"] for step in steps
            ],
            "post_action_semantics": all(
                step["_cinematic_observation_phase"] == "after_action"
                for step in steps
            ),
            "actions": [str(step["action"]) for step in steps],
            "final_action": str(steps[-1]["action"]),
            "final_confidence": float(steps[-1]["confidence"]),
            "final_completion_status": steps[-1].get(
                "completion_status",
                summary.get("completion_status"),
            ),
            "frame_stats": frame_stats,
            "kept_frames_dir": (
                relative_or_absolute(frame_dir, project_root)
                if keep_frames_dir is not None
                else None
            ),
        }
        write_json_atomic(verification_path, verification)
        return verification
    except Exception:
        output_video.unlink(missing_ok=True)
        verification_path.unlink(missing_ok=True)
        raise
    finally:
        if frame_context is not None:
            frame_context.cleanup()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate an action-aligned cinematic video from an AI2-THOR "
            "demo summary. Post-action observations are required by default."
        )
    )
    parser.add_argument(
        "--summary",
        required=True,
        help="Demo summary JSON path, relative to --project-root or absolute.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output MP4 path, relative to --project-root or absolute.",
    )
    parser.add_argument(
        "--verification",
        help=(
            "Verification JSON path. Defaults to OUTPUT with "
            "'.verification.json' suffix."
        ),
    )
    parser.add_argument(
        "--project-root",
        default=str(PROJECT_ROOT),
        help="Project root used to resolve summary asset paths.",
    )
    parser.add_argument(
        "--keep-frames",
        help="Optional directory for retaining rendered PNG source frames.",
    )
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--hold-frames", type=int, default=28)
    parser.add_argument("--intro-frames", type=int, default=48)
    parser.add_argument("--outro-frames", type=int, default=48)
    parser.add_argument(
        "--allow-non-ai2thor",
        action="store_true",
        help="Allow non-AI2-THOR backends for development-only rendering.",
    )
    parser.add_argument(
        "--allow-legacy-pre-action",
        action="store_true",
        help=(
            "Allow old summaries whose observation precedes the action. "
            "The generated HUD labels the trace as legacy."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if args.hold_frames <= 0:
        raise ValueError("--hold-frames must be positive")
    if args.intro_frames < 0 or args.outro_frames < 0:
        raise ValueError("intro/outro frame counts cannot be negative")

    project_root = Path(args.project_root)
    output_path = Path(args.output)
    verification_path = (
        Path(args.verification)
        if args.verification
        else output_path.with_suffix(".verification.json")
    )
    verification = generate_cinematic_demo(
        summary_path=Path(args.summary),
        output_video=output_path,
        verification_path=verification_path,
        project_root=project_root,
        keep_frames_dir=Path(args.keep_frames) if args.keep_frames else None,
        settings=CinematicSettings(
            fps=args.fps,
            hold_frames=args.hold_frames,
            intro_frames=args.intro_frames,
            outro_frames=args.outro_frames,
            strict_ai2thor=not args.allow_non_ai2thor,
            allow_legacy_pre_action=args.allow_legacy_pre_action,
        ),
    )
    print(json.dumps(verification, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
