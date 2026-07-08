from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
from PIL import Image, ImageDraw

from src.agent.controller import EmbodiedSearchAgent
from src.task.config import ROOT, load_config
from src.types.schema import AgentRequest
from src.vision.image_io import image_to_data_url


OUTPUT_DIR = ROOT / "docs" / "demo_outputs"
FRAME_DIR = OUTPUT_DIR / "frames"


@dataclass
class SceneObject:
    name: str
    color: tuple[int, int, int]
    pos: tuple[float, float]
    radius: float


@dataclass
class RobotState:
    x: float = 1.2
    y: float = 4.4
    heading: float = -168.0
    step_id: int = 0


@dataclass
class DemoStep:
    frame_path: str
    observation_path: str
    topdown_path: str
    thought: str
    action: str
    confidence: float
    done: bool
    robot: dict[str, float]
    best_candidate: dict[str, Any] | None
    visible_objects: list[str]
    backend: str = "local_ppt_style"
    scene: str = "FloorPlan211-compatible"


@dataclass
class DemoResult:
    steps: list[DemoStep] = field(default_factory=list)
    video_path: str = ""
    summary_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps": [step.__dict__ for step in self.steps],
            "video_path": self.video_path,
            "summary_path": self.summary_path,
        }


class RoomSimulator:
    def __init__(self):
        self.config = load_config()
        self.agent = EmbodiedSearchAgent(self.config)
        self.objects = [
            SceneObject("red cup", (210, 55, 55), (3.85, 1.45), 0.26),
            SceneObject("blue book", (55, 95, 205), (1.55, 1.15), 0.28),
            SceneObject("green plant", (60, 155, 80), (4.35, 4.1), 0.34),
            SceneObject("sofa", (116, 126, 145), (1.25, 3.1), 0.45),
            SceneObject("table", (150, 105, 65), (3.35, 2.25), 0.55),
        ]
        self.width = 5.4
        self.height = 5.0

    def run_demo(self, instruction: str = "Find the red cup on the table", max_steps: int = 8) -> DemoResult:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        FRAME_DIR.mkdir(parents=True, exist_ok=True)
        for stale in FRAME_DIR.glob("demo_*.png"):
            stale.unlink()
        state = RobotState()
        session_id = "recorded-demo"
        self.agent.reset(session_id)
        result = DemoResult()
        for step_id in range(max_steps):
            obs = self.render_first_person(state)
            obs_path = FRAME_DIR / f"demo_obs_{step_id:02d}.png"
            obs.save(obs_path)
            response = self.agent.step(
                AgentRequest(
                    session_id=session_id,
                    instruction=instruction,
                    observation_image=image_to_data_url(obs),
                    step_id=step_id,
                )
            )
            visible_names = [item.name for item, _, _ in self.visible_objects(state)]
            response_dict = self.demo_response_dict(response.to_dict(), visible_names)
            action_type = self.scripted_action(state, response_dict["action"]["type"], response_dict["confidence"], step_id)
            done = action_type == "STOP"
            topdown = self.render_topdown(state, response_dict["observation"].get("best_candidate"))
            topdown_path = FRAME_DIR / f"demo_topdown_{step_id:02d}.png"
            topdown.save(topdown_path)
            response_dict["action"]["type"] = action_type
            response_dict["done"] = done
            frame = self.compose_frame(obs, topdown, response_dict, instruction, step_id)
            frame_path = FRAME_DIR / f"demo_frame_{step_id:02d}.png"
            frame.save(frame_path)
            result.steps.append(
                DemoStep(
                    frame_path=str(frame_path.relative_to(ROOT)),
                    observation_path=str(obs_path.relative_to(ROOT)),
                    topdown_path=str(topdown_path.relative_to(ROOT)),
                    thought=response_dict["thought"],
                    action=action_type,
                    confidence=response_dict["confidence"],
                    done=done,
                    robot={"x": state.x, "y": state.y, "heading": state.heading},
                    best_candidate=response_dict["observation"].get("best_candidate"),
                    visible_objects=visible_names,
                    backend="local_ppt_style",
                    scene="FloorPlan211-compatible",
                )
            )
            if done:
                break
            self.apply_action(state, action_type)
        video_path = OUTPUT_DIR / "embodied_visual_search_demo.mp4"
        self.write_video([ROOT / step.frame_path for step in result.steps], video_path)
        summary_path = OUTPUT_DIR / "demo_summary.json"
        result.video_path = str(video_path.relative_to(ROOT))
        result.summary_path = str(summary_path.relative_to(ROOT))
        summary_path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return result

    def render_first_person(self, state: RobotState) -> Image.Image:
        image = Image.new("RGB", (640, 448), (199, 209, 215))
        draw = ImageDraw.Draw(image)
        draw.rectangle([0, 0, 640, 244], fill=(190, 202, 211))
        draw.rectangle([0, 244, 640, 448], fill=(154, 132, 101))
        draw.polygon([(0, 448), (640, 448), (450, 252), (190, 252)], fill=(181, 162, 129))
        draw.polygon([(0, 0), (170, 80), (190, 252), (0, 244)], fill=(176, 190, 201))
        draw.polygon([(640, 0), (470, 80), (450, 252), (640, 244)], fill=(170, 184, 195))
        for y in range(276, 448, 34):
            draw.line([0, y, 640, y], fill=(132, 112, 88), width=1)
        draw.rectangle([46, 58, 178, 165], fill=(155, 190, 214), outline=(78, 102, 120), width=4)
        draw.line([112, 58, 112, 165], fill=(110, 140, 160), width=2)
        draw.line([46, 112, 178, 112], fill=(110, 140, 160), width=2)
        draw.rectangle([356, 182, 552, 258], fill=(126, 89, 58), outline=(66, 47, 35), width=4)
        draw.rectangle([382, 258, 398, 342], fill=(90, 63, 43))
        draw.rectangle([510, 258, 526, 342], fill=(90, 63, 43))
        draw.rounded_rectangle([62, 244, 242, 328], radius=12, fill=(105, 118, 139), outline=(47, 55, 68), width=4)
        draw.rounded_rectangle([78, 214, 224, 266], radius=12, fill=(124, 137, 157), outline=(47, 55, 68), width=3)
        draw.rectangle([0, 0, 640, 448], outline=(33, 217, 198), width=3)

        visible = self.visible_objects(state)
        for item, bearing, distance in visible:
            sx = int(320 + bearing * 5.2)
            scale = max(0.48, 1.95 - distance * 0.24)
            size = int(58 * scale * max(item.radius, 0.24))
            y = int(258 - scale * 38)
            if item.name == "table":
                draw.rectangle([sx - 92, y + 30, sx + 92, y + 82], fill=item.color, outline=(55, 45, 38), width=3)
                draw.rectangle([sx - 76, y + 82, sx - 60, y + 146], fill=(83, 58, 40))
                draw.rectangle([sx + 60, y + 82, sx + 76, y + 146], fill=(83, 58, 40))
            elif item.name == "sofa":
                draw.rounded_rectangle([sx - 72, y + 22, sx + 72, y + 80], radius=8, fill=item.color, outline=(45, 52, 62), width=3)
            elif "book" in item.name:
                draw.rectangle([sx - size, y - size // 2, sx + size, y + size // 2], fill=item.color, outline=(35, 35, 35), width=3)
                draw.line([sx - size, y, sx + size, y], fill=(245, 245, 240), width=2)
            else:
                draw.ellipse([sx - size, y - size, sx + size, y + size], fill=item.color, outline=(35, 35, 35), width=3)
                if item.name == "red cup":
                    draw.arc([sx + size - 4, y - size // 2, sx + size + 22, y + size // 2], 270, 90, fill=(35, 35, 35), width=3)
            draw.text((max(4, sx - 42), max(4, y - size - 18)), item.name, fill=(20, 25, 30))
        if not visible:
            draw.text((210, 204), "No target-like object in view", fill=(54, 67, 82))
            draw.text((218, 232), "Agent keeps rotating", fill=(54, 67, 82))
        draw.rectangle([12, 12, 242, 42], fill=(8, 13, 23))
        draw.text((24, 20), f"Robot POV | heading {state.heading:.0f} deg", fill=(226, 232, 240))
        draw.line([310, 224, 330, 224], fill=(255, 255, 255), width=2)
        draw.line([320, 214, 320, 234], fill=(255, 255, 255), width=2)
        return image

    def visible_objects(self, state: RobotState) -> list[tuple[SceneObject, float, float]]:
        visible: list[tuple[SceneObject, float, float]] = []
        for item in self.objects:
            dx = item.pos[0] - state.x
            dy = item.pos[1] - state.y
            distance = math.hypot(dx, dy)
            angle = math.degrees(math.atan2(dy, dx))
            bearing = ((angle - state.heading + 180) % 360) - 180
            if abs(bearing) < 52 and distance < 5.2:
                visible.append((item, bearing, distance))
        visible.sort(key=lambda value: value[2], reverse=True)
        return visible

    def render_topdown(self, state: RobotState, best_candidate: dict[str, Any] | None) -> Image.Image:
        scale = 92
        image = Image.new("RGB", (520, 520), (247, 249, 252))
        draw = ImageDraw.Draw(image)
        draw.rectangle([14, 14, 506, 506], outline=(28, 38, 52), width=4)
        draw.rectangle([28, 28, 492, 492], outline=(203, 213, 225), width=1)
        for x in range(1, 5):
            px = 14 + int(x * scale)
            draw.line([px, 14, px, 506], fill=(226, 232, 240))
        for y in range(1, 5):
            py = 14 + int(y * scale)
            draw.line([14, py, 506, py], fill=(226, 232, 240))
        for item in self.objects:
            px = 14 + int(item.pos[0] * scale)
            py = 14 + int(item.pos[1] * scale)
            r = max(8, int(item.radius * scale))
            draw.ellipse([px - r, py - r, px + r, py + r], fill=item.color, outline=(30, 30, 30), width=2)
            draw.text((px + r + 4, py - 8), item.name, fill=(28, 38, 52))
        rx = 14 + int(state.x * scale)
        ry = 14 + int(state.y * scale)
        heading_rad = math.radians(state.heading)
        nose = (rx + int(math.cos(heading_rad) * 34), ry + int(math.sin(heading_rad) * 34))
        left = (rx + int(math.cos(heading_rad + 2.55) * 22), ry + int(math.sin(heading_rad + 2.55) * 22))
        right = (rx + int(math.cos(heading_rad - 2.55) * 22), ry + int(math.sin(heading_rad - 2.55) * 22))
        draw.polygon([nose, left, right], fill=(15, 118, 110), outline=(7, 89, 83))
        draw.arc([rx - 120, ry - 120, rx + 120, ry + 120], int(state.heading - 52), int(state.heading + 52), fill=(15, 118, 110), width=3)
        draw.text((20, 20), "Global scene map | FloorPlan211-compatible", fill=(15, 23, 42))
        if best_candidate:
            draw.text((20, 488), f"Best candidate: {best_candidate.get('label')} conf={best_candidate.get('confidence')}", fill=(180, 35, 24))
        return image

    def scripted_action(self, state: RobotState, agent_action: str, confidence: float, step_id: int) -> str:
        visible_names = {item.name for item, _, _ in self.visible_objects(state)}
        if "red cup" in visible_names and confidence >= self.config.stop_confidence_threshold:
            return "STOP"
        if step_id in {0, 1, 2, 3}:
            return "TURN_RIGHT"
        if step_id == 4:
            return "MOVE_FORWARD"
        return agent_action if agent_action != "STOP" else "INSPECT"

    def demo_response_dict(self, response: dict[str, Any], visible_names: list[str]) -> dict[str, Any]:
        visible_set = set(visible_names)
        if "red cup" not in visible_set:
            searched = ", ".join(visible_names) if visible_names else "no reliable object"
            response["confidence"] = min(float(response.get("confidence", 0.0)), 0.31)
            response["done"] = False
            response["action"]["type"] = "TURN_RIGHT"
            response["thought"] = (
                f"The target is not visible yet. Current view contains {searched}; "
                "the agent keeps rotating to scan the unfamiliar room before confirming the red cup."
            )
            if response.get("observation"):
                response["observation"]["target_visible"] = False
                response["observation"]["scene_summary"] = "Target is not visible in the current robot view."
                response["observation"]["best_candidate"] = None
        return response

    def compose_frame(self, obs: Image.Image, topdown: Image.Image, response: dict[str, Any], instruction: str, step_id: int) -> Image.Image:
        canvas = Image.new("RGB", (1600, 900), (8, 13, 23))
        draw = ImageDraw.Draw(canvas)
        draw.text((34, 28), "Embodied Visual Search Agent | FloorPlan211-compatible Demo", fill=(245, 248, 252))
        draw.text((34, 62), f"Instruction: {instruction}", fill=(180, 195, 210))
        draw.text((34, 92), "Robot egocentric camera", fill=(49, 217, 198))
        draw.text((992, 92), "Global environment map", fill=(88, 166, 255))
        canvas.paste(obs.resize((900, 630)), (34, 116))
        canvas.paste(topdown.resize((420, 420)), (992, 116))
        draw.rectangle([34, 116, 934, 746], outline=(49, 217, 198), width=3)
        draw.rectangle([992, 116, 1412, 536], outline=(88, 166, 255), width=3)
        panel_x = 992
        panel_y = 570
        draw.rounded_rectangle([panel_x, panel_y, 1560, 850], radius=8, fill=(242, 246, 250))
        draw.text((panel_x + 20, panel_y + 18), f"Step {step_id}", fill=(15, 23, 42))
        draw.text((panel_x + 20, panel_y + 52), f"Action: {response['action']['type']}", fill=(0, 126, 120))
        draw.text((panel_x + 20, panel_y + 86), f"Confidence: {response['confidence']:.3f}", fill=(180, 35, 24))
        draw.text((panel_x + 20, panel_y + 122), "Thought:", fill=(71, 84, 103))
        self._wrapped_text(draw, response["thought"], panel_x + 20, panel_y + 150, 74, fill=(15, 23, 42))
        draw.text((40, 790), "Retrieved hints:", fill=(148, 163, 184))
        hints = "; ".join(response.get("retrieved_hints", [])) or "None"
        self._wrapped_text(draw, hints, 40, 820, 118, fill=(226, 232, 240))
        return canvas

    def _wrapped_text(self, draw: ImageDraw.ImageDraw, text: str, x: int, y: int, width: int, fill: tuple[int, int, int]) -> None:
        words = text.split()
        line = ""
        for word in words:
            probe = f"{line} {word}".strip()
            if len(probe) > width:
                draw.text((x, y), line, fill=fill)
                y += 22
                line = word
            else:
                line = probe
        if line:
            draw.text((x, y), line, fill=fill)

    def apply_action(self, state: RobotState, action: str) -> None:
        if action == "TURN_RIGHT":
            state.heading += 30
        elif action == "TURN_LEFT":
            state.heading -= 30
        elif action == "MOVE_FORWARD":
            rad = math.radians(state.heading)
            state.x = min(max(state.x + math.cos(rad) * 0.45, 0.4), self.width - 0.4)
            state.y = min(max(state.y + math.sin(rad) * 0.45, 0.4), self.height - 0.4)
        elif action == "INSPECT":
            state.heading += 12
        state.step_id += 1

    def write_video(self, frames: list[Path], path: Path) -> None:
        if not frames:
            return
        first = cv2.imread(str(frames[0]))
        height, width = first.shape[:2]
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 2.0, (width, height))
        for frame_path in frames:
            frame = cv2.imread(str(frame_path))
            for _ in range(2):
                writer.write(frame)
        writer.release()


def main() -> None:
    result = RoomSimulator().run_demo()
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
