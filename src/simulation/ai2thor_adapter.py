from __future__ import annotations

import json
import math
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

from src.agent.controller import EmbodiedSearchAgent
from src.simulation.room_simulator import DemoResult, DemoStep
from src.task.config import ROOT, load_config
from src.types.schema import AgentRequest
from src.vision.image_io import image_to_data_url


AI2THOR_OUTPUT_DIR = ROOT / "docs" / "ai2thor_outputs"
AI2THOR_FRAME_DIR = AI2THOR_OUTPUT_DIR / "frames"


TARGET_ALIASES: dict[str, list[str]] = {
    "pillow": ["pillow", "枕头"],
    "sofa": ["sofa", "couch", "沙发"],
    "television": ["television", "tv", "电视", "电视机"],
    "floorlamp": ["floorlamp", "floor lamp", "lamp", "灯", "落地灯"],
    "remotecontrol": ["remotecontrol", "remote control", "remote", "遥控器"],
    "sidetable": ["sidetable", "side table"],
    "diningtable": ["diningtable", "dining table", "table", "桌子"],
    "coffeetable": ["coffeetable", "coffee table", "table", "桌子"],
    "armchair": ["armchair", "arm chair", "chair", "椅子"],
    "garbagecan": ["garbagecan", "garbage can", "trash can", "bin", "垃圾桶"],
}

STRUCTURAL_OBJECTS = {"floor", "wall", "ceiling", "window", "painting"}


@dataclass(frozen=True)
class SimulatorStatus:
    available: bool
    backend: str
    scene: str
    message: str
    diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "backend": self.backend,
            "scene": self.scene,
            "message": self.message,
            "diagnostics": self.diagnostics,
        }


class AI2ThorVisualSearchDemo:
    """Runs the embodied visual-search loop in AI2-THOR when the runtime is available."""

    def __init__(self, scene: str = "FloorPlan211"):
        self.config = load_config()
        self.agent = EmbodiedSearchAgent(self.config)
        self.scene = scene

    @staticmethod
    def status(scene: str = "FloorPlan211") -> SimulatorStatus:
        diagnostics: dict[str, Any] = {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "ai2thor_import": False,
            "vulkaninfo": shutil.which("vulkaninfo"),
            "nvidia_smi": shutil.which("nvidia-smi"),
            "wsl": bool(os.environ.get("WSL_DISTRO_NAME")),
        }
        try:
            import ai2thor  # type: ignore

            diagnostics["ai2thor_import"] = True
            diagnostics["ai2thor_version"] = getattr(ai2thor, "__version__", "unknown")
        except Exception as exc:
            diagnostics["ai2thor_error"] = repr(exc)
            return SimulatorStatus(
                available=False,
                backend="ai2thor",
                scene=scene,
                message="AI2-THOR is not importable in this Python environment.",
                diagnostics=diagnostics,
            )

        if platform.system().lower() == "windows":
            return SimulatorStatus(
                available=False,
                backend="ai2thor",
                scene=scene,
                message=(
                    "AI2-THOR PyPI builds do not provide a working native Windows Unity build here; "
                    "run through WSL2/Linux with Vulkan configured."
                ),
                diagnostics=diagnostics,
            )

        return SimulatorStatus(
            available=True,
            backend="ai2thor",
            scene=scene,
            message="AI2-THOR import succeeded; runtime launch will be tested when the demo starts.",
            diagnostics=diagnostics,
        )

    def run_demo(self, instruction: str, max_steps: int = 8) -> DemoResult:
        from ai2thor.controller import Controller  # type: ignore
        from ai2thor.platform import CloudRendering  # type: ignore

        max_steps = min(int(max_steps), self.config.max_steps)
        AI2THOR_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        AI2THOR_FRAME_DIR.mkdir(parents=True, exist_ok=True)
        for stale in AI2THOR_FRAME_DIR.glob("ai2thor_*.png"):
            stale.unlink()

        controller = None
        result = DemoResult()
        session_id = "ai2thor-demo"
        self.agent.reset(session_id)
        try:
            controller = Controller(
                scene=self.scene,
                platform=CloudRendering,
                width=960,
                height=540,
                quality="Low",
                gridSize=0.25,
                rotateStepDegrees=self.config.raw["agent"]["default_turn_angle_degrees"],
                snapToGrid=False,
                renderInstanceSegmentation=True,
            )
            event = controller.last_event
            confirmed_target_steps = 0
            for step_id in range(max_steps):
                obs = Image.fromarray(event.frame).convert("RGB")
                obs_path = AI2THOR_FRAME_DIR / f"ai2thor_obs_{step_id:02d}.png"
                obs.save(obs_path)
                response = self.agent.step(
                    AgentRequest(
                        session_id=session_id,
                        instruction=instruction,
                        observation_image=image_to_data_url(obs),
                        step_id=step_id,
                    )
                )
                response_dict = response.to_dict()
                grounded_target = self._ground_target_from_segmentation(event, instruction)
                if grounded_target:
                    confirmed_target_steps += 1
                    action_type = "STOP" if confirmed_target_steps >= 2 else "INSPECT"
                    self._apply_grounded_target(response_dict, grounded_target, action_type, confirmed_target_steps)
                else:
                    confirmed_target_steps = 0
                    action_type = response_dict["action"]["type"]
                    if action_type in {"STOP", "ASK_CLARIFY"}:
                        action_type = self._search_action(step_id)
                        self._apply_search_response(response_dict, action_type)
                visible_objects = self._visible_objects(event.metadata)
                if grounded_target:
                    visible_objects = sorted(set(visible_objects + [f"{grounded_target['object_type']} (segmented)"]))
                topdown = self._render_topdown(event.metadata, response_dict["observation"].get("best_candidate"))
                topdown_path = AI2THOR_FRAME_DIR / f"ai2thor_topdown_{step_id:02d}.png"
                topdown.save(topdown_path)
                frame = self._compose_frame(obs, topdown, response_dict, instruction, step_id, visible_objects)
                frame_path = AI2THOR_FRAME_DIR / f"ai2thor_frame_{step_id:02d}.png"
                frame.save(frame_path)
                done = action_type == "STOP"
                result.steps.append(
                    DemoStep(
                        frame_path=str(frame_path.relative_to(ROOT)),
                        observation_path=str(obs_path.relative_to(ROOT)),
                        topdown_path=str(topdown_path.relative_to(ROOT)),
                        thought=response_dict["thought"],
                        action=action_type,
                        confidence=response_dict["confidence"],
                        done=done,
                        robot=self._robot_state(event.metadata),
                        best_candidate=response_dict["observation"].get("best_candidate"),
                        visible_objects=visible_objects[:10],
                        backend="ai2thor",
                        scene=self.scene,
                    )
                )
                if done:
                    break
                event = controller.step(action=self._map_action(action_type))

            video_path = AI2THOR_OUTPUT_DIR / "ai2thor_visual_search_demo.mp4"
            self._write_video([ROOT / step.frame_path for step in result.steps], video_path)
            summary_path = AI2THOR_OUTPUT_DIR / "ai2thor_demo_summary.json"
            result.video_path = str(video_path.relative_to(ROOT))
            result.summary_path = str(summary_path.relative_to(ROOT))
            summary_path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
            return result
        finally:
            if controller is not None:
                controller.stop()

    def _map_action(self, action_type: str) -> str:
        return {
            "MOVE_FORWARD": "MoveAhead",
            "TURN_LEFT": "RotateLeft",
            "TURN_RIGHT": "RotateRight",
            "LOOK_UP": "LookUp",
            "LOOK_DOWN": "LookDown",
            "INSPECT": "Pass",
            "STOP": "Pass",
        }.get(action_type, "Pass")

    def _search_action(self, step_id: int) -> str:
        if step_id > 0 and step_id % 4 == 3:
            return "MOVE_FORWARD"
        return "TURN_RIGHT"

    def _ground_target_from_segmentation(self, event: Any, instruction: str) -> dict[str, Any] | None:
        masks = getattr(event, "instance_masks", None) or {}
        if not masks:
            return None
        terms = self._target_terms(instruction)
        if not terms:
            return None
        metadata_by_id = {
            str(item.get("objectId") or item.get("name") or ""): item
            for item in event.metadata.get("objects", [])
        }
        best: dict[str, Any] | None = None
        for object_id, raw_mask in masks.items():
            object_type = self._object_type(object_id, metadata_by_id.get(str(object_id)))
            normalized_type = self._normalize(object_type)
            if normalized_type in STRUCTURAL_OBJECTS or not self._matches_target(normalized_type, str(object_id), terms):
                continue
            mask = np.asarray(raw_mask).astype(bool)
            ys, xs = np.nonzero(mask)
            if len(xs) == 0:
                continue
            height, width = mask.shape[:2]
            bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
            area_ratio = float(len(xs)) / float(max(width * height, 1))
            center_x = (bbox[0] + bbox[2]) / 2.0 / max(width, 1)
            center_y = (bbox[1] + bbox[3]) / 2.0 / max(height, 1)
            center_distance = min(((center_x - 0.5) ** 2 + (center_y - 0.5) ** 2) ** 0.5 / 0.707, 1.0)
            confidence = min(0.98, 0.74 + min(area_ratio * 9.0, 0.18) + (1.0 - center_distance) * 0.08)
            candidate = {
                "label": object_type,
                "object_type": object_type,
                "object_id": str(object_id),
                "bbox": bbox,
                "confidence": round(confidence, 3),
                "color_name": "segmentation",
                "region": self._screen_region(center_x, center_y),
                "reason": "AI2-THOR instance segmentation matched the requested target object",
                "image_size": [width, height],
                "area_ratio": round(area_ratio, 5),
            }
            if best is None or candidate["confidence"] > best["confidence"]:
                best = candidate
        return best

    def _apply_grounded_target(self, response: dict[str, Any], target: dict[str, Any], action_type: str, confirmed_steps: int) -> None:
        done = action_type == "STOP"
        response["action"]["type"] = action_type
        response["action"]["args"] = {"reason": "confirmed by AI2-THOR instance segmentation"}
        response["confidence"] = target["confidence"]
        response["done"] = done
        response["thought"] = (
            f"AI2-THOR segmentation grounds the requested target as {target['object_type']} "
            f"at {target['region']} with confidence {target['confidence']:.2f}. "
            f"{'The agent stops after confirmation.' if done else 'The agent inspects once before final confirmation.'}"
        )
        response["observation"]["image_size"] = target["image_size"]
        response["observation"]["target_visible"] = True
        response["observation"]["scene_summary"] = (
            f"Target object {target['object_type']} is grounded by simulator instance segmentation."
        )
        candidate = {
            "label": target["label"],
            "bbox": target["bbox"],
            "confidence": target["confidence"],
            "color_name": target["color_name"],
            "region": target["region"],
            "reason": target["reason"],
        }
        response["observation"]["best_candidate"] = candidate
        response["observation"]["candidates"] = [candidate]

    def _apply_search_response(self, response: dict[str, Any], action_type: str) -> None:
        response["action"]["type"] = action_type
        response["action"]["args"] = {"reason": "target not grounded in AI2-THOR segmentation yet"}
        response["confidence"] = min(float(response.get("confidence", 0.0)), self.config.target_visible_threshold - 0.01)
        response["done"] = False
        response["thought"] = (
            "The target is not confirmed by AI2-THOR instance segmentation yet, "
            f"so the agent executes {action_type} to continue the embodied search."
        )
        response["observation"]["target_visible"] = False
        response["observation"]["scene_summary"] = "No simulator-grounded target object is visible in the current robot view."
        response["observation"]["best_candidate"] = None
        response["observation"]["candidates"] = []

    def _target_terms(self, instruction: str) -> list[str]:
        lower = instruction.lower()
        matches: list[tuple[int, str]] = []
        for normalized, aliases in TARGET_ALIASES.items():
            for alias in aliases:
                index = lower.find(alias.lower())
                if index >= 0:
                    matches.append((index, normalized))
                    break
        return [item[1] for item in sorted(matches)]

    def _matches_target(self, object_type: str, object_id: str, terms: list[str]) -> bool:
        normalized_id = self._normalize(object_id)
        return any(term in object_type or term in normalized_id or object_type in term for term in terms)

    def _object_type(self, object_id: str, metadata: dict[str, Any] | None) -> str:
        if metadata and metadata.get("objectType"):
            return str(metadata["objectType"])
        return object_id.split("|", 1)[0]

    def _normalize(self, text: str) -> str:
        return "".join(ch for ch in text.lower() if ch.isalnum())

    def _screen_region(self, center_x: float, center_y: float) -> str:
        col = "left" if center_x < 1 / 3 else "right" if center_x > 2 / 3 else "center"
        row = "top" if center_y < 1 / 3 else "bottom" if center_y > 2 / 3 else "middle"
        return f"{row} {col}"

    def _visible_objects(self, metadata: dict[str, Any]) -> list[str]:
        names = []
        for item in metadata.get("objects", []):
            if item.get("visible"):
                names.append(str(item.get("objectType") or item.get("name") or "object"))
        return sorted(set(names))

    def _robot_state(self, metadata: dict[str, Any]) -> dict[str, float]:
        agent = metadata.get("agent", {})
        position = agent.get("position", {})
        rotation = agent.get("rotation", {})
        return {
            "x": float(position.get("x", 0.0)),
            "y": float(position.get("z", 0.0)),
            "heading": float(rotation.get("y", 0.0)),
        }

    def _render_topdown(self, metadata: dict[str, Any], best_candidate: dict[str, Any] | None) -> Image.Image:
        image = Image.new("RGB", (520, 520), (245, 247, 250))
        draw = ImageDraw.Draw(image)
        draw.rectangle([16, 16, 504, 504], outline=(20, 33, 48), width=4)
        positions = []
        for item in metadata.get("objects", []):
            pos = item.get("position") or {}
            if "x" in pos and "z" in pos:
                positions.append((float(pos["x"]), float(pos["z"]), str(item.get("objectType", "object")), bool(item.get("visible"))))
        xs = [p[0] for p in positions] or [0.0]
        zs = [p[1] for p in positions] or [0.0]
        min_x, max_x = min(xs) - 0.5, max(xs) + 0.5
        min_z, max_z = min(zs) - 0.5, max(zs) + 0.5

        def project(x: float, z: float) -> tuple[int, int]:
            px = 28 + int((x - min_x) / max(max_x - min_x, 0.1) * 464)
            py = 28 + int((z - min_z) / max(max_z - min_z, 0.1) * 464)
            return px, py

        for x, z, label, visible in positions[:80]:
            px, py = project(x, z)
            color = (238, 95, 70) if visible else (150, 162, 176)
            draw.ellipse([px - 5, py - 5, px + 5, py + 5], fill=color)
            if visible:
                draw.text((px + 7, py - 6), label[:16], fill=(33, 43, 55))

        agent = metadata.get("agent", {})
        pos = agent.get("position", {})
        rot = math.radians(float(agent.get("rotation", {}).get("y", 0.0)))
        ax, ay = project(float(pos.get("x", 0.0)), float(pos.get("z", 0.0)))
        nose = (ax + int(math.sin(rot) * 28), ay + int(math.cos(rot) * 28))
        left = (ax + int(math.sin(rot + 2.5) * 18), ay + int(math.cos(rot + 2.5) * 18))
        right = (ax + int(math.sin(rot - 2.5) * 18), ay + int(math.cos(rot - 2.5) * 18))
        draw.polygon([nose, left, right], fill=(0, 126, 167), outline=(0, 70, 96))
        draw.text((24, 24), f"AI2-THOR {self.scene} global map", fill=(15, 23, 42))
        if best_candidate:
            draw.text((24, 486), f"Best candidate: {best_candidate.get('label')} {best_candidate.get('confidence')}", fill=(170, 35, 24))
        return image

    def _compose_frame(
        self,
        obs: Image.Image,
        topdown: Image.Image,
        response: dict[str, Any],
        instruction: str,
        step_id: int,
        visible_objects: list[str],
    ) -> Image.Image:
        canvas = Image.new("RGB", (1600, 900), (8, 13, 23))
        draw = ImageDraw.Draw(canvas)
        draw.text((34, 28), f"AI2-THOR x Agent | {self.scene}", fill=(245, 248, 252))
        draw.text((34, 62), f"Instruction: {instruction}", fill=(180, 195, 210))
        canvas.paste(obs.resize((900, 506)), (34, 116))
        canvas.paste(topdown.resize((420, 420)), (980, 116))
        draw.rectangle([34, 116, 934, 622], outline=(57, 217, 198), width=3)
        draw.rectangle([980, 116, 1400, 536], outline=(88, 166, 255), width=3)
        panel_x = 980
        panel_y = 570
        draw.rounded_rectangle([panel_x, panel_y, 1560, 850], radius=8, fill=(242, 246, 250))
        draw.text((panel_x + 20, panel_y + 18), f"Step {step_id}", fill=(15, 23, 42))
        draw.text((panel_x + 20, panel_y + 52), f"Action: {response['action']['type']}", fill=(0, 126, 120))
        draw.text((panel_x + 20, panel_y + 86), f"Confidence: {response['confidence']:.3f}", fill=(190, 45, 35))
        self._wrapped_text(draw, "Thought: " + response["thought"], panel_x + 20, panel_y + 126, 74, fill=(22, 31, 43))
        visible = ", ".join(visible_objects[:8]) or "None"
        self._wrapped_text(draw, "Visible: " + visible, 34, 656, 110, fill=(220, 230, 242))
        return canvas

    def _wrapped_text(self, draw: ImageDraw.ImageDraw, text: str, x: int, y: int, width: int, fill: tuple[int, int, int]) -> None:
        line = ""
        for word in text.split():
            probe = f"{line} {word}".strip()
            if len(probe) > width:
                draw.text((x, y), line, fill=fill)
                y += 23
                line = word
            else:
                line = probe
        if line:
            draw.text((x, y), line, fill=fill)

    def _write_video(self, frames: list[Path], path: Path) -> None:
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


def ai2thor_environment_report() -> dict[str, Any]:
    status = AI2ThorVisualSearchDemo.status().to_dict()
    player_log = Path.home() / ".config" / "unity3d" / "Allen Institute for Artificial Intelligence" / "AI2-THOR" / "Player.log"
    if player_log.exists():
        try:
            status["player_log_tail"] = player_log.read_text(encoding="utf-8", errors="replace").splitlines()[-40:]
        except OSError as exc:
            status["player_log_error"] = repr(exc)
    if shutil.which("vulkaninfo"):
        try:
            probe = subprocess.run(["vulkaninfo", "--summary"], capture_output=True, text=True, timeout=10, check=False)
            status["vulkan_summary"] = (probe.stdout + probe.stderr).splitlines()[:80]
        except Exception as exc:
            status["vulkan_summary_error"] = repr(exc)
    return status
