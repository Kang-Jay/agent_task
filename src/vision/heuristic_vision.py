from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from PIL import Image

from src.task.config import AgentConfig
from src.types.schema import Candidate, ObservationAnalysis


COLOR_PROTOTYPES: dict[str, tuple[int, int, int]] = {
    "red": (205, 55, 55),
    "green": (65, 150, 85),
    "blue": (65, 110, 210),
    "yellow": (220, 185, 55),
    "white": (230, 230, 220),
    "black": (35, 35, 35),
    "brown": (135, 90, 55),
    "gray": (145, 145, 145),
    "purple": (150, 75, 180),
    "orange": (230, 130, 50),
}


OBJECT_KEYWORDS = {
    "cup": ["cup", "mug", "杯", "杯子"],
    "book": ["book", "notebook", "书", "书本"],
    "plant": ["plant", "potted", "植物", "盆栽"],
    "remote": ["remote", "controller", "遥控器"],
    "bottle": ["bottle", "瓶", "瓶子"],
}


@dataclass(frozen=True)
class TargetDescriptor:
    text: str
    colors: list[str]
    objects: list[str]
    crop_signature: np.ndarray | None = None


class HeuristicVision:
    def __init__(self, config: AgentConfig):
        self.config = config
        self.vision_config = config.raw["vision"]

    def describe_target(self, instruction: str, target_crop: Image.Image | None = None) -> TargetDescriptor:
        lower = instruction.lower()
        colors = [name for name in COLOR_PROTOTYPES if name in lower or self._contains_chinese_color(name, instruction)]
        objects = [
            object_name
            for object_name, keywords in OBJECT_KEYWORDS.items()
            if any(keyword.lower() in lower or keyword in instruction for keyword in keywords)
        ]
        crop_signature = self._image_signature(target_crop) if target_crop is not None else None
        return TargetDescriptor(text=instruction, colors=colors, objects=objects, crop_signature=crop_signature)

    def analyze(
        self,
        image: Image.Image,
        instruction: str,
        target_crop: Image.Image | None = None,
    ) -> ObservationAnalysis:
        image = image.convert("RGB").resize(self.config.image_size)
        target = self.describe_target(instruction, target_crop)
        candidates = self._extract_candidates(image, target)
        candidates.sort(key=lambda item: item.confidence, reverse=True)
        best = candidates[0] if candidates else None
        target_visible = bool(best and best.confidence >= self.config.target_visible_threshold)
        summary = self._scene_summary(candidates, target_visible)
        return ObservationAnalysis(
            image_size=(image.width, image.height),
            scene_summary=summary,
            candidates=candidates[:6],
            best_candidate=best,
            target_visible=target_visible,
        )

    def _extract_candidates(self, image: Image.Image, target: TargetDescriptor) -> list[Candidate]:
        rows = int(self.vision_config["grid_rows"])
        cols = int(self.vision_config["grid_cols"])
        arr = np.asarray(image).astype(np.float32)
        candidates: list[Candidate] = []

        target_colors = target.colors or list(COLOR_PROTOTYPES)
        for color_name in target_colors:
            color_candidates = self._extract_color_regions(arr, color_name, target, rows, cols)
            candidates.extend(color_candidates)

        if target.crop_signature is not None and not candidates:
            candidates.extend(self._extract_sliding_crop_regions(arr, target, rows, cols))

        if candidates:
            return self._dedupe_candidates(candidates)

        cell_h = image.height // rows
        cell_w = image.width // cols
        for row in range(rows):
            for col in range(cols):
                left = col * cell_w
                top = row * cell_h
                right = image.width if col == cols - 1 else (col + 1) * cell_w
                bottom = image.height if row == rows - 1 else (row + 1) * cell_h
                patch = arr[top:bottom, left:right]
                mean = patch.reshape(-1, 3).mean(axis=0)
                color_name, color_strength = self._best_color(mean)
                target_score = self._score_patch(patch, target, color_name, color_strength, row, col, rows, cols)
                if target_score < float(self.vision_config["min_candidate_area_ratio"]):
                    continue
                label = self._label_for_target(target, color_name)
                region = self._region_name(row, col, rows, cols)
                candidates.append(
                    Candidate(
                        label=label,
                        bbox=[left, top, right, bottom],
                        confidence=round(float(target_score), 3),
                        color_name=color_name,
                        region=region,
                        reason=self._candidate_reason(target, color_name, region),
                    )
                )
        return candidates

    def _extract_color_regions(
        self,
        arr: np.ndarray,
        color_name: str,
        target: TargetDescriptor,
        rows: int,
        cols: int,
    ) -> list[Candidate]:
        prototype = np.array(COLOR_PROTOTYPES[color_name], dtype=np.float32)
        distance = np.linalg.norm(arr - prototype, axis=2)
        mask = self._color_mask(arr, color_name, distance)
        min_pixels = max(24, int(arr.shape[0] * arr.shape[1] * float(self.vision_config["min_candidate_area_ratio"]) * 0.08))
        regions = self._connected_components(mask, min_pixels=min_pixels)
        candidates: list[Candidate] = []
        for left, top, right, bottom, area in regions:
            patch = arr[top:bottom, left:right]
            if patch.size == 0:
                continue
            component_mask = mask[top:bottom, left:right]
            component_pixels = patch[component_mask]
            if component_pixels.size == 0:
                continue
            mean = component_pixels.reshape(-1, 3).mean(axis=0)
            _, color_strength = self._best_color(mean)
            center_y = (top + bottom) / 2
            center_x = (left + right) / 2
            row = min(rows - 1, max(0, int(center_y / (arr.shape[0] / rows))))
            col = min(cols - 1, max(0, int(center_x / (arr.shape[1] / cols))))
            score = self._score_patch(patch, target, color_name, color_strength, row, col, rows, cols)
            area_bonus = min(0.12, area / float(arr.shape[0] * arr.shape[1]) * 4.0)
            score = min(1.0, score + area_bonus)
            region = self._region_name(row, col, rows, cols)
            candidates.append(
                Candidate(
                    label=self._label_for_target(target, color_name),
                    bbox=[int(left), int(top), int(right), int(bottom)],
                    confidence=round(float(score), 3),
                    color_name=color_name,
                    region=region,
                    reason=self._candidate_reason(target, color_name, region),
                )
            )
        return candidates

    def _color_mask(
        self,
        arr: np.ndarray,
        color_name: str,
        distance: np.ndarray,
    ) -> np.ndarray:
        red = arr[:, :, 0]
        green = arr[:, :, 1]
        blue = arr[:, :, 2]
        base = distance < 92.0
        dominance = {
            "red": (red >= 145) & (red >= green * 1.35) & (red >= blue * 1.25),
            "green": (green >= 105) & (green >= red * 1.20) & (green >= blue * 1.12),
            "blue": (blue >= 135) & (blue >= red * 1.25) & (blue >= green * 1.20),
            "yellow": (red >= 150) & (green >= 125) & (blue <= green * 0.75),
            "orange": (red >= 160) & (green >= 75) & (green <= red * 0.78) & (blue <= green),
            "purple": (red >= 100) & (blue >= 110) & (green <= red * 0.78),
        }
        return base & dominance.get(color_name, np.ones_like(base, dtype=bool))

    def _extract_sliding_crop_regions(
        self,
        arr: np.ndarray,
        target: TargetDescriptor,
        rows: int,
        cols: int,
    ) -> list[Candidate]:
        patch_size = int(self.vision_config["candidate_patch_size"])
        stride = patch_size // 2
        candidates: list[Candidate] = []
        for top in range(0, max(1, arr.shape[0] - patch_size + 1), stride):
            for left in range(0, max(1, arr.shape[1] - patch_size + 1), stride):
                patch = arr[top : top + patch_size, left : left + patch_size]
                color_name, color_strength = self._best_color(patch.reshape(-1, 3).mean(axis=0))
                row = min(rows - 1, int((top + patch_size / 2) / (arr.shape[0] / rows)))
                col = min(cols - 1, int((left + patch_size / 2) / (arr.shape[1] / cols)))
                score = self._score_patch(patch, target, color_name, color_strength, row, col, rows, cols)
                if score >= self.config.target_visible_threshold:
                    region = self._region_name(row, col, rows, cols)
                    candidates.append(
                        Candidate(
                            label=self._label_for_target(target, color_name),
                            bbox=[left, top, left + patch_size, top + patch_size],
                            confidence=round(float(score), 3),
                            color_name=color_name,
                            region=region,
                            reason=self._candidate_reason(target, color_name, region),
                        )
                    )
        return candidates

    def _connected_components(self, mask: np.ndarray, min_pixels: int) -> list[tuple[int, int, int, int, int]]:
        height, width = mask.shape
        visited = np.zeros_like(mask, dtype=bool)
        components: list[tuple[int, int, int, int, int]] = []
        for y in range(height):
            for x in range(width):
                if not mask[y, x] or visited[y, x]:
                    continue
                queue: deque[tuple[int, int]] = deque([(x, y)])
                visited[y, x] = True
                min_x = max_x = x
                min_y = max_y = y
                area = 0
                while queue:
                    cx, cy = queue.popleft()
                    area += 1
                    min_x = min(min_x, cx)
                    max_x = max(max_x, cx)
                    min_y = min(min_y, cy)
                    max_y = max(max_y, cy)
                    for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                        if nx < 0 or ny < 0 or nx >= width or ny >= height:
                            continue
                        if visited[ny, nx] or not mask[ny, nx]:
                            continue
                        visited[ny, nx] = True
                        queue.append((nx, ny))
                if area >= min_pixels:
                    components.append((min_x, min_y, max_x + 1, max_y + 1, area))
        return components

    def _dedupe_candidates(self, candidates: list[Candidate]) -> list[Candidate]:
        deduped: list[Candidate] = []
        for candidate in sorted(candidates, key=lambda item: item.confidence, reverse=True):
            if all(self._iou(candidate.bbox, existing.bbox) < 0.35 for existing in deduped):
                deduped.append(candidate)
        return deduped

    def _iou(self, left: list[int], right: list[int]) -> float:
        x1 = max(left[0], right[0])
        y1 = max(left[1], right[1])
        x2 = min(left[2], right[2])
        y2 = min(left[3], right[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        left_area = max(0, left[2] - left[0]) * max(0, left[3] - left[1])
        right_area = max(0, right[2] - right[0]) * max(0, right[3] - right[1])
        denom = left_area + right_area - inter
        return inter / denom if denom else 0.0

    def _score_patch(
        self,
        patch: np.ndarray,
        target: TargetDescriptor,
        color_name: str,
        color_strength: float,
        row: int,
        col: int,
        rows: int,
        cols: int,
    ) -> float:
        color_weight = float(self.vision_config["color_match_weight"])
        text_weight = float(self.vision_config["text_match_weight"])
        center_weight = float(self.vision_config["center_prior_weight"])

        color_score = color_strength if not target.colors else (color_strength if color_name in target.colors else 0.08)
        crop_score = 0.0
        if target.crop_signature is not None:
            crop_score = self._signature_similarity(self._array_signature(patch), target.crop_signature)
            color_score = max(color_score, crop_score)

        object_score = 0.45 if target.objects else 0.25
        if target.colors and color_name in target.colors:
            object_score += 0.22
        if target.crop_signature is not None:
            object_score += 0.18 * crop_score

        center_row = (rows - 1) / 2
        center_col = (cols - 1) / 2
        max_dist = max(center_row + center_col, 1)
        center_score = 1.0 - ((abs(row - center_row) + abs(col - center_col)) / max_dist)

        return max(0.0, min(1.0, color_weight * color_score + text_weight * object_score + center_weight * center_score))

    def _scene_summary(self, candidates: Iterable[Candidate], target_visible: bool) -> str:
        candidates = list(candidates)
        if not candidates:
            return "No reliable target-like region is visible; continue exploring."
        best = max(candidates, key=lambda item: item.confidence)
        status = "Target-like evidence is strong" if target_visible else "Target-like evidence is weak"
        return f"{status}; strongest candidate is a {best.color_name} region at {best.region}."

    def _candidate_reason(self, target: TargetDescriptor, color_name: str, region: str) -> str:
        pieces = [f"{color_name} visual signature", f"located in {region}"]
        if target.colors:
            pieces.append("matches requested color" if color_name in target.colors else "does not match requested color")
        if target.crop_signature is not None:
            pieces.append("compared with clicked target crop")
        return "; ".join(pieces)

    def _label_for_target(self, target: TargetDescriptor, color_name: str) -> str:
        obj = target.objects[0] if target.objects else "object"
        return f"{color_name} {obj}"

    def _best_color(self, rgb: np.ndarray) -> tuple[str, float]:
        distances = {}
        for name, proto in COLOR_PROTOTYPES.items():
            proto_arr = np.array(proto, dtype=np.float32)
            distances[name] = float(np.linalg.norm(rgb - proto_arr))
        best = min(distances, key=distances.get)
        strength = 1.0 - min(distances[best] / 320.0, 1.0)
        return best, max(0.0, strength)

    def _image_signature(self, image: Image.Image | None) -> np.ndarray | None:
        if image is None:
            return None
        return self._array_signature(np.asarray(image.convert("RGB")).astype(np.float32))

    def _array_signature(self, arr: np.ndarray) -> np.ndarray:
        flat = arr.reshape(-1, 3)
        return flat.mean(axis=0) / 255.0

    def _signature_similarity(self, left: np.ndarray, right: np.ndarray) -> float:
        distance = float(np.linalg.norm(left - right))
        return max(0.0, 1.0 - distance / 1.35)

    def _region_name(self, row: int, col: int, rows: int, cols: int) -> str:
        vertical = ["upper", "middle", "lower"][min(row, 2)] if rows == 3 else f"row {row + 1}"
        horizontal = ["left", "center", "right"][min(col, 2)] if cols == 3 else f"column {col + 1}"
        return f"{vertical} {horizontal}"

    def _contains_chinese_color(self, name: str, text: str) -> bool:
        mapping = {
            "red": "红",
            "green": "绿",
            "blue": "蓝",
            "yellow": "黄",
            "white": "白",
            "black": "黑",
            "brown": "棕",
            "gray": "灰",
            "purple": "紫",
            "orange": "橙",
        }
        return mapping[name] in text
