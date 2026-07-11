from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


class ManifestValidationError(ValueError):
    """Raised when an evaluation manifest is not reproducible or well formed."""


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ManifestValidationError(f"{field} must be an object")
    return value


def _non_empty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ManifestValidationError(f"{field} must be a non-empty string")
    return value.strip()


def _finite_float(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ManifestValidationError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ManifestValidationError(f"{field} must be a finite number")
    return result


def _non_negative_float(value: Any, field: str) -> float:
    result = _finite_float(value, field)
    if result < 0:
        raise ManifestValidationError(f"{field} must be non-negative")
    return result


def _string_list(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ManifestValidationError(f"{field} must be an array")
    result = tuple(_non_empty_string(item, f"{field}[]") for item in value)
    if len(set(result)) != len(result):
        raise ManifestValidationError(f"{field} must not contain duplicates")
    return result


@dataclass(frozen=True)
class InitialPose:
    position: dict[str, float]
    rotation: dict[str, float]
    horizon: float
    standing: bool

    @classmethod
    def from_dict(cls, raw: Any, field: str) -> InitialPose:
        data = _mapping(raw, field)
        position_raw = _mapping(data.get("position"), f"{field}.position")
        rotation_raw = _mapping(data.get("rotation"), f"{field}.rotation")
        position = {
            axis: _finite_float(position_raw.get(axis), f"{field}.position.{axis}")
            for axis in ("x", "y", "z")
        }
        rotation = {
            axis: _finite_float(rotation_raw.get(axis), f"{field}.rotation.{axis}")
            for axis in ("x", "y", "z")
        }
        horizon = _finite_float(data.get("horizon"), f"{field}.horizon")
        if not -90.0 <= horizon <= 90.0:
            raise ManifestValidationError(f"{field}.horizon must be between -90 and 90")
        standing = data.get("standing")
        if not isinstance(standing, bool):
            raise ManifestValidationError(f"{field}.standing must be a boolean")
        return cls(
            position=position,
            rotation=rotation,
            horizon=horizon,
            standing=standing,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "position": dict(self.position),
            "rotation": dict(self.rotation),
            "horizon": self.horizon,
            "standing": self.standing,
        }


@dataclass(frozen=True)
class TaskSpec:
    instruction: str
    task_type: str
    target: dict[str, str]
    required_actions: tuple[str, ...]
    allows_approximate_success: bool

    @classmethod
    def from_dict(cls, raw: Any, field: str) -> TaskSpec:
        data = _mapping(raw, field)
        target_raw = _mapping(data.get("target"), f"{field}.target")
        target = {
            key: _non_empty_string(value, f"{field}.target.{key}")
            for key, value in sorted(target_raw.items())
        }
        if "object_type" not in target:
            raise ManifestValidationError(f"{field}.target.object_type is required")
        allows_approximate = data.get("allows_approximate_success", False)
        if not isinstance(allows_approximate, bool):
            raise ManifestValidationError(
                f"{field}.allows_approximate_success must be a boolean"
            )
        return cls(
            instruction=_non_empty_string(data.get("instruction"), f"{field}.instruction"),
            task_type=_non_empty_string(data.get("task_type"), f"{field}.task_type"),
            target=target,
            required_actions=_string_list(
                data.get("required_actions", []),
                f"{field}.required_actions",
            ),
            allows_approximate_success=allows_approximate,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "instruction": self.instruction,
            "task_type": self.task_type,
            "target": dict(self.target),
            "required_actions": list(self.required_actions),
            "allows_approximate_success": self.allows_approximate_success,
        }


@dataclass(frozen=True)
class ReferenceSpec:
    optimal_path_length_meters: float | None
    source: str
    allowed_error_meters: float | None = None

    @classmethod
    def from_dict(cls, raw: Any, field: str) -> ReferenceSpec:
        data = _mapping(raw, field)
        optimal_raw = data.get("optimal_path_length_meters")
        optimal = (
            None
            if optimal_raw is None
            else _non_negative_float(optimal_raw, f"{field}.optimal_path_length_meters")
        )
        allowed_raw = data.get("allowed_error_meters")
        allowed = (
            None
            if allowed_raw is None
            else _non_negative_float(allowed_raw, f"{field}.allowed_error_meters")
        )
        return cls(
            optimal_path_length_meters=optimal,
            source=_non_empty_string(data.get("source"), f"{field}.source"),
            allowed_error_meters=allowed,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "optimal_path_length_meters": self.optimal_path_length_meters,
            "source": self.source,
            "allowed_error_meters": self.allowed_error_meters,
        }


@dataclass(frozen=True)
class EpisodeSpec:
    episode_id: str
    pair_id: str
    group: str
    split: str
    scene: str
    seed: int
    initial_pose: InitialPose
    task: TaskSpec
    reference: ReferenceSpec
    result_file: str

    @classmethod
    def from_dict(cls, raw: Any, index: int) -> EpisodeSpec:
        field = f"episodes[{index}]"
        data = _mapping(raw, field)
        seed = data.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ManifestValidationError(f"{field}.seed must be a non-negative integer")
        result_file = _non_empty_string(data.get("result_file"), f"{field}.result_file")
        path = PurePosixPath(result_file.replace("\\", "/"))
        if path.is_absolute() or ".." in path.parts or path.suffix.lower() != ".json":
            raise ManifestValidationError(
                f"{field}.result_file must be a relative JSON path without '..'"
            )
        return cls(
            episode_id=_non_empty_string(data.get("episode_id"), f"{field}.episode_id"),
            pair_id=_non_empty_string(data.get("pair_id"), f"{field}.pair_id"),
            group=_non_empty_string(data.get("group"), f"{field}.group"),
            split=_non_empty_string(data.get("split"), f"{field}.split"),
            scene=_non_empty_string(data.get("scene"), f"{field}.scene"),
            seed=seed,
            initial_pose=InitialPose.from_dict(
                data.get("initial_pose"),
                f"{field}.initial_pose",
            ),
            task=TaskSpec.from_dict(data.get("task"), f"{field}.task"),
            reference=ReferenceSpec.from_dict(
                data.get("reference"),
                f"{field}.reference",
            ),
            result_file=path.as_posix(),
        )

    def comparison_key(self) -> dict[str, Any]:
        return {
            "scene": self.scene,
            "seed": self.seed,
            "initial_pose": self.initial_pose.to_dict(),
            "task": self.task.to_dict(),
            "reference": self.reference.to_dict(),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "pair_id": self.pair_id,
            "group": self.group,
            "split": self.split,
            "scene": self.scene,
            "seed": self.seed,
            "initial_pose": self.initial_pose.to_dict(),
            "task": self.task.to_dict(),
            "reference": self.reference.to_dict(),
            "result_file": self.result_file,
        }


@dataclass(frozen=True)
class BenchmarkManifest:
    schema_version: str
    benchmark_id: str
    dataset_version: str
    inference_only: bool
    description: str
    required_groups: tuple[str, ...]
    minimum_scene_count: int
    episodes: tuple[EpisodeSpec, ...]

    def ordered_episodes(self) -> tuple[EpisodeSpec, ...]:
        group_order = {name: index for index, name in enumerate(self.required_groups)}
        return tuple(
            sorted(
                self.episodes,
                key=lambda episode: (
                    episode.scene,
                    episode.pair_id,
                    group_order.get(episode.group, len(group_order)),
                    episode.episode_id,
                ),
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "benchmark_id": self.benchmark_id,
            "dataset_version": self.dataset_version,
            "inference_only": self.inference_only,
            "description": self.description,
            "protocol": {
                "required_groups": list(self.required_groups),
                "minimum_scene_count": self.minimum_scene_count,
            },
            "episodes": [
                episode.to_dict() for episode in self.ordered_episodes()
            ],
        }

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    def sha256(self) -> str:
        return hashlib.sha256(
            self.canonical_json().encode("utf-8")
        ).hexdigest()

    def coverage(self) -> dict[str, Any]:
        return {
            "episode_count": len(self.episodes),
            "pair_count": len({episode.pair_id for episode in self.episodes}),
            "scenes": sorted({episode.scene for episode in self.episodes}),
            "splits": sorted({episode.split for episode in self.episodes}),
            "groups": sorted({episode.group for episode in self.episodes}),
            "task_types": sorted(
                {episode.task.task_type for episode in self.episodes}
            ),
        }


def load_manifest(path: Path | str) -> BenchmarkManifest:
    manifest_path = Path(path)
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ManifestValidationError(f"manifest does not exist: {manifest_path}") from exc
    except json.JSONDecodeError as exc:
        raise ManifestValidationError(
            f"manifest is not valid JSON at line {exc.lineno}, column {exc.colno}"
        ) from exc
    return parse_manifest(raw)


def parse_manifest(raw: Any) -> BenchmarkManifest:
    data = _mapping(raw, "manifest")
    schema_version = _non_empty_string(data.get("schema_version"), "schema_version")
    if schema_version != "1.0":
        raise ManifestValidationError(
            f"unsupported schema_version {schema_version!r}; expected '1.0'"
        )
    protocol = _mapping(data.get("protocol"), "protocol")
    required_groups = _string_list(
        protocol.get("required_groups"),
        "protocol.required_groups",
    )
    if not required_groups:
        raise ManifestValidationError("protocol.required_groups must not be empty")
    minimum_scene_count = protocol.get("minimum_scene_count")
    if (
        isinstance(minimum_scene_count, bool)
        or not isinstance(minimum_scene_count, int)
        or minimum_scene_count < 1
    ):
        raise ManifestValidationError(
            "protocol.minimum_scene_count must be a positive integer"
        )
    raw_episodes = data.get("episodes")
    if not isinstance(raw_episodes, list) or not raw_episodes:
        raise ManifestValidationError("episodes must be a non-empty array")
    episodes = tuple(
        EpisodeSpec.from_dict(episode, index)
        for index, episode in enumerate(raw_episodes)
    )

    episode_ids = [episode.episode_id for episode in episodes]
    if len(set(episode_ids)) != len(episode_ids):
        raise ManifestValidationError("episode_id values must be unique")
    result_files = [episode.result_file for episode in episodes]
    if len(set(result_files)) != len(result_files):
        raise ManifestValidationError("result_file values must be unique")
    unknown_groups = sorted(
        {episode.group for episode in episodes} - set(required_groups)
    )
    if unknown_groups:
        raise ManifestValidationError(
            f"episodes contain groups not declared by protocol: {unknown_groups}"
        )
    scenes = {episode.scene for episode in episodes}
    if len(scenes) < minimum_scene_count:
        raise ManifestValidationError(
            f"manifest requires at least {minimum_scene_count} scenes; found {len(scenes)}"
        )
    scene_splits: dict[str, set[str]] = {}
    for episode in episodes:
        scene_splits.setdefault(episode.scene, set()).add(episode.split)
    leaking_scenes = {
        scene: sorted(splits)
        for scene, splits in scene_splits.items()
        if len(splits) > 1
    }
    if leaking_scenes:
        raise ManifestValidationError(
            f"scene split leakage detected: {leaking_scenes}"
        )

    pairs: dict[str, list[EpisodeSpec]] = {}
    for episode in episodes:
        pairs.setdefault(episode.pair_id, []).append(episode)
    required_group_set = set(required_groups)
    for pair_id, members in sorted(pairs.items()):
        member_groups = {member.group for member in members}
        if member_groups != required_group_set or len(members) != len(required_groups):
            raise ManifestValidationError(
                f"pair {pair_id!r} must contain exactly one episode for each "
                f"required group {list(required_groups)}"
            )
        comparison = members[0].comparison_key()
        if any(member.comparison_key() != comparison for member in members[1:]):
            raise ManifestValidationError(
                f"pair {pair_id!r} has mismatched scene/seed/pose/task/reference"
            )

    return BenchmarkManifest(
        schema_version=schema_version,
        benchmark_id=_non_empty_string(data.get("benchmark_id"), "benchmark_id"),
        dataset_version=_non_empty_string(
            data.get("dataset_version", "unspecified"),
            "dataset_version",
        ),
        inference_only=bool(data.get("inference_only", True)),
        description=_non_empty_string(
            data.get("description", "Frozen embodied-agent benchmark"),
            "description",
        ),
        required_groups=required_groups,
        minimum_scene_count=minimum_scene_count,
        episodes=episodes,
    )
