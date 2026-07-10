"""Dataset expansion utility for Phase 5.

According to Plan_1_agent_demo_repair.md Phase 5 requirements.

This module helps generate additional episodes including:
- Target absent cases
- Target occluded cases
- Wrong object clicked cases
- Multi-step navigation cases
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = ROOT / "datasets" / "embodied_search_v1"
ANNOTATION_FILE = DATASET_ROOT / "annotations" / "episodes.jsonl"


@dataclass
class EpisodeSpec:
    """Specification for generating an episode."""
    episode_id: str
    instruction: str
    target: dict[str, Any]
    image_path: str
    expected_action: str
    category: str  # positive, negative_absent, negative_wrong, multi_step
    difficulty: str  # easy, medium, hard
    steps: list[dict[str, Any]] = field(default_factory=list)


def load_episodes() -> list[dict[str, Any]]:
    """Load existing episodes from annotations file."""
    if not ANNOTATION_FILE.exists():
        return []

    episodes = []
    with open(ANNOTATION_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                episodes.append(json.loads(line))
    return episodes


def save_episodes(episodes: list[dict[str, Any]]) -> None:
    """Save episodes to annotations file."""
    ANNOTATION_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(ANNOTATION_FILE, 'w', encoding='utf-8') as f:
        for episode in episodes:
            f.write(json.dumps(episode, ensure_ascii=False) + '\n')


def add_episode(spec: EpisodeSpec) -> dict[str, Any]:
    """Add a new episode to the dataset."""
    episode = {
        "episode_id": spec.episode_id,
        "instruction": spec.instruction,
        "target": spec.target,
        "image": spec.image_path,
        "expected_action": spec.expected_action,
        "category": spec.category,
        "difficulty": spec.difficulty,
        "steps": spec.steps if spec.steps else [
            {
                "image": spec.image_path,
                "thought": "Generated episode",
                "action": spec.expected_action,
                "done": spec.expected_action == "STOP"
            }
        ]
    }
    return episode


def split_dataset(episodes: list[dict[str, Any]], train_ratio: float = 0.7, val_ratio: float = 0.15) -> dict[str, list[dict[str, Any]]]:
    """Split dataset into train/val/test splits.

    Args:
        episodes: List of all episodes
        train_ratio: Ratio for training set (default 0.7)
        val_ratio: Ratio for validation set (default 0.15)

    Returns:
        Dict with 'train', 'val', 'test' splits
    """
    total = len(episodes)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    return {
        "train": episodes[:train_end],
        "val": episodes[train_end:val_end],
        "test": episodes[val_end:]
    }


def save_splits(splits: dict[str, list[dict[str, Any]]]) -> None:
    """Save train/val/test splits to separate files."""
    splits_dir = DATASET_ROOT / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    for split_name, episodes in splits.items():
        split_file = splits_dir / f"{split_name}.jsonl"
        with open(split_file, 'w', encoding='utf-8') as f:
            for episode in episodes:
                f.write(json.dumps(episode, ensure_ascii=False) + '\n')
        print(f"Saved {len(episodes)} episodes to {split_file}")


def generate_negative_episodes() -> list[EpisodeSpec]:
    """Generate negative example episodes.

    Returns:
        List of negative episode specifications
    """
    negative_specs = [
        EpisodeSpec(
            episode_id="ep_target_absent_cup",
            instruction="Find the red cup on the table",
            target={"name": "red cup", "type": "cup", "bbox": None, "present": False},
            image_path="images/ep_target_absent_cup_000.png",
            expected_action="ASK_CLARIFY",
            category="negative_absent",
            difficulty="medium",
            steps=[]
        ),
        EpisodeSpec(
            episode_id="ep_wrong_object_book",
            instruction="Find the red cup",
            target={"name": "red cup", "type": "cup", "bbox": None, "present": False},
            image_path="images/ep_blue_book_visible_000.png",  # Reuse existing image
            expected_action="TURN_RIGHT",
            category="negative_wrong",
            difficulty="medium",
            steps=[]
        ),
        EpisodeSpec(
            episode_id="ep_occluded_plant",
            instruction="Find the green plant",
            target={"name": "green plant", "type": "plant", "bbox": [72, 82, 100, 120], "occluded": 0.6},
            image_path="images/ep_green_plant_visible_000.png",  # Reuse, but mark as partially occluded
            expected_action="MOVE_FORWARD",
            category="negative_occluded",
            difficulty="hard",
            steps=[]
        )
    ]
    return negative_specs


def dataset_statistics(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute dataset statistics."""
    categories = {}
    difficulties = {}
    actions = {}

    for ep in episodes:
        cat = ep.get("category", "unknown")
        diff = ep.get("difficulty", "unknown")
        act = ep.get("expected_action", "unknown")

        categories[cat] = categories.get(cat, 0) + 1
        difficulties[diff] = difficulties.get(diff, 0) + 1
        actions[act] = actions.get(act, 0) + 1

    return {
        "total_episodes": len(episodes),
        "categories": categories,
        "difficulties": difficulties,
        "actions": actions
    }


def expand_dataset() -> None:
    """Main function to expand the dataset."""
    print("=" * 60)
    print("Dataset Expansion Utility")
    print("=" * 60)
    print()

    # Load existing episodes
    episodes = load_episodes()
    print(f"Loaded {len(episodes)} existing episodes")

    # Mark existing episodes as positive/easy
    for ep in episodes:
        if "category" not in ep:
            ep["category"] = "positive"
        if "difficulty" not in ep:
            ep["difficulty"] = "easy"

    # Generate negative episodes
    negative_specs = generate_negative_episodes()
    print(f"Generated {len(negative_specs)} negative episode specs")

    for spec in negative_specs:
        episode = add_episode(spec)
        # Check if already exists
        if not any(e["episode_id"] == episode["episode_id"] for e in episodes):
            episodes.append(episode)
            print(f"  + Added: {episode['episode_id']} ({episode['category']}, {episode['difficulty']})")

    # Save updated episodes
    save_episodes(episodes)
    print(f"\nSaved {len(episodes)} total episodes to {ANNOTATION_FILE}")

    # Create splits
    splits = split_dataset(episodes, train_ratio=0.6, val_ratio=0.2)
    save_splits(splits)

    # Print statistics
    stats = dataset_statistics(episodes)
    print("\n" + "=" * 60)
    print("Dataset Statistics")
    print("=" * 60)
    print(f"Total Episodes: {stats['total_episodes']}")
    print(f"\nCategories:")
    for cat, count in stats['categories'].items():
        print(f"  {cat}: {count}")
    print(f"\nDifficulties:")
    for diff, count in stats['difficulties'].items():
        print(f"  {diff}: {count}")
    print(f"\nExpected Actions:")
    for act, count in stats['actions'].items():
        print(f"  {act}: {count}")
    print("=" * 60)


if __name__ == "__main__":
    expand_dataset()
