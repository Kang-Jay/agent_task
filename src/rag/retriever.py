from __future__ import annotations

from src.memory.session_memory import SessionState
from src.task.config import AgentConfig


OBJECT_LOCATION_PRIORS = {
    "cup": ["cups are often on tables, counters, or near sinks", "inspect tabletop regions before floor regions"],
    "book": ["books often appear on desks, shelves, or sofas", "scan horizontal surfaces and shelving"],
    "plant": ["plants often stand near windows, corners, or tables", "look for green regions at room edges"],
    "remote": ["remote controls often lie on sofas, coffee tables, or TV stands", "inspect living-room seating areas"],
    "bottle": ["bottles often appear on counters, desks, or dining tables", "prioritize upright colored objects"],
}


class HintRetriever:
    def __init__(self, config: AgentConfig):
        self.config = config

    def retrieve(self, instruction: str, state: SessionState) -> list[str]:
        lower = instruction.lower()
        hints: list[str] = []
        for key, priors in OBJECT_LOCATION_PRIORS.items():
            if key in lower or self._has_chinese_keyword(key, instruction):
                hints.extend(priors)

        if state.negative_memory:
            hints.append("avoid repeating low-confidence regions: " + " | ".join(state.negative_memory[-2:]))

        if state.explored_regions:
            least_seen = sorted(state.explored_regions.items(), key=lambda item: item[1])[0][0]
            hints.append(f"least repeated explored region so far: {least_seen}")

        top_k = int(self.config.raw["memory"]["retrieval_top_k"])
        return hints[:top_k]

    def _has_chinese_keyword(self, key: str, text: str) -> bool:
        mapping = {
            "cup": ["杯", "杯子"],
            "book": ["书", "书本"],
            "plant": ["植物", "盆栽"],
            "remote": ["遥控器"],
            "bottle": ["瓶", "瓶子"],
        }
        return any(item in text for item in mapping.get(key, []))

