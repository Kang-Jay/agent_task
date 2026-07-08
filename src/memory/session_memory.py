from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.task.config import AgentConfig


@dataclass
class SessionState:
    session_id: str
    instruction: str
    steps: list[dict[str, Any]] = field(default_factory=list)
    negative_memory: list[str] = field(default_factory=list)
    explored_regions: dict[str, int] = field(default_factory=dict)

    def recent_steps(self, window: int) -> list[dict[str, Any]]:
        return self.steps[-window:]


class SessionMemory:
    def __init__(self, config: AgentConfig):
        self.config = config
        self.sessions: dict[str, SessionState] = {}
        self.long_term_events: list[dict[str, Any]] = []
        self.trace_dir = config.trajectory_dir
        self.trace_dir.mkdir(parents=True, exist_ok=True)

    def get_or_create(self, session_id: str, instruction: str) -> SessionState:
        if session_id not in self.sessions:
            self.sessions[session_id] = SessionState(session_id=session_id, instruction=instruction)
        return self.sessions[session_id]

    def reset(self, session_id: str) -> None:
        self.sessions.pop(session_id, None)
        trace_path = self.trace_dir / f"{session_id}.json"
        if trace_path.exists():
            trace_path.unlink()

    def record_step(self, state: SessionState, step: dict[str, Any]) -> None:
        state.steps.append(step)
        best = step.get("best_candidate")
        if best:
            region = best.get("region", "unknown")
            state.explored_regions[region] = state.explored_regions.get(region, 0) + 1
        if not step.get("done") and best and step.get("confidence", 0.0) < self.config.target_visible_threshold:
            state.negative_memory.append(f"Searched {best.get('region', 'unknown')} but confidence stayed low.")
            capacity = int(self.config.raw["memory"]["negative_memory_capacity"])
            state.negative_memory = state.negative_memory[-capacity:]
        self.long_term_events.append(
            {
                "session_id": state.session_id,
                "instruction": state.instruction,
                "action": step.get("action", {}).get("type"),
                "confidence": step.get("confidence"),
                "region": best.get("region") if best else None,
            }
        )
        capacity = int(self.config.raw["memory"]["long_term_capacity"])
        self.long_term_events = self.long_term_events[-capacity:]
        if self.config.raw["memory"]["persist_traces"]:
            self._persist_trace(state)

    def summarize(self, state: SessionState) -> str:
        if not state.steps:
            return "No prior steps in this session."
        last = state.steps[-1]
        explored = ", ".join(f"{region}:{count}" for region, count in sorted(state.explored_regions.items()))
        return (
            f"{len(state.steps)} steps recorded. Last action={last.get('action', {}).get('type')}; "
            f"explored regions={explored or 'none'}; negative memories={len(state.negative_memory)}."
        )

    def search_map(self, state: SessionState) -> dict[str, Any]:
        regions = [
            "upper left",
            "upper center",
            "upper right",
            "middle left",
            "middle center",
            "middle right",
            "lower left",
            "lower center",
            "lower right",
        ]
        visited = {region: state.explored_regions.get(region, 0) for region in regions}
        confidence_by_region: dict[str, float] = {}
        for step in state.steps:
            best = step.get("best_candidate") or {}
            region = best.get("region")
            if region:
                confidence_by_region[region] = max(
                    confidence_by_region.get(region, 0.0),
                    float(step.get("confidence", 0.0)),
                )
        unexplored = [region for region, count in visited.items() if count == 0]
        return {
            "visited_counts": visited,
            "unexplored_regions": unexplored,
            "confidence_by_region": confidence_by_region,
            "negative_memory": list(state.negative_memory),
        }

    def confidence_trace(self, state: SessionState) -> list[float]:
        return [float(step.get("confidence", 0.0)) for step in state.steps]

    def export_trace(self, session_id: str) -> dict[str, Any]:
        state = self.sessions.get(session_id)
        if state is None:
            trace_path = self.trace_dir / f"{session_id}.json"
            if trace_path.exists():
                return json.loads(trace_path.read_text(encoding="utf-8"))
            return {"session_id": session_id, "steps": []}
        return {
            "session_id": state.session_id,
            "instruction": state.instruction,
            "steps": state.steps,
            "negative_memory": state.negative_memory,
            "explored_regions": state.explored_regions,
        }

    def _persist_trace(self, state: SessionState) -> None:
        path = self.trace_dir / f"{state.session_id}.json"
        payload = {
            "session_id": state.session_id,
            "instruction": state.instruction,
            "steps": state.steps,
            "negative_memory": state.negative_memory,
            "explored_regions": state.explored_regions,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
