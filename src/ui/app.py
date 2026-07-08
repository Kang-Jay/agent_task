from __future__ import annotations

from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.agent.controller import EmbodiedSearchAgent
from src.agent.model_adapter import smoke_test
from src.simulation.ai2thor_adapter import AI2ThorVisualSearchDemo, ai2thor_environment_report
from src.simulation.room_simulator import RoomSimulator
from src.types.schema import AgentRequest


class StepPayload(BaseModel):
    session_id: str
    instruction: str
    observation_image: str
    step_id: int = 0
    target_crop: str | None = None
    clicked_point: list[int] | None = None


app = FastAPI(title="Embodied Visual Search Agent", version="1.0.0")
agent = EmbodiedSearchAgent()
ROOT = Path(__file__).resolve().parents[2]
app.mount("/datasets", StaticFiles(directory=str(ROOT / "datasets")), name="datasets")
app.mount("/docs", StaticFiles(directory=str(ROOT / "docs")), name="docs")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (Path(__file__).resolve().parent / "static" / "index.html").read_text(encoding="utf-8")


@app.get("/api/agent/audit")
def audit() -> dict[str, object]:
    payload = agent.audit()
    payload["model_adapter"] = smoke_test()
    return payload


@app.post("/api/demo/run")
def run_demo(payload: dict[str, Any]) -> dict[str, Any]:
    instruction = payload.get("instruction") or "Find the red cup on the table"
    max_steps = int(payload.get("max_steps") or agent.config.max_steps)
    return RoomSimulator().run_demo(instruction=instruction, max_steps=max_steps).to_dict()


@app.get("/api/simulator/status")
def simulator_status() -> dict[str, Any]:
    status = AI2ThorVisualSearchDemo.status().to_dict()
    status["fallback"] = {
        "available": True,
        "backend": "local_ppt_style",
        "message": "Local demo remains available for full web replay and video recording.",
    }
    return status


@app.get("/api/simulator/diagnostics")
def simulator_diagnostics() -> dict[str, Any]:
    return ai2thor_environment_report()


@app.post("/api/demo/ai2thor/run")
def run_ai2thor_demo(payload: dict[str, Any]) -> dict[str, Any]:
    instruction = payload.get("instruction") or "Find the television in the room"
    max_steps = int(payload.get("max_steps") or agent.config.max_steps)
    scene = payload.get("scene") or "FloorPlan211"
    allow_fallback = bool(payload.get("allow_fallback", False))
    status = AI2ThorVisualSearchDemo.status(scene=scene)
    if not status.available:
        if allow_fallback:
            fallback = RoomSimulator().run_demo(instruction=instruction, max_steps=max_steps).to_dict()
            fallback["backend"] = "local_ppt_style_fallback"
            fallback["requested_backend"] = "ai2thor"
            fallback["scene"] = scene
            fallback["ai2thor_error"] = status.message
            fallback["ai2thor_status"] = status.to_dict()
            return fallback
        raise HTTPException(status_code=503, detail={"message": status.message, "status": status.to_dict()})
    try:
        return AI2ThorVisualSearchDemo(scene=scene).run_demo(instruction=instruction, max_steps=max_steps).to_dict()
    except Exception as exc:
        if allow_fallback:
            fallback = RoomSimulator().run_demo(instruction=instruction, max_steps=max_steps).to_dict()
            fallback["backend"] = "local_ppt_style_fallback"
            fallback["requested_backend"] = "ai2thor"
            fallback["scene"] = scene
            fallback["ai2thor_error"] = str(exc)
            fallback["ai2thor_status"] = AI2ThorVisualSearchDemo.status(scene=scene).to_dict()
            return fallback
        raise HTTPException(status_code=500, detail={"message": str(exc), "status": AI2ThorVisualSearchDemo.status(scene=scene).to_dict()}) from exc


@app.get("/api/agent/export/{session_id}")
def export_trace(session_id: str) -> dict[str, object]:
    return agent.export_trace(session_id)


@app.post("/api/agent/reset")
def reset(payload: dict[str, str]) -> dict[str, str]:
    session_id = payload.get("session_id", "demo")
    return agent.reset(session_id)


@app.post("/api/agent/step")
def step(payload: StepPayload) -> dict[str, Any]:
    try:
        data = payload.model_dump() if hasattr(payload, "model_dump") else payload.dict()
        request = AgentRequest(**data)
        return agent.step(request).to_dict()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def main() -> None:
    uvicorn.run("src.ui.app:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
