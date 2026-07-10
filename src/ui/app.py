from __future__ import annotations

import asyncio
import threading
import uuid
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.agent.controller import EmbodiedSearchAgent
from src.agent.model_adapter import smoke_test
from src.simulation.ai2thor_actions import AI2ThorActionCatalog
from src.simulation.ai2thor_adapter import AI2ThorVisualSearchDemo, ai2thor_environment_report
from src.simulation.ai2thor_session import AI2ThorSessionManager
from src.simulation.room_simulator import RoomSimulator
from src.simulation.stream_protocol import encode_ndjson
from src.types.schema import AgentRequest


class StepPayload(BaseModel):
    session_id: str
    instruction: str
    observation_image: str
    step_id: int = 0
    target_crop: str | None = None
    clicked_point: list[int] | None = None
    agent_mode: str = "default"
    environment_context: dict[str, Any] | None = None


app = FastAPI(title="Embodied Visual Search Agent", version="1.0.0")
agent = EmbodiedSearchAgent()
action_catalog = AI2ThorActionCatalog()
simulator_sessions = AI2ThorSessionManager(catalog=action_catalog)
active_stream_sessions: set[str] = set()
active_live_sessions: set[str] = set()
active_stream_lock = threading.Lock()
simulator_slot = threading.BoundedSemaphore(1)
ROOT = Path(__file__).resolve().parents[2]
app.mount("/datasets", StaticFiles(directory=str(ROOT / "datasets")), name="datasets")
app.mount("/docs", StaticFiles(directory=str(ROOT / "docs")), name="docs")


def _fresh_agent() -> EmbodiedSearchAgent:
    return EmbodiedSearchAgent(
        config=agent.config,
        model_adapter=agent.model_adapter,
    )


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (Path(__file__).resolve().parent / "static" / "index.html").read_text(encoding="utf-8")


@app.get("/api/agent/audit")
def audit() -> dict[str, object]:
    payload = agent.audit()
    payload["model_adapter"] = agent.model_adapter.audit()
    return payload


@app.post("/api/agent/model/smoke-test")
def model_smoke_test() -> dict[str, Any]:
    return smoke_test()


@app.post("/api/demo/run")
def run_demo(payload: dict[str, Any]) -> dict[str, Any]:
    instruction = payload.get("instruction") or "Find the red cup on the table"
    max_steps = int(payload.get("max_steps") or agent.config.max_steps)
    clicked_point = payload.get("clicked_point")
    session_id = str(payload.get("session_id") or "recorded-demo")
    return RoomSimulator(agent=_fresh_agent()).run_demo(
        instruction=instruction,
        max_steps=max_steps,
        clicked_point=clicked_point,
        session_id=session_id,
    ).to_dict()


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


@app.get("/api/simulator/actions")
def simulator_actions(
    mode: str = "default",
    actor: str = "agent",
    include_internal: bool = False,
) -> dict[str, Any]:
    try:
        actions = action_catalog.list_actions(
            mode=mode,
            actor=actor,
            include_internal=include_internal,
        )
        return {
            "mode": mode,
            "actor": actor,
            "count": len(actions),
            "catalog": action_catalog.summary(),
            "actions": actions,
        }
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/simulator/actions/validate")
def validate_simulator_action(payload: dict[str, Any]) -> dict[str, Any]:
    validation = action_catalog.validate(
        mode=str(payload.get("mode") or "default"),
        action=str(payload.get("action") or ""),
        args=payload.get("args") or {},
        actor=str(payload.get("actor") or "agent"),
    )
    return validation.to_dict()


@app.get("/api/simulator/sessions")
def list_simulator_sessions() -> dict[str, Any]:
    return {"sessions": simulator_sessions.list_sessions()}


@app.post("/api/simulator/session/start")
def start_simulator_session(payload: dict[str, Any]) -> dict[str, Any]:
    session_id = str(payload.get("session_id") or "interactive-ai2thor")
    with active_stream_lock:
        if session_id in active_live_sessions:
            raise HTTPException(
                status_code=409,
                detail=f"AI2-THOR live session is already running: {session_id}",
            )
        acquired_slot = simulator_slot.acquire(blocking=False)
        if not acquired_slot:
            raise HTTPException(
                status_code=409,
                detail="AI2-THOR runtime is busy with another Unity controller",
            )
    try:
        result = simulator_sessions.start(
            session_id=session_id,
            scene=str(payload.get("scene") or "FloorPlan211"),
            mode=str(payload.get("mode") or "default"),
            width=int(payload.get("width") or 960),
            height=int(payload.get("height") or 540),
            quality=str(payload.get("quality") or "Low"),
            grid_size=float(payload.get("grid_size") or 0.25),
            rotate_step_degrees=float(payload.get("rotate_step_degrees") or 90.0),
            render_instance_segmentation=bool(
                payload.get("render_instance_segmentation", True)
            ),
        )
        with active_stream_lock:
            active_live_sessions.add(session_id)
        return result
    except Exception as exc:
        if acquired_slot:
            simulator_slot.release()
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/simulator/session/action")
def execute_simulator_action(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        return simulator_sessions.execute(
            session_id=str(payload.get("session_id") or "interactive-ai2thor"),
            action=str(payload.get("action") or ""),
            args=payload.get("args") or {},
            actor=str(payload.get("actor") or "manual"),
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/simulator/session/{session_id}")
def simulator_session_snapshot(session_id: str) -> dict[str, Any]:
    try:
        return simulator_sessions.snapshot(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.delete("/api/simulator/session/{session_id}")
def close_simulator_session(session_id: str) -> dict[str, Any]:
    closed = simulator_sessions.close(session_id)
    if closed:
        with active_stream_lock:
            if session_id in active_live_sessions:
                active_live_sessions.remove(session_id)
                simulator_slot.release()
    return {"session_id": session_id, "closed": closed}


@app.post("/api/demo/ai2thor/run")
def run_ai2thor_demo(payload: dict[str, Any]) -> dict[str, Any]:
    instruction = payload.get("instruction") or "Find the television in the room"
    max_steps = int(payload.get("max_steps") or agent.config.max_steps)
    scene = payload.get("scene") or "FloorPlan211"
    agent_mode = str(payload.get("agent_mode") or "default")
    allow_fallback = bool(payload.get("allow_fallback", False))
    clicked_point = payload.get("clicked_point")
    session_id = str(payload.get("session_id") or "ai2thor-demo")
    status = AI2ThorVisualSearchDemo.status(scene=scene)
    if not status.available:
        if allow_fallback:
            fallback = RoomSimulator(agent=_fresh_agent()).run_demo(
                instruction=instruction,
                max_steps=max_steps,
                clicked_point=clicked_point,
                session_id=session_id,
            ).to_dict()
            fallback["backend"] = "local_ppt_style_fallback"
            fallback["requested_backend"] = "ai2thor"
            fallback["scene"] = scene
            fallback["ai2thor_error"] = status.message
            fallback["ai2thor_status"] = status.to_dict()
            return fallback
        raise HTTPException(status_code=503, detail={"message": status.message, "status": status.to_dict()})
    if not simulator_slot.acquire(blocking=False):
        raise HTTPException(
            status_code=409,
            detail="AI2-THOR runtime is busy with another Unity controller",
        )
    try:
        return AI2ThorVisualSearchDemo(
            scene=scene,
            agent=_fresh_agent(),
            agent_mode=agent_mode,
        ).run_demo(
            instruction=instruction,
            max_steps=max_steps,
            clicked_point=clicked_point,
            session_id=session_id,
        ).to_dict()
    except Exception as exc:
        if allow_fallback:
            fallback = RoomSimulator(agent=_fresh_agent()).run_demo(
                instruction=instruction,
                max_steps=max_steps,
                clicked_point=clicked_point,
                session_id=session_id,
            ).to_dict()
            fallback["backend"] = "local_ppt_style_fallback"
            fallback["requested_backend"] = "ai2thor"
            fallback["scene"] = scene
            fallback["ai2thor_error"] = str(exc)
            fallback["ai2thor_status"] = AI2ThorVisualSearchDemo.status(scene=scene).to_dict()
            return fallback
        raise HTTPException(status_code=500, detail={"message": str(exc), "status": AI2ThorVisualSearchDemo.status(scene=scene).to_dict()}) from exc
    finally:
        simulator_slot.release()


@app.post("/api/demo/ai2thor/stream")
async def stream_ai2thor_demo(
    payload: dict[str, Any],
    request: Request,
) -> StreamingResponse:
    instruction = payload.get("instruction") or "Find the television in the room"
    max_steps = int(payload.get("max_steps") or agent.config.max_steps)
    scene = str(payload.get("scene") or "FloorPlan211")
    agent_mode = str(payload.get("agent_mode") or "default")
    clicked_point = payload.get("clicked_point")
    session_id = str(payload.get("session_id") or "ai2thor-demo")
    status = AI2ThorVisualSearchDemo.status(scene=scene)
    if not status.available:
        raise HTTPException(
            status_code=503,
            detail={"message": status.message, "status": status.to_dict()},
        )

    with active_stream_lock:
        if session_id in active_stream_sessions:
            raise HTTPException(
                status_code=409,
                detail=f"AI2-THOR session is already running: {session_id}",
            )
        if not simulator_slot.acquire(blocking=False):
            raise HTTPException(
                status_code=409,
                detail="AI2-THOR runtime is busy with another Unity controller",
            )
        active_stream_sessions.add(session_id)

    episode_id = uuid.uuid4().hex
    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
    cancel_event = threading.Event()
    loop = asyncio.get_running_loop()

    def emit(message: dict[str, Any]) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, message)

    def worker() -> None:
        try:
            AI2ThorVisualSearchDemo(
                scene=scene,
                agent=_fresh_agent(),
                agent_mode=agent_mode,
            ).run_demo(
                instruction=instruction,
                max_steps=max_steps,
                clicked_point=clicked_point,
                session_id=session_id,
                episode_id=episode_id,
                emit=emit,
                cancel_event=cancel_event,
            )
        finally:
            with active_stream_lock:
                active_stream_sessions.discard(session_id)
            simulator_slot.release()
            loop.call_soon_threadsafe(queue.put_nowait, None)

    async def event_stream():
        worker_task = asyncio.create_task(asyncio.to_thread(worker))
        try:
            while True:
                try:
                    message = await asyncio.wait_for(queue.get(), timeout=0.25)
                except asyncio.TimeoutError:
                    if await request.is_disconnected():
                        cancel_event.set()
                    if worker_task.done() and queue.empty():
                        break
                    continue
                if message is None:
                    break
                yield encode_ndjson(message)
        finally:
            cancel_event.set()
            try:
                await worker_task
            except Exception:
                pass

    return StreamingResponse(
        event_stream(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "X-Episode-Id": episode_id,
        },
    )


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
