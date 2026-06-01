"""FastAPI app exposing the pet runtime over WebSocket + a tiny HTTP API.

Endpoints
---------
GET  /healthz            — liveness
GET  /pet/state          — current PetState (JSON)
POST /pet/action         — apply a single action (testing / non-WS clients)
POST /pet/perception     — push a perception result; current Phase 2 wires this to
                           a placeholder behavior (move toward the highest-confidence
                           detection). The full grounding lives in Phase 6/7.
WS   /ws/pet             — stream PetAction events; client may also send actions back
"""
from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .pet_runtime import PetRuntime

log = logging.getLogger("pet_agent.ws")

runtime = PetRuntime()


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    log.info("pet runtime starting")
    yield
    log.info("pet runtime stopping")


app = FastAPI(title="3D Pet Agent", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ActionRequest(BaseModel):
    action: str
    target_position_3d: tuple[float, float, float] | None = None
    animation: str | None = None
    emotion: str | None = None
    speed: float | None = None
    speech: str | None = None


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/pet/state")
async def get_state() -> dict[str, Any]:
    return runtime.state.model_dump()


@app.post("/pet/action")
async def apply_action(req: ActionRequest) -> dict[str, Any]:
    runtime.apply(req.model_dump(exclude_none=True))
    return runtime.state.model_dump()


@app.post("/pet/perception")
async def push_perception(payload: dict[str, Any]) -> dict[str, Any]:
    """Wire perception output into a placeholder pet behavior.

    Phase 2 only: pick the highest-confidence detection and move the pet near it.
    The full grounding pipeline lives in later phases.
    """
    objects = payload.get("objects_2d", [])
    if not objects:
        runtime.play_animation("curious")
        return {"applied": False, "reason": "no objects"}
    best = max(objects, key=lambda o: float(o.get("detector_confidence", 0.0)))
    label = best.get("label", "object")
    cx_n = float(best.get("center_normalized", [0.5, 0.5])[0])
    # Map normalized x∈[0,1] → world x∈[-1, 1]; small z offset in front of camera.
    runtime.ask(f"I see a {label}")
    runtime.set_emotion("curious")
    runtime.move_to(x=(cx_n - 0.5) * 2.0, y=0.0, z=1.2)
    return {"applied": True, "target_label": label}


@app.websocket("/ws/pet")
async def ws_pet(ws: WebSocket) -> None:
    await ws.accept()
    q = runtime.subscribe()
    # Send current state immediately.
    await ws.send_text(runtime.snapshot().model_dump_json())

    async def pump_outgoing() -> None:
        try:
            while True:
                action = await q.get()
                await ws.send_text(action.model_dump_json())
        except (WebSocketDisconnect, RuntimeError):
            return

    async def pump_incoming() -> None:
        try:
            while True:
                raw = await ws.receive_text()
                try:
                    payload = json.loads(raw)
                    runtime.apply(payload)
                except Exception as e:
                    log.warning("bad ws action: %s", e)
        except WebSocketDisconnect:
            return

    try:
        await asyncio.gather(pump_outgoing(), pump_incoming())
    finally:
        runtime.unsubscribe(q)
