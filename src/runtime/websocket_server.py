"""FastAPI app exposing the pet runtime over WebSocket + a tiny HTTP API.

Endpoints
---------
GET  /healthz            — liveness
GET  /pet/state          — current PetState (JSON)
POST /pet/action         — apply a single action (testing / non-WS clients)
POST /pet/perception     — push a 2D perception result; Phase 2 placeholder
                           moves the pet toward the highest-confidence detection.
                           Full grounding lives in Phase 6.
POST /perception/lifted  — push a Phase 3 lifted result (center_3d_world per object);
                           broadcasts a ``world_update`` so the renderer drops
                           phosphor markers at each centroid.
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

from .pet_runtime import PetAction, PetRuntime

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


Waypoint = tuple[float, float, float]


class ActionRequest(BaseModel):
    action: str
    target_position_3d: Waypoint | None = None
    path: list[Waypoint] | None = None
    look_at_object_id: str | None = None
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
    target_x = (cx_n - 0.5) * 2.0
    target_z = 1.2
    runtime.ask(f"I see a {label}")
    runtime.set_emotion("curious")
    # Emit a tiny 3-waypoint approach path so the frontend exercises the
    # path-following code path. The real A* planner arrives in Phase 7.
    start = runtime.state.position
    midpoint = (
        (start.x + target_x) * 0.5,
        0.0,
        (start.z + target_z) * 0.5 + 0.1,
    )
    path = [
        (start.x, 0.0, start.z),
        midpoint,
        (target_x, 0.0, target_z),
    ]
    runtime.move_follow_path(path, speed=0.6)
    return {"applied": True, "target_label": label, "path_waypoints": len(path)}


@app.post("/perception/lifted")
async def push_lifted(payload: dict[str, Any]) -> dict[str, Any]:
    """Accept a Phase 3 lifted JSON and broadcast ``world_update``.

    Expects ``{"objects": [ObjectState3D, ...]}`` — exactly the shape the
    snapshot CLI writes to ``runs/lifted_*.json``.
    """
    objects = payload.get("objects", [])
    markers: list[dict[str, Any]] = []
    for o in objects:
        c3 = o.get("center_3d_world")
        if not c3:
            continue
        markers.append(
            {
                "object_id": o.get("object_id", "obj_unknown"),
                "class_label": o.get("class_label", "object"),
                "center_3d_world": [float(c3[0]), float(c3[1]), float(c3[2])],
                "extent_3d": o.get("extent_3d"),
                "median_depth": o.get("median_depth"),
                "depth_uncertainty": o.get("depth_uncertainty"),
                "confidence": (o.get("confidence") or {}).get("overall"),
            }
        )

    action = PetAction(action="world_update", world_objects=markers)
    runtime._broadcast(action)  # noqa: SLF001 — runtime exposes this for sibling modules
    return {"applied": True, "markers": len(markers)}


@app.websocket("/ws/pet")
async def ws_pet(ws: WebSocket) -> None:
    await ws.accept()
    q = runtime.subscribe()
    # Send current state immediately, then replay the latest world_update
    # so a freshly-opened browser sees the same scene markers as one that
    # has been connected the whole time.
    await ws.send_text(runtime.snapshot().model_dump_json())
    last_world = runtime.last_world_update()
    if last_world is not None:
        await ws.send_text(last_world.model_dump_json())

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
