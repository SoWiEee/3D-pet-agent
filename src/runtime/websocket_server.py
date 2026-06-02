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
import math
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ..config import load_control, load_navigation
from ..control import (
    PathFollower,
    PIDController,
    PurePursuitController,
    UnicycleState,
)
from ..exploration import (
    CoverageGrid,
    CoverageGridConfig,
    ExplorationGoal,
    ExplorationPlanner,
)
from ..language import parse_command
from ..planning import GridConfig, GroundingResolver, Planner, PlannerConfig
from ..spatial import SceneGraphBuilder, SemanticMap
from ..spatial.object_lifter import ObjectState3D
from ..tracking import Tracker
from .pet_runtime import PetAction, PetRuntime

log = logging.getLogger("pet_agent.ws")

runtime = PetRuntime()
# Server-held tracker + SemanticMap + SceneGraphBuilder + GroundingResolver +
# Planner. Phase 4–7 demo: clients POST lifted JSON, the server tracks ids,
# fuses into the map, rebuilds the scene graph, broadcasts both to the
# renderer; user utterances POSTed to /command parse → ground → plan →
# emit move_follow_path (or move_to if planner has no goal cell).
tracker = Tracker()
semantic_map = SemanticMap(map_id="live")
scene_graph_builder = SceneGraphBuilder()
grounding_resolver = GroundingResolver()

_nav_cfg = load_navigation()
planner = Planner(
    PlannerConfig(
        grid=GridConfig(
            resolution=_nav_cfg.grid.resolution,
            origin_x=_nav_cfg.grid.origin_x,
            origin_z=_nav_cfg.grid.origin_z,
            width=_nav_cfg.grid.width,
            height=_nav_cfg.grid.height,
            obstacle_padding=_nav_cfg.grid.obstacle_padding,
        ),
        connectivity=_nav_cfg.planner.connectivity,
        nearest_free_radius_m=_nav_cfg.planner.nearest_free_radius,
        smoothing=_nav_cfg.planner.smoothing,
        smoothing_subdivisions=_nav_cfg.planner.smoothing_subdivisions,
        avoid_default_min_distance=_nav_cfg.constraints.avoid_default_min_distance,
    )
)
_default_speed = _nav_cfg.planner.default_speed

# ── Phase 8: pure-pursuit controller (offline simulator) ────────────────────
_ctrl_cfg = load_control()
_pure_pursuit = PurePursuitController(
    lookahead_distance=_ctrl_cfg.pure_pursuit.lookahead_distance,
    base_speed=_ctrl_cfg.pure_pursuit.base_speed,
    kp_heading=_ctrl_cfg.pure_pursuit.kp_heading,
    v_max=_ctrl_cfg.kinematic.v_max,
    v_min=_ctrl_cfg.kinematic.v_min,
    omega_max=_ctrl_cfg.kinematic.omega_max,
)
_speed_pid = PIDController(
    kp=_ctrl_cfg.speed_pid.kp,
    ki=_ctrl_cfg.speed_pid.ki,
    kd=_ctrl_cfg.speed_pid.kd,
    integral_clamp=_ctrl_cfg.speed_pid.integral_clamp,
)
path_follower = PathFollower(
    controller=_pure_pursuit,
    pid=_speed_pid,
    v_max=_ctrl_cfg.kinematic.v_max,
    omega_max=_ctrl_cfg.kinematic.omega_max,
    dt=_ctrl_cfg.kinematic.dt,
    max_steps=_ctrl_cfg.kinematic.max_steps,
    goal_tolerance=_ctrl_cfg.pure_pursuit.goal_tolerance,
)

# Sticky last trace for debug overlay.
_last_trace_summary: dict[str, Any] | None = None

# ── Phase 9: exploration ────────────────────────────────────────────────────
# Coverage grid aligned with the navigation grid so unobserved cells are
# discoverable in the same coordinate frame.
coverage_grid = CoverageGrid(
    CoverageGridConfig(
        resolution=_nav_cfg.grid.resolution,
        origin_x=_nav_cfg.grid.origin_x,
        origin_z=_nav_cfg.grid.origin_z,
        width=_nav_cfg.grid.width,
        height=_nav_cfg.grid.height,
    )
)
exploration_planner = ExplorationPlanner()
# Object ids seen prior to the last exploration tick; used to report what was
# newly discovered during the step.
_last_exploration_ids: set[str] = set()

# ── A1: live perception loop ───────────────────────────────────────────────
# Held but **not started** at boot — heavy models only load when the user
# POSTs /perception/start. Loop reuses the shared tracker + semantic_map +
# scene_graph_builder so its broadcasts are indistinguishable from
# /perception/lifted POSTs.
# Bootstrap config — pulled here so we can pass cfg + prompts default.
from ..config import AppConfig as _AppConfig  # noqa: E402
from ..config import load_prompts as _load_prompts  # noqa: E402
from .perception_loop import PerceptionLoop  # noqa: E402 — must follow tracker/map setup

try:
    _app_cfg = _AppConfig.load()
except Exception as e:  # pragma: no cover — only on broken configs
    log.warning("AppConfig.load failed (%s); perception loop will be unavailable", e)
    _app_cfg = None  # type: ignore[assignment]

perception_loop: PerceptionLoop | None = (
    PerceptionLoop(
        cfg=_app_cfg,
        tracker=tracker,
        semantic_map=semantic_map,
        scene_graph_builder=scene_graph_builder,
        broadcast=lambda action: runtime._broadcast(action),  # noqa: SLF001
        markers_fn=lambda m: _map_to_markers(m),
    )
    if _app_cfg is not None
    else None
)


# ── B2: SemanticMap autoload / autosave ───────────────────────────────────
# Restart resilience — `uvicorn` restart preserves the map by default. The
# env var lets tests / staging override the path; absence of the file is
# a clean "fresh boot" (no error, no log noise beyond a debug line).
_SEMANTIC_MAP_PATH = Path(os.environ.get("PET_AGENT_SEMANTIC_MAP_PATH", "runs/last_map.json"))


def _try_autoload_semantic_map(target: SemanticMap, path: Path) -> int:
    """Populate ``target`` from a JSON snapshot on disk.

    The shared SemanticMap reference is **not** swapped — that would break
    every component that captured a pointer to it at module import time
    (perception loop, etc.). Instead we reset + repopulate in place.

    Returns the number of objects loaded (0 means clean boot).
    """
    if not path.exists():
        log.debug("no SemanticMap snapshot at %s; starting clean", path)
        return 0
    try:
        loaded = SemanticMap.load(path)
    except Exception as e:  # noqa: BLE001 — corrupt snapshot must not block boot
        log.warning("SemanticMap autoload failed (%s); starting clean", e)
        return 0
    target.reset()
    for k, v in loaded._objects.items():  # noqa: SLF001 — sibling module access
        target._objects[k] = v
    target.last_frame_id = loaded.last_frame_id
    target.last_updated = loaded.last_updated
    log.info("autoloaded SemanticMap: %d objects from %s", len(loaded._objects), path)
    return len(loaded._objects)


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    log.info("pet runtime starting")
    n_loaded = _try_autoload_semantic_map(semantic_map, _SEMANTIC_MAP_PATH)
    if n_loaded > 0:
        # Broadcast the autoloaded scene so freshly-connected clients see it
        # without having to wait for the next perception POST.
        markers = _map_to_markers(semantic_map)
        graph = scene_graph_builder.build(semantic_map)
        runtime._broadcast(  # noqa: SLF001
            PetAction(
                action="world_update",
                world_objects=markers,
                scene_graph=graph.to_dict(),
            )
        )
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


class PerceptionStartRequest(BaseModel):
    prompts: list[str] | None = None
    camera_index: int = 0
    fov_deg: float = 60.0
    hz: float | None = None


@app.post("/perception/start")
async def post_perception_start(req: PerceptionStartRequest) -> dict[str, Any]:
    """A1 live perception loop — opens webcam, loads models, starts ticking.

    Heavy models load **here** (not at server boot) so the demo can run
    without GPU until the user opts in. Subsequent calls while running
    return 409.
    """
    if perception_loop is None:
        return {"started": False, "reason": "perception loop unavailable (config load failed)"}
    if perception_loop.running:
        return {
            "started": False,
            "reason": "already running",
            "status": perception_loop.status.__dict__,
        }
    prompts = req.prompts or _load_prompts()
    try:
        await perception_loop.start(
            prompts=prompts,
            camera_index=req.camera_index,
            fov_deg=req.fov_deg,
            hz=req.hz,
        )
    except Exception as e:  # noqa: BLE001
        log.exception("failed to start perception loop")
        return {"started": False, "reason": f"{type(e).__name__}: {e}"}
    return {"started": True, "status": perception_loop.status.__dict__}


@app.post("/perception/stop")
async def post_perception_stop() -> dict[str, Any]:
    if perception_loop is None or not perception_loop.running:
        return {"stopped": False, "reason": "not running"}
    await perception_loop.stop()
    return {"stopped": True}


@app.get("/perception/status")
async def get_perception_status() -> dict[str, Any]:
    if perception_loop is None:
        return {"available": False}
    return {"available": True, **perception_loop.status.__dict__}


@app.post("/perception/lifted")
async def push_lifted(payload: dict[str, Any]) -> dict[str, Any]:
    """Accept a Phase 3/4 lifted JSON: track ids, fuse into the SemanticMap,
    broadcast the resulting map as ``world_update``.

    Expects ``{"objects": [ObjectState3D, ...], "frame_id": int?}``.
    A POST without ``frame_id`` auto-advances by one — useful for stitching
    independent snapshot runs into a track.
    """
    raw = payload.get("objects", [])
    frame_id = int(payload.get("frame_id", semantic_map.last_frame_id + 1))

    detections: list[ObjectState3D] = []
    for o in raw:
        if not o.get("center_3d_world"):
            continue
        try:
            detections.append(ObjectState3D(**o))
        except Exception as e:
            log.warning("dropping unparseable object: %s", e)
    tracked = tracker.update(detections, frame_id)
    semantic_map.update(tracked, frame_id)

    markers = _map_to_markers(semantic_map)
    graph = scene_graph_builder.build(semantic_map, frame_id=frame_id)
    action = PetAction(
        action="world_update",
        world_objects=markers,
        scene_graph=graph.to_dict(),
    )
    runtime._broadcast(action)  # noqa: SLF001 — runtime exposes this for sibling modules
    return {
        "applied": True,
        "frame_id": frame_id,
        "raw_detections": len(detections),
        "tracked": len(tracked),
        "map_size": len(markers),
        "relations": len(graph.relations),
    }


@app.get("/semantic/map")
async def get_semantic_map() -> dict[str, Any]:
    return semantic_map.to_dict()


@app.get("/scene/graph")
async def get_scene_graph() -> dict[str, Any]:
    return scene_graph_builder.build(semantic_map).to_dict()


@app.get("/planning/occupancy")
async def get_occupancy_grid() -> dict[str, Any]:
    """Phase 7 debug overlay: current occupancy grid serialised for the UI.

    Excludes nothing; pass ``?exclude=track_001`` later if we add per-target
    visualisation. Returned shape matches ``OccupancyGrid.to_dict``.
    """
    from ..planning import build_occupancy_grid

    grid = build_occupancy_grid(semantic_map, cfg=planner.cfg.grid)
    return grid.to_dict()


class SaveSemanticMapRequest(BaseModel):
    path: str | None = None


@app.post("/semantic/save")
async def save_semantic_map(req: SaveSemanticMapRequest) -> dict[str, Any]:
    """Persist the current SemanticMap to disk.

    Defaults to ``PET_AGENT_SEMANTIC_MAP_PATH`` (``runs/last_map.json``)
    so server restarts auto-restore the same snapshot on boot. Pass
    ``{"path": "..."}`` to override.
    """
    target = Path(req.path) if req.path else _SEMANTIC_MAP_PATH
    try:
        semantic_map.save(target)
    except Exception as e:  # noqa: BLE001
        log.exception("SemanticMap save failed")
        return {"saved": False, "reason": f"{type(e).__name__}: {e}"}
    return {
        "saved": True,
        "path": str(target),
        "objects": len(semantic_map.values()),
    }


@app.post("/semantic/reset")
async def reset_semantic_map() -> dict[str, Any]:
    tracker.reset()
    semantic_map.reset()
    runtime._broadcast(  # noqa: SLF001
        PetAction(action="world_update", world_objects=[], scene_graph=None)
    )
    return {"reset": True}


class CommandRequest(BaseModel):
    text: str


@app.post("/command")
async def push_command(req: CommandRequest) -> dict[str, Any]:
    """Phase 6: parse a user utterance and ground it against the live map.

    On success, emits an ``ask`` (explanation in the speech bubble) plus a
    planned ``move_follow_path``. Phase 7's planner consumes the
    NavigationGoal, builds an occupancy grid from the live SemanticMap, runs
    A* + LOS smoothing, and hands the path here. If the planner can't reach
    the goal we still emit the ``ask`` so the user gets the structured
    failure reason — no silent fallthrough.
    """
    intent = parse_command(req.text)
    if intent is None:
        runtime.ask(f"I didn't catch that: {req.text!r}")
        return {"parsed": False, "reason": "unparseable"}

    graph = scene_graph_builder.build(semantic_map)
    result = grounding_resolver.resolve(intent, semantic_map, graph)
    log.info(
        "command: %r → intent=%s status=%s",
        req.text,
        intent.intent_type,
        result.status,
    )

    if result.status == "clarification":
        runtime.ask(result.explanation)
        return {
            "parsed": True,
            "intent": intent.model_dump(),
            "status": result.status,
            "candidates": result.candidates,
        }

    if result.status == "empty_map":
        runtime.ask(result.explanation)
        return {"parsed": True, "intent": intent.model_dump(), "status": result.status}

    if result.status == "no_match":
        runtime.ask(result.explanation or "I don't see that.")
        return {"parsed": True, "intent": intent.model_dump(), "status": result.status}

    # success
    goal = result.goal
    assert goal is not None
    runtime.ask(goal.explanation)

    if intent.intent_type == "stop":
        runtime.play_animation("sit")
        return {
            "parsed": True,
            "intent": intent.model_dump(),
            "status": result.status,
            "goal": goal.model_dump(),
        }

    if intent.intent_type == "look_at" and goal.target_position_world is not None:
        x, y, z = goal.target_position_world
        runtime.look_at(x, y, z)
        return {
            "parsed": True,
            "intent": intent.model_dump(),
            "status": result.status,
            "goal": goal.model_dump(),
        }

    if intent.intent_type == "explore":
        # Phase 9: pick a viewpoint goal from the live coverage + map and
        # drive to it via the standard planner → controller pipeline.
        target_class = intent.target.class_label if intent.target else None
        ex_goal = exploration_planner.next_goal(
            semantic_map,
            coverage_grid,
            cat_xz=(runtime.state.position.x, runtime.state.position.z),
            target_class=target_class,
        )
        if ex_goal is None:
            runtime.ask("The map looks fully explored — nothing more to inspect.")
            return {
                "parsed": True,
                "intent": intent.model_dump(),
                "status": "fully_explored",
                "goal": goal.model_dump(),
            }
        return _execute_exploration_goal(intent, ex_goal)

    if goal.target_position_world is None:
        # No-target goals (report etc.) — caller-side animation only.
        return {
            "parsed": True,
            "intent": intent.model_dump(),
            "status": result.status,
            "goal": goal.model_dump(),
        }

    plan = planner.plan(goal, semantic_map, start_world=runtime.state.position.as_tuple())
    if plan.status != "success" or not plan.path_world:
        # Surface the planner's structured failure to the user instead of
        # silently teleporting the cat.
        runtime.ask(f"I can't plan a path there ({plan.status}).")
        return {
            "parsed": True,
            "intent": intent.model_dump(),
            "status": "plan_failed",
            "planner_status": plan.status,
            "goal": goal.model_dump(),
        }

    # Phase 8: run the offline pure-pursuit simulator over the planner's
    # waypoints so the renderer receives a dense, dynamically-feasible
    # trajectory rather than chunky LOS-pruned segments. The cat's heading
    # comes from the first non-degenerate segment of the planner path (no
    # heading is carried in PetState today; this is a deliberate trade-off
    # so the controller stays stateless across commands).
    initial = _initial_state_from_plan(plan.path_world)
    trace = path_follower.simulate(plan.path_world, initial)
    global _last_trace_summary
    _last_trace_summary = _trace_summary_dict(trace.summary, plan.path_world)
    runtime.move_follow_path(
        trace.path_world,
        speed=_ctrl_cfg.pure_pursuit.base_speed,
        look_at_object_id=goal.target_object_id,
        controller_trace=_last_trace_summary,
    )
    return {
        "parsed": True,
        "intent": intent.model_dump(),
        "status": result.status,
        "goal": goal.model_dump(),
        "planner_status": plan.status,
        "path_waypoints": len(plan.path_world),
        "control_steps": trace.summary.steps,
        "control_status": trace.summary.status,
    }


def _execute_exploration_goal(intent: Any, ex_goal: ExplorationGoal) -> dict[str, Any]:
    """Route an :class:`ExplorationGoal` through the planner + controller.

    Failures from the planner are surfaced via ``runtime.ask`` and reported
    back to the caller so the user sees structured feedback rather than a
    silent no-op. On success, any objects newly discovered since the last
    exploration tick are mentioned in the speech bubble.
    """
    global _last_exploration_ids, _last_trace_summary
    nav_goal = ex_goal.to_navigation_goal()
    runtime.ask(ex_goal.explanation)
    plan = planner.plan(nav_goal, semantic_map, start_world=runtime.state.position.as_tuple())
    if plan.status != "success" or not plan.path_world:
        runtime.ask(f"I can't plan a path there ({plan.status}).")
        return {
            "parsed": True,
            "intent": intent.model_dump(),
            "status": "plan_failed",
            "planner_status": plan.status,
            "exploration_goal": {
                "kind": ex_goal.kind,
                "target_position_world": list(ex_goal.target_position_world),
                "score": ex_goal.score,
                "related_object_id": ex_goal.related_object_id,
            },
        }
    initial = _initial_state_from_plan(plan.path_world)
    trace = path_follower.simulate(plan.path_world, initial)
    _last_trace_summary = _trace_summary_dict(trace.summary, plan.path_world)

    current_ids = {o.object_id for o in semantic_map.values()}
    discovered = sorted(current_ids - _last_exploration_ids)
    _last_exploration_ids = current_ids
    if discovered:
        runtime.ask("I found: " + ", ".join(discovered[:5]) + ("…" if len(discovered) > 5 else ""))

    goal_payload = {
        "kind": ex_goal.kind,
        "target_position_world": list(ex_goal.target_position_world),
        "score": round(ex_goal.score, 4),
        "related_object_id": ex_goal.related_object_id,
        "explanation": ex_goal.explanation,
    }
    runtime.move_follow_path(
        trace.path_world,
        speed=_ctrl_cfg.pure_pursuit.base_speed,
        look_at_object_id=ex_goal.related_object_id,
        controller_trace=_last_trace_summary,
        exploration_goal=goal_payload,
    )
    return {
        "parsed": True,
        "intent": intent.model_dump(),
        "status": "success",
        "exploration_goal": {
            "kind": ex_goal.kind,
            "target_position_world": list(ex_goal.target_position_world),
            "score": ex_goal.score,
            "related_object_id": ex_goal.related_object_id,
            "explanation": ex_goal.explanation,
        },
        "planner_status": plan.status,
        "path_waypoints": len(plan.path_world),
        "control_steps": trace.summary.steps,
        "control_status": trace.summary.status,
        "discovered_object_ids": discovered,
    }


def _initial_state_from_plan(path_world: list[tuple[float, float, float]]) -> UnicycleState:
    """Build an :class:`UnicycleState` from the cat's current position and the
    planner-supplied path direction.

    PetState doesn't carry yaw, so we derive heading from the first
    non-degenerate segment of the planner path. This biases the controller
    toward the path's natural direction on the first tick — small heading
    errors are then absorbed by the pure-pursuit P-gain over the rest of
    the trajectory.
    """
    pos = runtime.state.position
    theta = 0.0
    for a, b in zip(path_world[:-1], path_world[1:], strict=True):
        dx, dz = b[0] - a[0], b[2] - a[2]
        if (dx * dx + dz * dz) > 1e-9:
            theta = math.atan2(dz, dx)
            break
    return UnicycleState(x=pos.x, y=pos.z, theta=theta)


def _trace_summary_dict(
    summary: Any, planner_path: list[tuple[float, float, float]]
) -> dict[str, Any]:
    return {
        "status": summary.status,
        "steps": summary.steps,
        "duration_s": round(summary.duration_s, 3),
        "final_distance_to_goal": round(summary.final_distance_to_goal, 4),
        "max_cross_track_error": round(summary.max_cross_track_error, 4),
        "max_heading_error": round(summary.max_heading_error, 4),
        "mean_speed": round(summary.mean_speed, 4),
        "planner_waypoints": len(planner_path),
    }


@app.get("/control/last_trace")
async def get_last_control_trace() -> dict[str, Any]:
    """Phase 8 debug endpoint. Empty dict when no command has been planned yet."""
    return _last_trace_summary or {}


class ObserveConeRequest(BaseModel):
    camera_xz: tuple[float, float]
    heading_rad: float
    fov_rad: float
    range_m: float


@app.post("/exploration/observe")
async def post_observe(req: ObserveConeRequest) -> dict[str, Any]:
    """Mark a viewpoint cone as observed in the coverage grid."""
    new_cells = coverage_grid.observe_cone(
        camera_xz=req.camera_xz,
        heading=req.heading_rad,
        fov_rad=req.fov_rad,
        range_m=req.range_m,
    )
    return {
        "new_cells": new_cells,
        "unobserved_ratio": coverage_grid.unobserved_ratio(),
    }


class ExplorationStepRequest(BaseModel):
    target_class: str | None = None


@app.post("/exploration/step")
async def post_exploration_step(req: ExplorationStepRequest) -> dict[str, Any]:
    """Pick the next exploration goal and drive to it via planner+controller."""
    ex_goal = exploration_planner.next_goal(
        semantic_map,
        coverage_grid,
        cat_xz=(runtime.state.position.x, runtime.state.position.z),
        target_class=req.target_class,
    )
    if ex_goal is None:
        runtime.ask("The map looks fully explored — nothing more to inspect.")
        return {"status": "fully_explored"}

    # Fake an "intent.model_dump()" wrapper since this endpoint bypasses
    # /command. We pass a minimal stand-in object the helper can dump.
    class _Stub:
        def model_dump(self) -> dict[str, Any]:
            return {
                "intent_type": "explore",
                "target_class": req.target_class,
                "source": "exploration/step",
            }

    return _execute_exploration_goal(_Stub(), ex_goal)


@app.get("/exploration/coverage")
async def get_coverage_grid() -> dict[str, Any]:
    return coverage_grid.to_dict()


@app.post("/exploration/reset")
async def reset_exploration() -> dict[str, Any]:
    global _last_exploration_ids
    coverage_grid.reset()
    _last_exploration_ids = set()
    return {"reset": True}


class SimulateRequest(BaseModel):
    path: list[Waypoint]
    start: Waypoint | None = None
    start_theta: float | None = None


@app.post("/control/simulate")
async def simulate_path(req: SimulateRequest) -> dict[str, Any]:
    """Offline simulation of the controller against an arbitrary path.

    Useful for the demo notebook + acceptance harness; does not touch
    pet state or broadcast anything.
    """
    if not req.path:
        return {"status": "empty_path"}
    if req.start is not None:
        sx, _, sz = req.start
    else:
        pos = runtime.state.position
        sx, sz = pos.x, pos.z
    theta = req.start_theta if req.start_theta is not None else 0.0
    initial = UnicycleState(x=sx, y=sz, theta=theta)
    trace = path_follower.simulate(req.path, initial)
    return {
        "summary": _trace_summary_dict(trace.summary, req.path),
        "path_world": trace.path_world,
    }


def _map_to_markers(m: SemanticMap) -> list[dict[str, Any]]:
    """Convert SemanticMap → frontend marker payload (stable order)."""
    out: list[dict[str, Any]] = []
    for o in m.values():
        c = o.center_3d_world
        out.append(
            {
                "object_id": o.object_id,
                "class_label": o.class_label,
                "center_3d_world": [float(c[0]), float(c[1]), float(c[2])],
                "extent_3d": list(o.extent_3d),
                "median_depth": o.median_depth,
                "depth_uncertainty": o.depth_uncertainty,
                "confidence": o.confidence.overall,
                "tracking_status": o.tracking_status,
                "last_seen_frame": o.last_seen_frame,
            }
        )
    return out


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
