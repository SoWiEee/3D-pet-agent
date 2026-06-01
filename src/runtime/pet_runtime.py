"""3D pet runtime (Phase 1).

Owns the canonical pet state on the backend and provides the action API from spec §4.3.
The frontend is a renderer of this state — every state change is broadcast as a
PetAction event so the Three.js scene can react.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, Literal

from pydantic import BaseModel, Field

Animation = Literal["idle", "walk", "run", "look_at", "sit", "hide", "curious", "confused"]
Emotion = Literal["neutral", "happy", "curious", "confused", "scared", "playful"]


class Vec3(BaseModel):
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)


class PetState(BaseModel):
    """Authoritative pet state held by the backend."""

    position: Vec3 = Field(default_factory=Vec3)
    look_at: Vec3 | None = None
    animation: Animation = "idle"
    emotion: Emotion = "neutral"
    speed: float = 0.8
    speech: str | None = None
    updated_at: float = Field(default_factory=time.time)


Waypoint = tuple[float, float, float]


class PetAction(BaseModel):
    """A single broadcast event — matches spec §3.7 (PetAction, v2).

    Movement actions are split into:
      - ``move_to``           direct manual move; backend tweens A → B.
      - ``move_follow_path``  controller-produced path; frontend traverses
                              waypoints with smooth heading.
    """

    action: Literal[
        "move_to",
        "move_follow_path",
        "look_at",
        "play_animation",
        "set_emotion",
        "ask",
        "state",
        "world_update",
    ]
    target_position_3d: Waypoint | None = None
    path: list[Waypoint] | None = None
    look_at_object_id: str | None = None
    animation: Animation | None = None
    emotion: Emotion | None = None
    speed: float | None = None
    speech: str | None = None
    state: PetState | None = None
    # Phase 3: lifted 3D centroids broadcast as a world update.
    world_objects: list[dict[str, Any]] | None = None
    # Phase 5: scene graph snapshot (edges over world_objects) — piggybacks on
    # the world_update broadcast so renderers can draw both atomically.
    scene_graph: dict[str, Any] | None = None
    timestamp: float = Field(default_factory=time.time)


class PetRuntime:
    """In-process pet controller. Broadcasts every change to all subscribers."""

    def __init__(self) -> None:
        self.state = PetState()
        self._subscribers: list[asyncio.Queue[PetAction]] = []
        # Latest sticky world_update so newly-connected subscribers see the
        # current scene markers without having to wait for the next push.
        self._last_world_update: PetAction | None = None

    # ── subscription ────────────────────────────────────────────────────────
    def subscribe(self) -> asyncio.Queue[PetAction]:
        q: asyncio.Queue[PetAction] = asyncio.Queue(maxsize=64)
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[PetAction]) -> None:
        if q in self._subscribers:
            self._subscribers.remove(q)

    def _broadcast(self, action: PetAction) -> None:
        if action.action == "world_update":
            self._last_world_update = action
        for q in list(self._subscribers):
            try:
                q.put_nowait(action)
            except asyncio.QueueFull:
                # Slow consumer — drop oldest, keep newest.
                try:
                    q.get_nowait()
                    q.put_nowait(action)
                except Exception:
                    pass

    # ── spec §4.3 action API ────────────────────────────────────────────────
    def move_to(self, x: float, y: float, z: float, speed: float | None = None) -> PetAction:
        self.state.position = Vec3(x=x, y=y, z=z)
        if speed is not None:
            self.state.speed = speed
        self.state.animation = "walk"
        self.state.updated_at = time.time()
        action = PetAction(
            action="move_to",
            target_position_3d=(x, y, z),
            animation="walk",
            speed=self.state.speed,
            state=self.state.model_copy(),
        )
        self._broadcast(action)
        return action

    def move_follow_path(
        self,
        path: list[Waypoint],
        *,
        speed: float | None = None,
        look_at_object_id: str | None = None,
    ) -> PetAction:
        """Walk along a planned path (spec §3.7 ``move_follow_path``).

        The authoritative pet state snaps to the final waypoint at broadcast
        time; the renderer is responsible for the smooth traversal. The
        controller is expected to send new ``move_follow_path`` events at its
        own update rate, replacing in-flight paths as the plan refines.
        """
        if not path:
            raise ValueError("path must contain at least one waypoint")
        end = path[-1]
        self.state.position = Vec3(x=end[0], y=end[1], z=end[2])
        if speed is not None:
            self.state.speed = speed
        self.state.animation = "walk"
        self.state.updated_at = time.time()
        action = PetAction(
            action="move_follow_path",
            path=[(float(x), float(y), float(z)) for x, y, z in path],
            target_position_3d=(float(end[0]), float(end[1]), float(end[2])),
            look_at_object_id=look_at_object_id,
            animation="walk",
            speed=self.state.speed,
            state=self.state.model_copy(),
        )
        self._broadcast(action)
        return action

    def look_at(self, x: float, y: float, z: float) -> PetAction:
        self.state.look_at = Vec3(x=x, y=y, z=z)
        self.state.updated_at = time.time()
        action = PetAction(
            action="look_at",
            target_position_3d=(x, y, z),
            state=self.state.model_copy(),
        )
        self._broadcast(action)
        return action

    def play_animation(self, name: Animation) -> PetAction:
        self.state.animation = name
        self.state.updated_at = time.time()
        action = PetAction(action="play_animation", animation=name, state=self.state.model_copy())
        self._broadcast(action)
        return action

    def set_emotion(self, name: Emotion) -> PetAction:
        self.state.emotion = name
        self.state.updated_at = time.time()
        action = PetAction(action="set_emotion", emotion=name, state=self.state.model_copy())
        self._broadcast(action)
        return action

    def ask(self, text: str) -> PetAction:
        self.state.speech = text
        self.state.updated_at = time.time()
        action = PetAction(action="ask", speech=text, state=self.state.model_copy())
        self._broadcast(action)
        return action

    def snapshot(self) -> PetAction:
        return PetAction(action="state", state=self.state.model_copy())

    def last_world_update(self) -> PetAction | None:
        return self._last_world_update

    # ── batch / scripting ───────────────────────────────────────────────────
    def apply(self, action: dict[str, Any]) -> PetAction:
        """Execute an action dict from a script file (jsonl)."""
        kind = action.get("action")
        if kind == "move_to":
            x, y, z = action["target_position_3d"]
            return self.move_to(x, y, z, speed=action.get("speed"))
        if kind == "move_follow_path":
            return self.move_follow_path(
                action["path"],
                speed=action.get("speed"),
                look_at_object_id=action.get("look_at_object_id"),
            )
        if kind == "look_at":
            x, y, z = action["target_position_3d"]
            return self.look_at(x, y, z)
        if kind == "play_animation":
            return self.play_animation(action["animation"])
        if kind == "set_emotion":
            return self.set_emotion(action["emotion"])
        if kind == "ask":
            return self.ask(action["speech"])
        raise ValueError(f"unknown action kind: {kind!r}")
