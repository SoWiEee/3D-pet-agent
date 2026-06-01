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


class PetAction(BaseModel):
    """A single broadcast event — matches spec §3.4.5 (PetAction)."""

    action: Literal[
        "move_to", "look_at", "play_animation", "set_emotion", "ask", "state"
    ]
    target_position_3d: tuple[float, float, float] | None = None
    look_at_object_id: str | None = None
    animation: Animation | None = None
    emotion: Emotion | None = None
    speed: float | None = None
    speech: str | None = None
    state: PetState | None = None
    timestamp: float = Field(default_factory=time.time)


class PetRuntime:
    """In-process pet controller. Broadcasts every change to all subscribers."""

    def __init__(self) -> None:
        self.state = PetState()
        self._subscribers: list[asyncio.Queue[PetAction]] = []

    # ── subscription ────────────────────────────────────────────────────────
    def subscribe(self) -> asyncio.Queue[PetAction]:
        q: asyncio.Queue[PetAction] = asyncio.Queue(maxsize=64)
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[PetAction]) -> None:
        if q in self._subscribers:
            self._subscribers.remove(q)

    def _broadcast(self, action: PetAction) -> None:
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

    # ── batch / scripting ───────────────────────────────────────────────────
    def apply(self, action: dict[str, Any]) -> PetAction:
        """Execute an action dict from a script file (jsonl)."""
        kind = action.get("action")
        if kind == "move_to":
            x, y, z = action["target_position_3d"]
            return self.move_to(x, y, z, speed=action.get("speed"))
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
