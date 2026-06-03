"""Phase A1 — live perception background loop.

Spec §13 / `docs/review.md` A1: the standalone `--mode demo` only spun up
the FastAPI server. Real-time webcam → perception → tracker → SemanticMap
→ broadcast wasn't wired up, so the demo could only be driven by curl-ing
pre-computed lifted JSON.

This module owns one background asyncio task that:

1. Lazily instantiates :class:`PerceptionPipeline` and :class:`Webcam`
   (heavy models — only loaded when the user explicitly starts the loop).
2. Reads a frame, runs `run_frame_tracked`, fuses into the shared
   SemanticMap.
3. Broadcasts a ``world_update`` PetAction with the latest markers +
   scene graph.
4. Sleeps to maintain the configured rate (frame drops are OK; we never
   queue past the current frame so latency stays bounded).

The loop is **opt-in**: instantiating ``PerceptionLoop`` does nothing
heavy; only :meth:`start` triggers model loading and the background task.

Thread / process safety: the loop runs in the same asyncio loop as the
FastAPI app and mutates the shared `tracker` / `semantic_map` /
`scene_graph_builder`. That's safe because Python's GIL serialises the
hot path and the FastAPI request handlers `await` between mutations — no
two perception ticks run concurrently.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from ..config import AppConfig
from ..spatial import SceneGraphBuilder, SemanticMap
from ..spatial.frame_packet import CameraIntrinsics
from ..spatial.pose_source import FixedPoseSource, PoseSource
from ..tracking import Tracker

log = logging.getLogger("pet_agent.perception_loop")


@dataclass
class PerceptionLoopStatus:
    """What's currently happening inside the loop. Returned by the API."""

    running: bool = False
    camera_index: int = 0
    prompts: list[str] = field(default_factory=list)
    target_hz: float = 2.0
    frames_processed: int = 0
    last_frame_id: int = 0
    last_frame_ms: float = 0.0
    last_error: str | None = None
    started_at: float | None = None


class PerceptionLoop:
    """Background loop that drives the live perception pipeline.

    Owns nothing the server doesn't already own — instead, it takes the
    shared tracker / semantic_map / scene_graph_builder + a broadcast
    callback so each tick can publish via the existing PetRuntime fanout.
    """

    def __init__(
        self,
        *,
        cfg: AppConfig,
        tracker: Tracker,
        semantic_map: SemanticMap,
        scene_graph_builder: SceneGraphBuilder,
        broadcast: callable,  # type: ignore[valid-type]
        markers_fn: callable,  # type: ignore[valid-type]
    ) -> None:
        self.cfg = cfg
        self.tracker = tracker
        self.semantic_map = semantic_map
        self.scene_graph_builder = scene_graph_builder
        self.broadcast = broadcast
        self.markers_fn = markers_fn

        self._task: asyncio.Task[None] | None = None
        self._stop_event = asyncio.Event()
        self.status = PerceptionLoopStatus(
            target_hz=cfg.runtime.runtime.perception_update_hz,
        )

        # Lazily created; only when start() is called.
        self._pipeline = None
        self._webcam = None
        self._pose_source: PoseSource = FixedPoseSource()
        self._intrinsics: CameraIntrinsics | None = None
        self._prompts: list[str] = []

    # ── lifecycle ─────────────────────────────────────────────────────────
    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def start(
        self,
        *,
        prompts: list[str],
        camera_index: int = 0,
        fov_deg: float = 60.0,
        hz: float | None = None,
        webcam_factory: Any | None = None,
        pipeline_factory: Any | None = None,
    ) -> None:
        """Boot perception models + open webcam + spawn the background task.

        ``webcam_factory`` / ``pipeline_factory`` are injection points for
        tests — the production path uses :class:`Webcam` and
        :class:`PerceptionPipeline` directly so they only import torch when
        the user actually starts the loop.
        """
        if self.running:
            raise RuntimeError("perception loop already running")
        self.status.last_error = None
        self.status.camera_index = camera_index
        self.status.prompts = prompts
        if hz is not None:
            self.status.target_hz = hz
        self._prompts = prompts

        # Lazy-load heavy modules — only here, never at import time.
        if pipeline_factory is None:
            from ..perception.pipeline import PerceptionPipeline

            self._pipeline = PerceptionPipeline(self.cfg)
        else:
            self._pipeline = pipeline_factory(self.cfg)

        if webcam_factory is None:
            from ..camera_service.webcam import Webcam

            self._webcam = Webcam(index=camera_index)
        else:
            self._webcam = webcam_factory(camera_index)

        # Probe one frame to size intrinsics correctly.
        frame = await asyncio.to_thread(self._webcam.read)
        self._intrinsics = CameraIntrinsics.from_fov(
            image_size=frame.shape[:2], horizontal_fov_deg=fov_deg
        )

        # Optional Visual SLAM sidecar (spec §14.1) — opt-in via config; the
        # default keeps the camera fixed at the world origin. Lazy-imported so
        # cv2/scipy only load when SLAM is actually requested.
        self._pose_source = self._make_pose_source(self._intrinsics)

        self._stop_event.clear()
        self.status.running = True
        self.status.started_at = time.time()
        self.status.frames_processed = 0
        self._task = asyncio.create_task(self._run_forever())
        log.info(
            "perception loop started: camera=%d hz=%.1f prompts=%d",
            camera_index,
            self.status.target_hz,
            len(prompts),
        )

    def _make_pose_source(self, intrinsics: CameraIntrinsics) -> PoseSource:
        """Select the pose source from config. ``slam`` enables the ORB visual
        odometry sidecar; anything else keeps the fixed origin pose."""
        if self.cfg.settings.pose_source == "slam":
            from ..research.slam_adapter import SLAMPoseSource

            log.info("perception loop using SLAM pose source (ORB visual odometry)")
            return SLAMPoseSource(intrinsics)
        return FixedPoseSource()

    async def stop(self) -> None:
        """Cancel the background task and close the webcam."""
        if not self.running:
            return
        self._stop_event.set()
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
            except TimeoutError:
                log.warning("perception loop did not stop in 5 s, cancelling")
                self._task.cancel()
        self._cleanup()

    def _cleanup(self) -> None:
        self.status.running = False
        self._task = None
        if self._webcam is not None:
            try:
                self._webcam.close()
            except Exception as e:
                log.warning("webcam close error: %s", e)
            self._webcam = None
        self._pipeline = None

    # ── hot loop ──────────────────────────────────────────────────────────
    async def _run_forever(self) -> None:
        """One frame per tick — drop frames if processing is slower than hz.

        We never queue: the loop reads → processes → broadcasts → sleeps.
        If processing takes longer than the period, the next read just
        returns the freshest frame, which is what we want for a live demo.
        """
        period = 1.0 / max(self.status.target_hz, 0.1)
        try:
            while not self._stop_event.is_set():
                t0 = time.perf_counter()
                try:
                    await self._tick()
                except Exception as e:  # noqa: BLE001 — never let one bad frame kill the loop
                    log.exception("perception tick failed")
                    self.status.last_error = f"{type(e).__name__}: {e}"
                elapsed = time.perf_counter() - t0
                self.status.last_frame_ms = round(elapsed * 1000.0, 2)
                sleep_for = max(0.0, period - elapsed)
                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=sleep_for)
                    break  # stop_event was set during sleep
                except TimeoutError:
                    continue
        finally:
            self._cleanup()
            log.info("perception loop stopped after %d frames", self.status.frames_processed)

    async def _tick(self) -> None:
        """Run one perception → broadcast cycle. Off-thread for the heavy bit."""
        assert self._pipeline is not None
        assert self._webcam is not None
        assert self._intrinsics is not None

        frame_id = self.status.last_frame_id + 1
        frame = await asyncio.to_thread(self._webcam.read)

        # Streaming pose sources (SLAM) need the raw frame pushed in before the
        # pipeline queries the pose for this frame. Off-thread — ORB is CPU work.
        track = getattr(self._pose_source, "track", None)
        if track is not None:
            await asyncio.to_thread(track, frame_id, frame)

        # Heavy work (torch inference) runs in a thread so the event loop
        # stays responsive for WebSocket clients.
        result, _depth, tracked = await asyncio.to_thread(
            self._pipeline.run_frame_tracked,
            frame,
            self._prompts,
            tracker=self.tracker,
            semantic_map=self.semantic_map,
            frame_id=frame_id,
            intrinsics=self._intrinsics,
            pose_source=self._pose_source,
            save_masks=False,
        )
        _ = result, tracked  # used only for logging shape; broadcast uses map

        # Broadcast world_update — same shape that POST /perception/lifted emits.
        markers = self.markers_fn(self.semantic_map)
        graph = self.scene_graph_builder.build(self.semantic_map, frame_id=frame_id)
        action = self._make_world_update(markers, graph)
        self.broadcast(action)

        self.status.frames_processed += 1
        self.status.last_frame_id = frame_id

    def _make_world_update(self, markers: list[dict[str, Any]], graph: Any) -> Any:
        # Imported here so this module doesn't depend on PetAction at import
        # time (avoids a cycle with pet_runtime.py).
        from .pet_runtime import PetAction

        return PetAction(
            action="world_update",
            world_objects=markers,
            scene_graph=graph.to_dict(),
        )
