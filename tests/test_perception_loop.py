"""Phase A1 — live perception loop tests.

Heavy models (GroundingDINO + SAM + Depth Anything V2) are mocked via
the factory injection points so the test suite stays fast and CPU-only.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from src.config import AppConfig
from src.runtime.perception_loop import PerceptionLoop
from src.spatial import SceneGraphBuilder, SemanticMap
from src.spatial.object_lifter import ObjectState3D
from src.tracking import Tracker
from tests.factories import make_object


def _obj() -> ObjectState3D:
    return make_object(
        object_id="cup_001",
        class_label="cup",
        center_3d_world=(0.5, 0.0, 0.6),
        extent_3d=(0.08, 0.12, 0.08),
        bbox_xyxy=(0.0, 0.0, 10.0, 10.0),
        center_2d=(5.0, 5.0),
        median_depth=1.0,
        depth_uncertainty=0.05,
        last_seen_frame=1,
        detector=0.0,
        mask_quality=0.0,
        depth_quality=0.0,
        overall=0.85,
    )


class FakeWebcam:
    """Yields synthetic BGR frames, never raises."""

    def __init__(self, _index: int) -> None:
        self.read_count = 0

    def read(self) -> np.ndarray:
        self.read_count += 1
        return np.zeros((240, 320, 3), dtype=np.uint8)

    def close(self) -> None:
        pass


class FakePipeline:
    """Mimics PerceptionPipeline.run_frame_tracked without torch."""

    def __init__(self, _cfg: AppConfig) -> None:
        self.tick_count = 0

    def run_frame_tracked(
        self,
        _frame,
        _prompts,
        *,
        tracker: Tracker,
        semantic_map: SemanticMap,
        frame_id: int,
        intrinsics,
        pose_source,
        save_masks=False,
    ):
        self.tick_count += 1
        # Inject the same fake object every tick — tracker will keep one
        # stable track id, SemanticMap will fuse it in place.
        obj = _obj()
        tracked = tracker.update([obj], frame_id)
        semantic_map.update(tracked, frame_id)
        depth = np.zeros((240, 320), dtype=np.float32)
        return None, depth, tracked


@pytest.fixture
def loop_with_broadcast() -> tuple[PerceptionLoop, list[object]]:
    cfg = AppConfig.load()
    tracker = Tracker()
    semantic_map = SemanticMap()
    builder = SceneGraphBuilder()
    broadcasts: list[object] = []

    loop = PerceptionLoop(
        cfg=cfg,
        tracker=tracker,
        semantic_map=semantic_map,
        scene_graph_builder=builder,
        broadcast=lambda action: broadcasts.append(action),
        markers_fn=lambda m: [
            {"object_id": o.object_id, "class_label": o.class_label} for o in m.values()
        ],
    )
    return loop, broadcasts


@pytest.mark.asyncio
async def test_loop_runs_one_tick_and_broadcasts(loop_with_broadcast) -> None:
    loop, broadcasts = loop_with_broadcast
    await loop.start(
        prompts=["cup"],
        camera_index=0,
        fov_deg=60.0,
        hz=20.0,
        webcam_factory=FakeWebcam,
        pipeline_factory=FakePipeline,
    )
    # Let the loop tick a couple of times.
    await asyncio.sleep(0.25)
    await loop.stop()

    assert loop.status.frames_processed >= 1
    assert loop.status.last_frame_id >= 1
    # At least one world_update was broadcast.
    assert any(getattr(a, "action", None) == "world_update" for a in broadcasts)


@pytest.mark.asyncio
async def test_loop_double_start_rejected(loop_with_broadcast) -> None:
    loop, _ = loop_with_broadcast
    await loop.start(
        prompts=["cup"],
        webcam_factory=FakeWebcam,
        pipeline_factory=FakePipeline,
        hz=10.0,
    )
    with pytest.raises(RuntimeError):
        await loop.start(
            prompts=["cup"],
            webcam_factory=FakeWebcam,
            pipeline_factory=FakePipeline,
        )
    await loop.stop()


@pytest.mark.asyncio
async def test_loop_stop_idempotent(loop_with_broadcast) -> None:
    loop, _ = loop_with_broadcast
    # stop without start
    await loop.stop()
    assert not loop.running


@pytest.mark.asyncio
async def test_loop_recovers_from_tick_errors(loop_with_broadcast) -> None:
    """A bad frame must not kill the loop — the next tick should still run."""
    loop, broadcasts = loop_with_broadcast

    class FlakyPipeline(FakePipeline):
        def run_frame_tracked(self, *args, **kwargs):
            self.tick_count += 1
            if self.tick_count == 1:
                raise RuntimeError("synthetic perception error")
            return super().run_frame_tracked(*args, **kwargs)

    await loop.start(
        prompts=["cup"],
        webcam_factory=FakeWebcam,
        pipeline_factory=FlakyPipeline,
        hz=30.0,
    )
    await asyncio.sleep(0.3)
    await loop.stop()
    # We hit the error on tick 1 and still managed at least one successful
    # subsequent tick.
    assert loop.status.last_error is not None
    assert loop.status.frames_processed >= 1


@pytest.mark.asyncio
async def test_loop_status_fields_populated(loop_with_broadcast) -> None:
    loop, _ = loop_with_broadcast
    await loop.start(
        prompts=["cup"],
        camera_index=2,
        webcam_factory=FakeWebcam,
        pipeline_factory=FakePipeline,
        hz=15.0,
    )
    await asyncio.sleep(0.15)
    status = loop.status
    assert status.camera_index == 2
    assert status.prompts == ["cup"]
    assert status.target_hz == 15.0
    assert status.started_at is not None
    await loop.stop()


def test_make_pose_source_defaults_to_fixed(loop_with_broadcast) -> None:
    from src.spatial.frame_packet import CameraIntrinsics
    from src.spatial.pose_source import FixedPoseSource

    loop, _ = loop_with_broadcast
    intr = CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0)
    assert isinstance(loop._make_pose_source(intr), FixedPoseSource)


def test_make_pose_source_selects_slam_when_configured(loop_with_broadcast) -> None:
    from src.research.slam_adapter import SLAMPoseSource
    from src.spatial.frame_packet import CameraIntrinsics

    loop, _ = loop_with_broadcast
    loop.cfg.settings.pose_source = "slam"
    intr = CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0)
    assert isinstance(loop._make_pose_source(intr), SLAMPoseSource)


def test_make_pose_source_selects_graph_slam_when_configured(
    loop_with_broadcast,
) -> None:
    pytest.importorskip("pypose")  # graph_slam pose source needs the .[slam] extra
    from src.research.graph_slam import GraphSlamPoseSource
    from src.spatial.frame_packet import CameraIntrinsics

    loop, _ = loop_with_broadcast
    loop.cfg.settings.pose_source = "graph_slam"
    intr = CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0)
    assert isinstance(loop._make_pose_source(intr), GraphSlamPoseSource)
