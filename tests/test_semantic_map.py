"""Phase 4 — SemanticMap fusion + persistence tests."""

from __future__ import annotations

from pathlib import Path

from src.spatial import SemanticMap
from src.spatial.object_lifter import ObjectState3D
from tests.factories import make_object


def _obs(
    *,
    object_id: str,
    label: str = "cup",
    center: tuple[float, float, float] = (0.0, 0.0, -2.0),
    overall: float = 0.7,
    frame_id: int = 0,
) -> ObjectState3D:
    return make_object(
        object_id=object_id,
        class_label=label,
        center_3d_world=center,
        last_seen_frame=frame_id,
        detector=overall,
        mask_quality=overall,
        depth_quality=overall,
        overall=overall,
    )


def test_first_observation_inserted_as_tracked() -> None:
    m = SemanticMap()
    m.update([_obs(object_id="track_001", frame_id=0)], frame_id=0)
    assert "track_001" in m.objects
    assert m.objects["track_001"].tracking_status == "tracked"
    assert m.last_frame_id == 0


def test_position_ema_pulls_toward_new_observation() -> None:
    m = SemanticMap(position_alpha=0.5)
    m.update([_obs(object_id="t1", center=(0.0, 0.0, 0.0), frame_id=0)], frame_id=0)
    m.update([_obs(object_id="t1", center=(2.0, 0.0, 0.0), frame_id=1)], frame_id=1)
    fused = m.objects["t1"].center_3d_world
    assert fused[0] == 1.0  # 0.5 * 2.0 + 0.5 * 0.0


def test_unseen_object_decays_through_status_machine() -> None:
    m = SemanticMap(persistence_frames=2, stale_frames=5, lost_frames=20, confidence_decay=0.1)
    m.update([_obs(object_id="t1", overall=0.9, frame_id=0)], frame_id=0)
    # Frame 1: gap of 1 ≤ persistence_frames → occluded.
    m.update([], frame_id=1)
    assert m.objects["t1"].tracking_status == "occluded"
    # Frame 4: gap of 4 > persistence (2) but ≤ stale (5) → stale.
    m.update([], frame_id=4)
    assert m.objects["t1"].tracking_status == "stale"
    # Frame 10: gap of 10 > stale (5) → lost.
    m.update([], frame_id=10)
    assert m.objects["t1"].tracking_status == "lost"
    # Confidence has decayed monotonically.
    assert m.objects["t1"].confidence.overall < 0.9


def test_lost_object_pruned_after_lost_frames() -> None:
    m = SemanticMap(persistence_frames=2, stale_frames=5, lost_frames=10)
    m.update([_obs(object_id="t1", frame_id=0)], frame_id=0)
    m.update([], frame_id=11)
    assert "t1" not in m.objects


def test_save_load_round_trip_byte_identical(tmp_path: Path) -> None:
    """Acceptance §7.2: map can be saved and reloaded byte-identically."""
    m = SemanticMap(map_id="t")
    m.update(
        [
            _obs(object_id="track_002", label="mouse", center=(1.0, 0.0, -2.0), frame_id=0),
            _obs(object_id="track_001", label="cup", center=(0.0, 0.0, -2.0), frame_id=0),
        ],
        frame_id=0,
    )
    m.update([_obs(object_id="track_001", center=(0.1, 0.0, -2.0), frame_id=1)], frame_id=1)

    path_a = tmp_path / "map.json"
    m.save(path_a)
    loaded = SemanticMap.load(path_a)
    path_b = tmp_path / "map_round.json"
    loaded.save(path_b)
    assert path_a.read_bytes() == path_b.read_bytes()
    assert sorted(loaded.objects) == sorted(m.objects)
    assert loaded.objects["track_001"].center_3d_world == m.objects["track_001"].center_3d_world


def test_reset_clears_objects() -> None:
    m = SemanticMap()
    m.update([_obs(object_id="t1", frame_id=0)], frame_id=0)
    m.reset()
    assert m.objects == {}
    assert m.last_frame_id == -1


def test_remove_drops_single_object_and_reports_existence() -> None:
    m = SemanticMap()
    m.update(
        [_obs(object_id="t1", frame_id=0), _obs(object_id="t2", frame_id=0)],
        frame_id=0,
    )
    assert m.remove("t1") is True
    assert "t1" not in m.objects
    assert "t2" in m.objects  # other objects untouched
    assert m.remove("t1") is False  # already gone


def test_object_leaving_view_remains_with_decayed_confidence() -> None:
    """Acceptance §7.2: an object that leaves view stays in the map with
    decayed confidence (not just dropped immediately)."""
    m = SemanticMap(persistence_frames=2, stale_frames=10, lost_frames=50, confidence_decay=0.05)
    m.update([_obs(object_id="t1", overall=0.9, frame_id=0)], frame_id=0)
    initial = m.objects["t1"].confidence.overall
    for fi in range(1, 6):
        m.update([], frame_id=fi)
    assert "t1" in m.objects
    decayed = m.objects["t1"].confidence.overall
    assert decayed < initial
    assert decayed > 0
