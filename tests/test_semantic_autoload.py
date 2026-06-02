"""Phase B2 — SemanticMap autoload + save round-trip tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.runtime import websocket_server as srv
from src.runtime.websocket_server import _try_autoload_semantic_map
from src.spatial import SemanticMap
from src.spatial.object_lifter import ObjectConfidence, ObjectState3D


def _make_obj(object_id: str, class_label: str = "cup") -> ObjectState3D:
    return ObjectState3D(
        object_id=object_id,
        class_label=class_label,
        bbox_xyxy=(0.0, 0.0, 10.0, 10.0),
        center_2d=(5.0, 5.0),
        center_3d_world=(0.5, 0.0, 0.6),
        extent_3d=(0.08, 0.12, 0.08),
        median_depth=1.0,
        depth_uncertainty=0.05,
        confidence=ObjectConfidence(overall=0.85),
        tracking_status="tracked",
        last_seen_frame=1,
        source_backend="mainline_grounding_sam",
    )


def _populated_map() -> SemanticMap:
    m = SemanticMap(map_id="autoload_test")
    m.update([_make_obj("cup_001"), _make_obj("kbd_001", "keyboard")], frame_id=1)
    return m


# ── _try_autoload_semantic_map ────────────────────────────────────────────


def test_autoload_silent_when_file_missing(tmp_path: Path) -> None:
    target = SemanticMap()
    n = _try_autoload_semantic_map(target, tmp_path / "missing.json")
    assert n == 0
    assert len(target.values()) == 0


def test_autoload_populates_target_in_place(tmp_path: Path) -> None:
    snapshot = tmp_path / "map.json"
    _populated_map().save(snapshot)

    target = SemanticMap()
    target_id_before = id(target)
    n = _try_autoload_semantic_map(target, snapshot)
    assert n == 2
    # Same reference — caller's pointer still valid.
    assert id(target) == target_id_before
    ids = {o.object_id for o in target.values()}
    assert ids == {"cup_001", "kbd_001"}


def test_autoload_replaces_existing_contents(tmp_path: Path) -> None:
    snapshot = tmp_path / "map.json"
    _populated_map().save(snapshot)

    target = SemanticMap()
    target.update([_make_obj("stale_001", "stale")], frame_id=99)
    assert len(target.values()) == 1

    _try_autoload_semantic_map(target, snapshot)
    ids = {o.object_id for o in target.values()}
    assert "stale_001" not in ids
    assert ids == {"cup_001", "kbd_001"}


def test_autoload_silent_on_corrupt_snapshot(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{ not valid json", encoding="utf-8")

    target = SemanticMap()
    target.update([_make_obj("orig_001")], frame_id=1)

    n = _try_autoload_semantic_map(target, bad)
    assert n == 0
    # Existing state preserved on failure — caller can still serve.
    assert len(target.values()) == 1


# ── /semantic/save endpoint ──────────────────────────────────────────────


@pytest.fixture
def client() -> TestClient:
    srv.semantic_map.reset()
    srv.tracker.reset()
    return TestClient(srv.app)


def test_save_endpoint_writes_snapshot_to_default_path(
    client: TestClient, monkeypatch, tmp_path: Path
) -> None:
    target = tmp_path / "last_map.json"
    monkeypatch.setattr(srv, "_SEMANTIC_MAP_PATH", target)
    # Populate the map directly via /perception/lifted.
    body = {
        "objects": [
            {
                "object_id": "cup_001",
                "class_label": "cup",
                "bbox_xyxy": [0, 0, 10, 10],
                "center_2d": [5, 5],
                "center_3d_world": [0.5, 0.0, 0.6],
                "extent_3d": [0.08, 0.12, 0.08],
                "median_depth": 1.0,
                "depth_uncertainty": 0.05,
                "source_backend": "mainline_grounding_sam",
                "confidence": {"overall": 0.85},
                "last_seen_frame": 1,
                "tracking_status": "tracked",
            }
        ]
    }
    client.post("/perception/lifted", json=body)

    resp = client.post("/semantic/save", json={})
    body = resp.json()
    assert body["saved"] is True
    assert body["objects"] == 1
    assert target.exists()


def test_save_endpoint_honours_explicit_path(
    client: TestClient, tmp_path: Path
) -> None:
    custom = tmp_path / "custom.json"
    resp = client.post("/semantic/save", json={"path": str(custom)})
    assert resp.json()["saved"] is True
    assert custom.exists()


def test_save_round_trip_via_endpoint_and_autoload(
    client: TestClient, tmp_path: Path
) -> None:
    """End-to-end: save via endpoint → reset map → autoload populates it."""
    custom = tmp_path / "rt.json"
    obj = {
        "object_id": "bookend_001",
        "class_label": "bookend",
        "bbox_xyxy": [0, 0, 10, 10],
        "center_2d": [5, 5],
        "center_3d_world": [1.0, 0.0, 1.0],
        "extent_3d": [0.1, 0.1, 0.1],
        "median_depth": 1.0,
        "depth_uncertainty": 0.05,
        "source_backend": "mainline_grounding_sam",
        "confidence": {"overall": 0.7},
        "last_seen_frame": 1,
        "tracking_status": "tracked",
    }
    client.post("/perception/lifted", json={"objects": [obj]})
    client.post("/semantic/save", json={"path": str(custom)})

    # Simulate server restart by clearing the in-process map.
    srv.semantic_map.reset()
    assert len(srv.semantic_map.values()) == 0

    n = _try_autoload_semantic_map(srv.semantic_map, custom)
    assert n == 1
    # Tracker rewrites the incoming id; the surviving class label is the
    # invariant that round-trips intact through save/load.
    restored = next(iter(srv.semantic_map.values()))
    assert restored.class_label == "bookend"
