"""Stage E (§14.5) — the `pick up X` command flow end-to-end through the API.

Seeds a graspable object, fires `pick up the cup`, and asserts the server
broadcasts both a `move_follow_path` (drive there) and a `pick_object` (the arm
sequence the robot avatar animates).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.language import parse_command
from src.runtime.websocket_server import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        # Start each test from a clean map.
        c.post("/semantic/reset")
        yield c
        c.post("/semantic/reset")


def _seed(client, oid, label, x, z, ext=(0.1, 0.1, 0.1)):
    obj = {
        "object_id": oid,
        "class_label": label,
        "attributes": [],
        "bbox_xyxy": [100, 100, 200, 200],
        "center_2d": [150, 150],
        "coordinate_frame": "world",
        "center_3d_world": [x, 0.0, z],
        "extent_3d": list(ext),
        "median_depth": abs(z) or 1.0,
        "depth_uncertainty": 0.05,
        "source_backend": "mainline_grounding_sam",
        "confidence": {
            "detector": 0.9,
            "mask_quality": 0.85,
            "depth_quality": 0.8,
            "tracking": 1.0,
            "overall": 0.85,
        },
        "last_seen_frame": 0,
        "tracking_status": "tracked",
    }
    return client.post("/perception/lifted", json={"objects": [obj], "frame_id": 0})


# ── parser ──────────────────────────────────────────────────────────────────


def test_parser_recognises_pick_phrasings():
    for text in ["pick up the cup", "grab the cup", "pick the cup up", "grasp the cup"]:
        intent = parse_command(text)
        assert intent is not None, text
        assert intent.intent_type == "pick_up", text
        assert intent.target.class_label == "cup"


def test_go_to_is_not_a_pick():
    assert parse_command("go to the cup").intent_type == "move_to"


# ── /command pick flow ──────────────────────────────────────────────────────


def test_pick_command_returns_pick_payload(client):
    _seed(client, "cup1", "cup", 0.6, -0.4, ext=(0.1, 0.1, 0.1))
    r = client.post("/command", json={"text": "pick up the cup"}).json()
    assert r["parsed"] is True
    assert r["status"] == "success"
    assert r["intent"]["intent_type"] == "pick_up"
    # The grasp sequence rode along.
    assert "pick" in r
    assert r["pick"]["actions"] == 6
    assert r["pick"]["grasp"]["target_object_id"] == "track_001"


def test_pick_broadcasts_move_then_pick_object(client):
    _seed(client, "cup1", "cup", 0.6, -0.4, ext=(0.1, 0.1, 0.1))
    with client.websocket_connect("/ws/pet") as ws:
        # Drain the sticky world_update sent on connect.
        ws.receive_json()
        client.post("/command", json={"text": "pick up the cup"})
        kinds = []
        for _ in range(8):
            msg = ws.receive_json()
            kinds.append(msg["action"])
            if msg["action"] == "pick_object":
                assert msg["target_object_id"] == "track_001"
                assert msg["grasp"]["gripper_width"] > 0
                assert len(msg["manipulation_actions"]) == 6
                break
        assert "move_follow_path" in kinds
        assert "pick_object" in kinds


def test_oversized_object_is_not_grippable(client):
    # A keyboard wider than the gripper on both horizontal axes → no grip.
    _seed(client, "kbd1", "keyboard", 0.6, -0.4, ext=(0.4, 0.05, 0.3))
    r = client.post("/command", json={"text": "pick up the keyboard"}).json()
    assert r["status"] == "success"  # navigation still succeeds
    assert r["pick"] == {"feasible": False}
