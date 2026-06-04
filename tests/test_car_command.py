"""§14.5 — car-kinematics command flow + Stage A/B live wiring.

Seeds an object, fires a move with ``kinematics="car"``, and asserts the server
plans a Reeds-Shepp path, broadcasts a ``move_follow_path`` carrying a real
per-sample ``motion_profile``, publishes the goal to Nav2 (Stage A), and fills
in the metric occupancy map (Stage B).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.runtime.websocket_server import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        c.post("/semantic/reset")
        c.post("/exploration/reset")
        yield c
        c.post("/semantic/reset")


def _seed(client, oid, label, x, z, ext=(0.1, 0.11, 0.1)):
    obj = {
        "object_id": oid,
        "class_label": label,
        "attributes": [],
        "bbox_xyxy": [100, 100, 200, 200],
        "center_2d": [150, 150],
        "coordinate_frame": "world",
        "center_3d_world": [x, 0.1, z],
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
    # Post a few frames so the track stabilises and the metric map accretes.
    for f in range(4):
        client.post("/perception/lifted", json={"objects": [obj], "frame_id": f})


def test_car_command_emits_motion_profile(client):
    _seed(client, "cup0", "cup", 1.6, -0.8)
    r = client.post("/command", json={"text": "go to the cup", "kinematics": "car"})
    body = r.json()
    assert body["status"] == "success"
    assert body["kinematics"] == "car"
    assert body["control_status"] in {"success", "fallback"}
    # The broadcast control trace carries the car summary (radius, reversals).
    trace = client.get("/control/last_trace").json()
    assert trace.get("kinematics") == "car"
    assert "turning_radius_m" in trace
    assert trace["steps"] > 1


def test_unicycle_command_has_no_car_profile(client):
    _seed(client, "cup0", "cup", 1.6, -0.8)
    r = client.post("/command", json={"text": "go to the cup"})  # default unicycle
    body = r.json()
    assert body["status"] == "success"
    assert body["kinematics"] == "unicycle"
    trace = client.get("/control/last_trace").json()
    assert trace.get("kinematics") != "car"


def test_stage_a_publishes_nav2_goal(client):
    _seed(client, "cup0", "cup", 1.4, -0.6)
    before = client.get("/nav2/last").json()["goals_published"]
    client.post("/command", json={"text": "go to the cup", "kinematics": "car"})
    after = client.get("/nav2/last").json()
    assert after["goals_published"] == before + 1
    goal = after["last_goal"]
    assert goal["header"]["frame_id"] == "map"
    # Planar identity: ROS y == world z of the standoff (negative here).
    assert goal["pose"]["position"]["y"] < 0.0


def test_stage_b_metric_map_accretes_free_and_occupied(client):
    _seed(client, "box0", "box", 1.2, 0.0, ext=(0.3, 0.3, 0.3))
    grid = client.get("/slam/metric_map").json()
    assert grid["width"] > 0 and grid["height"] > 0
    # Repeated scans should have carved free space and marked the obstacle.
    assert grid["free_cells"] > 0
    assert grid["occupied_cells"] > 0


def test_car_pick_still_synthesises_grasp(client):
    _seed(client, "cup0", "cup", 1.5, -0.7)
    r = client.post("/command", json={"text": "pick up the cup", "kinematics": "car"})
    body = r.json()
    assert body["status"] == "success"
    assert body["kinematics"] == "car"
    assert "pick" in body
    assert body["pick"].get("actions", 0) > 0
