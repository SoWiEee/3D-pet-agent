"""Phase 8 — controller wiring + server integration."""

from __future__ import annotations

import math

from fastapi.testclient import TestClient

from src.control.kinematic import UnicycleState
from src.runtime import websocket_server as srv


def test_simulate_endpoint_returns_summary_and_dense_path(client: TestClient) -> None:
    resp = client.post(
        "/control/simulate",
        json={
            "path": [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            "start": [0.0, 0.0, 0.0],
            "start_theta": math.pi / 2,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["summary"]["status"] == "success"
    assert len(body["path_world"]) > 2  # densified beyond the 2 raw waypoints


def test_simulate_with_empty_path_handled_gracefully(client: TestClient) -> None:
    resp = client.post("/control/simulate", json={"path": []})
    assert resp.status_code == 200
    assert resp.json()["status"] == "empty_path"


def test_perception_status_endpoint_when_not_running(client: TestClient) -> None:
    body = client.get("/perception/status").json()
    assert body["available"] is True
    assert body["running"] is False


def test_perception_stop_when_not_running(client: TestClient) -> None:
    body = client.post("/perception/stop").json()
    assert body["stopped"] is False


def test_last_trace_endpoint_starts_empty(client: TestClient) -> None:
    body = client.get("/control/last_trace").json()
    assert body == {}


def _lift_one_cup(c: TestClient, x: float = 0.6, z: float = 0.8) -> None:
    c.post(
        "/perception/lifted",
        json={
            "objects": [
                {
                    "object_id": "cup_x",
                    "class_label": "cup",
                    "bbox_xyxy": [0, 0, 10, 10],
                    "center_2d": [5, 5],
                    "center_3d_world": [x, 0.0, z],
                    "extent_3d": [0.1, 0.12, 0.1],
                    "median_depth": 1.0,
                    "depth_uncertainty": 0.05,
                    "source_backend": "mainline_grounding_sam",
                    "confidence": {"detector": 0.9, "overall": 0.85},
                    "last_seen_frame": 1,
                    "tracking_status": "tracked",
                }
            ]
        },
    )


def test_command_success_returns_reasoning_fields(client: TestClient) -> None:
    _lift_one_cup(client)
    body = client.post("/command", json={"text": "go to the cup"}).json()
    assert body["parsed"] is True
    assert body["status"] == "success"
    # Reasoning panel payload.
    assert "weights" in body and set(body["weights"]) >= {"semantic", "relation"}
    assert "explanation" in body and body["explanation"]
    assert isinstance(body["candidate_breakdowns"], list) and body["candidate_breakdowns"]
    assert body["candidate_breakdowns"][0]["object_id"] == "track_001"
    assert "goal_score" in body


def test_initial_state_derives_heading_from_path() -> None:
    srv.runtime.state.position.x = 0.0
    srv.runtime.state.position.z = 0.0
    initial = srv._initial_state_from_plan([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)])
    assert isinstance(initial, UnicycleState)
    assert initial.theta == 0.0
    initial_z = srv._initial_state_from_plan([(0.0, 0.0, 0.0), (0.0, 0.0, 1.0)])
    assert initial_z.theta == math.pi / 2


def test_initial_state_skips_collocated_first_segment() -> None:
    srv.runtime.state.position.x = 0.0
    srv.runtime.state.position.z = 0.0
    initial = srv._initial_state_from_plan([(0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)])
    assert initial.theta == 0.0
