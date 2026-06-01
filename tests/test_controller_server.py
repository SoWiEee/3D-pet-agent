"""Phase 8 — controller wiring + server integration."""

from __future__ import annotations

import math

from fastapi.testclient import TestClient

from src.control.kinematic import UnicycleState
from src.runtime import websocket_server as srv


def _client() -> TestClient:
    srv.runtime.state.position.x = 0.0
    srv.runtime.state.position.y = 0.0
    srv.runtime.state.position.z = 0.0
    srv.semantic_map.reset()
    srv.tracker.reset()
    srv._last_trace_summary = None
    return TestClient(srv.app)


def test_simulate_endpoint_returns_summary_and_dense_path() -> None:
    c = _client()
    resp = c.post(
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


def test_simulate_with_empty_path_handled_gracefully() -> None:
    c = _client()
    resp = c.post("/control/simulate", json={"path": []})
    assert resp.status_code == 200
    assert resp.json()["status"] == "empty_path"


def test_last_trace_endpoint_starts_empty() -> None:
    c = _client()
    body = c.get("/control/last_trace").json()
    assert body == {}


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
