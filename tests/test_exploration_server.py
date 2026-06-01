"""Phase 9 — server-level exploration smoke tests."""

from __future__ import annotations

import math

from fastapi.testclient import TestClient

from src.runtime import websocket_server as srv


def _client() -> TestClient:
    srv.runtime.state.position.x = 0.0
    srv.runtime.state.position.y = 0.0
    srv.runtime.state.position.z = 0.0
    srv.semantic_map.reset()
    srv.tracker.reset()
    srv.coverage_grid.reset()
    srv._last_exploration_ids = set()
    srv._last_trace_summary = None
    return TestClient(srv.app)


def test_coverage_endpoint_starts_fully_unknown() -> None:
    c = _client()
    body = c.get("/exploration/coverage").json()
    assert body["unobserved_ratio"] == 1.0


def test_observe_endpoint_reduces_unknown_area() -> None:
    c = _client()
    resp = c.post(
        "/exploration/observe",
        json={
            "camera_xz": [0.0, 0.0],
            "heading_rad": 0.0,
            "fov_rad": math.pi / 2,
            "range_m": 1.0,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["new_cells"] > 0
    assert body["unobserved_ratio"] < 1.0


def test_step_returns_fully_explored_on_blank_map() -> None:
    c = _client()
    srv.coverage_grid.grid[:, :] = 5
    body = c.post("/exploration/step", json={}).json()
    assert body["status"] == "fully_explored"


def test_step_runs_planner_when_unknown_region_exists() -> None:
    c = _client()
    c.post(
        "/exploration/observe",
        json={
            "camera_xz": [0.0, 0.0],
            "heading_rad": 0.0,
            "fov_rad": math.radians(20.0),
            "range_m": 0.5,
        },
    )
    resp = c.post("/exploration/step", json={})
    body = resp.json()
    assert body["status"] in ("success", "plan_failed")
    assert "exploration_goal" in body


def test_reset_endpoint_clears_coverage() -> None:
    c = _client()
    c.post(
        "/exploration/observe",
        json={
            "camera_xz": [0.0, 0.0],
            "heading_rad": 0.0,
            "fov_rad": math.pi / 2,
            "range_m": 1.0,
        },
    )
    c.post("/exploration/reset")
    body = c.get("/exploration/coverage").json()
    assert body["unobserved_ratio"] == 1.0
