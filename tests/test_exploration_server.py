"""Phase 9 — server-level exploration smoke tests."""

from __future__ import annotations

import math

from fastapi.testclient import TestClient

from src.runtime import websocket_server as srv


def test_coverage_endpoint_starts_fully_unknown(client: TestClient) -> None:
    body = client.get("/exploration/coverage").json()
    assert body["unobserved_ratio"] == 1.0


def test_observe_endpoint_reduces_unknown_area(client: TestClient) -> None:
    resp = client.post(
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


def test_step_returns_fully_explored_on_blank_map(client: TestClient) -> None:
    srv.coverage_grid.grid[:, :] = 5
    body = client.post("/exploration/step", json={}).json()
    assert body["status"] == "fully_explored"


def test_step_runs_planner_when_unknown_region_exists(client: TestClient) -> None:
    client.post(
        "/exploration/observe",
        json={
            "camera_xz": [0.0, 0.0],
            "heading_rad": 0.0,
            "fov_rad": math.radians(20.0),
            "range_m": 0.5,
        },
    )
    resp = client.post("/exploration/step", json={})
    body = resp.json()
    assert body["status"] in ("success", "plan_failed")
    assert "exploration_goal" in body


def test_reset_endpoint_clears_coverage(client: TestClient) -> None:
    client.post(
        "/exploration/observe",
        json={
            "camera_xz": [0.0, 0.0],
            "heading_rad": 0.0,
            "fov_rad": math.pi / 2,
            "range_m": 1.0,
        },
    )
    client.post("/exploration/reset")
    body = client.get("/exploration/coverage").json()
    assert body["unobserved_ratio"] == 1.0
