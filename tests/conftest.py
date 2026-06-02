"""Shared pytest fixtures.

Importable builders (e.g. ``make_object``) live in ``tests/factories.py``;
this module holds only fixtures. See that module for the rationale.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client() -> TestClient:
    """A FastAPI TestClient over the live app with all server-held singletons
    reset to a clean slate. Resets the superset of state the server tests need
    (pose, semantic map, tracker, coverage grid, exploration + control caches),
    so it's safe for every endpoint test regardless of which it exercises.
    """
    from src.runtime import websocket_server as srv

    srv.runtime.state.position.x = 0.0
    srv.runtime.state.position.y = 0.0
    srv.runtime.state.position.z = 0.0
    srv.semantic_map.reset()
    srv.tracker.reset()
    srv.coverage_grid.reset()
    srv._last_exploration_ids = set()
    srv._last_trace_summary = None
    return TestClient(srv.app)
