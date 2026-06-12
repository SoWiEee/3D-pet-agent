"""§14.6.4 — /command wiring for LLM-assisted grounding.

Hermetic: the model pick is injected by monkeypatching ``llm_pick_target`` in
the server module, so no live Ollama is needed.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


def _box(oid: str, x: float, attrs: list[str], bbox: list[float]) -> dict:
    return {
        "object_id": oid,
        "class_label": "box",
        "attributes": attrs,
        "bbox_xyxy": bbox,
        "center_2d": [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2],
        "coordinate_frame": "world",
        "center_3d_world": [x, 0.1, -2.0],
        "extent_3d": [0.1, 0.1, 0.1],
        "median_depth": 2.0,
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


def _seed_two_boxes(client: TestClient) -> None:
    left = _box("boxL", -1.5, ["red"], [50, 100, 150, 200])
    right = _box("boxR", 1.5, ["blue"], [400, 100, 500, 200])
    for f in range(4):
        client.post("/perception/lifted", json={"objects": [left, right], "frame_id": f})


@pytest.fixture
def srv_client():
    from src.runtime import websocket_server as srv

    with TestClient(srv.app) as c:
        c.post("/semantic/reset")
        srv.dialogue_store.resolve("g1")
        yield c, srv
        c.post("/semantic/reset")
        srv.dialogue_store.resolve("g1")


def test_llm_grounding_shortcircuits_clarification(srv_client, monkeypatch) -> None:
    client, srv = srv_client
    _seed_two_boxes(client)
    red_id = next(o.object_id for o in srv.semantic_map.values() if "red" in o.attributes)

    monkeypatch.setenv("PET_AGENT_LLM_GROUNDING", "on")
    monkeypatch.setattr(
        srv, "llm_pick_target", lambda *a, **k: (red_id, "the red box you described")
    )

    body = client.post("/command", json={"text": "go to the box", "session_id": "g1"}).json()
    assert body["status"] == "success"
    assert body["goal"]["target_object_id"] == red_id
    assert body["goal"]["explanation"] == "the red box you described"
    # No pending dialogue — the confident pick short-circuited the question.
    assert srv.dialogue_store.get("g1") is None


def test_llm_grounding_off_keeps_clarification(srv_client, monkeypatch) -> None:
    client, srv = srv_client
    _seed_two_boxes(client)
    # Gate off (default) → heuristic clarification stands even if a picker exists.
    monkeypatch.delenv("PET_AGENT_LLM_GROUNDING", raising=False)
    monkeypatch.setattr(srv, "llm_pick_target", lambda *a, **k: ("boxL", "should be ignored"))

    body = client.post("/command", json={"text": "go to the box", "session_id": "g1"}).json()
    assert body["status"] == "clarification"


def test_llm_grounding_none_pick_keeps_clarification(srv_client, monkeypatch) -> None:
    client, srv = srv_client
    _seed_two_boxes(client)
    monkeypatch.setenv("PET_AGENT_LLM_GROUNDING", "on")
    # Model abstains (None) → heuristic clarification stands.
    monkeypatch.setattr(srv, "llm_pick_target", lambda *a, **k: None)

    body = client.post("/command", json={"text": "go to the box", "session_id": "g1"}).json()
    assert body["status"] == "clarification"
