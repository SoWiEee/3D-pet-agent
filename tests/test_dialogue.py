"""Task 2 — multi-turn clarification: per-session dialogue state.

Covers the pure dialogue module (DialogueStore, discriminating_question,
merge_followup) and the `/command` wiring (session_id → discriminating
question → folded follow-up → re-grounding) via the FastAPI TestClient.

All tests are hermetic: the LLM path is never exercised (templated fallback
only), so no live model is required.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.language.dialogue import (
    DialogueStore,
    PendingClarification,
    discriminating_question,
    merge_followup,
)
from src.language.schema import CommandIntent, TargetSpec
from src.spatial.semantic_map import SemanticMap
from tests.factories import make_object

# ── DialogueStore ───────────────────────────────────────────────────────────


def _intent(text: str = "go to the box") -> CommandIntent:
    return CommandIntent(
        raw_text=text,
        intent_type="move_to",
        target=TargetSpec(class_label="box", attributes=[]),
    )


def test_open_and_get_returns_pending_turn() -> None:
    store = DialogueStore()
    intent = _intent()
    candidates = [("box_001", 0.61), ("box_002", 0.55)]
    store.open_clarification("s1", intent, candidates, "Which box?")

    pending = store.get("s1")
    assert isinstance(pending, PendingClarification)
    assert pending.intent is intent
    assert pending.candidates == candidates
    assert pending.question == "Which box?"
    assert pending.created_at > 0


def test_resolve_clears_pending_turn() -> None:
    store = DialogueStore()
    store.open_clarification("s1", _intent(), [("box_001", 0.6)], "Which box?")
    store.resolve("s1")
    assert store.get("s1") is None


def test_get_unknown_session_returns_none() -> None:
    store = DialogueStore()
    assert store.get("does-not-exist") is None


def test_resolve_unknown_session_is_noop() -> None:
    store = DialogueStore()
    store.resolve("nope")  # must not raise
    assert store.get("nope") is None


def test_bounded_store_evicts_oldest() -> None:
    cap = 8
    store = DialogueStore(capacity=cap)
    for i in range(cap + 5):
        store.open_clarification(f"s{i}", _intent(), [("box_001", 0.6)], "Q?")
        assert len(store) <= cap

    assert len(store) == cap
    # The first 5 sessions were evicted (oldest-first).
    for i in range(5):
        assert store.get(f"s{i}") is None
    # The most recent `cap` survive.
    for i in range(5, cap + 5):
        assert store.get(f"s{i}") is not None


def test_default_capacity_is_64() -> None:
    store = DialogueStore()
    for i in range(70):
        store.open_clarification(f"s{i}", _intent(), [("box_001", 0.6)], "Q?")
    assert len(store) == 64


# ── discriminating_question (templated fallback) ────────────────────────────


def test_question_distinguishes_by_attribute() -> None:
    m = SemanticMap()
    m.update(
        [
            make_object(
                object_id="box_001",
                class_label="box",
                attributes=["red"],
                center_3d_world=(-1.0, 0.0, -2.0),
            ),
            make_object(
                object_id="box_002",
                class_label="box",
                attributes=["blue"],
                center_3d_world=(1.0, 0.0, -2.0),
            ),
        ],
        frame_id=0,
    )
    q = discriminating_question([("box_001", 0.6), ("box_002", 0.55)], m)
    assert isinstance(q, str)
    assert "box" in q.lower()
    assert "red" in q.lower()
    assert "blue" in q.lower()
    assert q.strip().endswith("?")


def test_question_falls_back_to_position_when_attributes_match() -> None:
    m = SemanticMap()
    m.update(
        [
            make_object(
                object_id="box_001",
                class_label="box",
                attributes=[],
                center_3d_world=(-1.5, 0.0, -2.0),
            ),
            make_object(
                object_id="box_002",
                class_label="box",
                attributes=[],
                center_3d_world=(1.5, 0.0, -2.0),
            ),
        ],
        frame_id=0,
    )
    q = discriminating_question([("box_001", 0.6), ("box_002", 0.55)], m).lower()
    # No attributes differ → use a coarse left/right hint from center_3d_world.
    assert "left" in q
    assert "right" in q


def test_question_never_raises_on_missing_object() -> None:
    m = SemanticMap()  # empty — candidate ids resolve to None
    q = discriminating_question([("ghost_001", 0.6), ("ghost_002", 0.5)], m)
    assert isinstance(q, str)
    assert q  # non-empty fallback


def _two_box_map() -> SemanticMap:
    m = SemanticMap()
    m.update(
        [
            make_object(
                object_id="box_001",
                class_label="box",
                attributes=["red"],
                center_3d_world=(-1.0, 0.0, -2.0),
            ),
            make_object(
                object_id="box_002",
                class_label="box",
                attributes=["blue"],
                center_3d_world=(1.0, 0.0, -2.0),
            ),
        ],
        frame_id=0,
    )
    return m


class _FakeGen:
    def __init__(self, content: str | None, *, raises: bool = False) -> None:
        self._content = content
        self._raises = raises

    def chat(self, **_kwargs: object) -> object:
        if self._raises:
            raise RuntimeError("ollama down")
        return {"message": {"content": self._content}}


def test_question_uses_generative_client_when_provided() -> None:
    gen = _FakeGen("Did you mean the red box or the blue box?")
    q = discriminating_question(
        [("box_001", 0.6), ("box_002", 0.55)], _two_box_map(), gen_client=gen, model="m"
    )
    assert q == "Did you mean the red box or the blue box?"


def test_question_falls_back_to_template_when_generative_fails() -> None:
    gen = _FakeGen(None, raises=True)
    q = discriminating_question(
        [("box_001", 0.6), ("box_002", 0.55)], _two_box_map(), gen_client=gen, model="m"
    )
    # Deterministic template still answers (names the distinguishing attributes).
    assert "red" in q.lower() and "blue" in q.lower()


# ── merge_followup ──────────────────────────────────────────────────────────


def test_merge_followup_folds_attribute_keeps_intent_type() -> None:
    prior = CommandIntent(
        raw_text="go to the box",
        intent_type="move_to",
        target=TargetSpec(class_label="box", attributes=[]),
    )
    merged = merge_followup(prior, "the red one")

    assert merged.intent_type == "move_to"
    assert merged.target is not None
    assert "red" in merged.target.attributes
    assert merged.target.class_label == "box"
    assert merged.raw_text == "the red one"


def test_merge_followup_is_immutable() -> None:
    prior = CommandIntent(
        raw_text="go to the box",
        intent_type="move_to",
        target=TargetSpec(class_label="box", attributes=[]),
    )
    merged = merge_followup(prior, "the left one")
    # Prior intent untouched.
    assert prior.target is not None
    assert prior.target.attributes == []
    assert prior.raw_text == "go to the box"
    assert merged is not prior


def test_merge_followup_folds_lexical_left() -> None:
    prior = CommandIntent(
        raw_text="go to the box",
        intent_type="move_to",
        target=TargetSpec(class_label="box", attributes=[]),
    )
    merged = merge_followup(prior, "left")
    assert merged.target is not None
    assert "left" in merged.target.attributes


def test_merge_followup_no_discriminator_returns_prior_unchanged() -> None:
    prior = CommandIntent(
        raw_text="go to the box",
        intent_type="move_to",
        target=TargetSpec(class_label="box", attributes=[]),
    )
    merged = merge_followup(prior, "uhh")
    assert merged is prior


# ── /command wiring ─────────────────────────────────────────────────────────


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
    # Both boxes in EVERY frame, well separated in image + world space so the
    # tracker keeps two distinct tracks (not one collapsed instance).
    left = _box("boxL", -1.5, ["red"], [50, 100, 150, 200])
    right = _box("boxR", 1.5, ["blue"], [400, 100, 500, 200])
    for f in range(4):
        client.post("/perception/lifted", json={"objects": [left, right], "frame_id": f})


@pytest.fixture
def dlg_client():
    from src.runtime import websocket_server as srv

    with TestClient(srv.app) as c:
        c.post("/semantic/reset")
        srv.dialogue_store.resolve("sess-1")
        yield c
        c.post("/semantic/reset")
        srv.dialogue_store.resolve("sess-1")


def test_command_without_session_keeps_canned_clarification(dlg_client) -> None:
    _seed_two_boxes(dlg_client)
    r = dlg_client.post("/command", json={"text": "go to the box"})
    body = r.json()
    assert body["status"] == "clarification"
    # No session_id → today's behaviour: no `question` key, canned explanation.
    assert "question" not in body


def test_command_with_session_asks_discriminating_question(dlg_client) -> None:
    _seed_two_boxes(dlg_client)
    r = dlg_client.post("/command", json={"text": "go to the box", "session_id": "sess-1"})
    body = r.json()
    assert body["status"] == "clarification"
    assert "question" in body
    q = body["question"].lower()
    assert "red" in q and "blue" in q


def test_followup_reply_resolves_to_target(dlg_client) -> None:
    _seed_two_boxes(dlg_client)
    from src.runtime import websocket_server as srv

    # Tracking rewrites object_ids → resolve the red box's live track id.
    red_id = next(o.object_id for o in srv.semantic_map.values() if "red" in o.attributes)

    dlg_client.post("/command", json={"text": "go to the box", "session_id": "sess-1"})
    # Follow-up with the same session_id, carrying just the discriminator.
    r = dlg_client.post("/command", json={"text": "the red one", "session_id": "sess-1"})
    body = r.json()
    assert body["status"] == "success"
    assert body["goal"]["target_object_id"] == red_id

    # The dialogue state is cleared after a successful resolution.
    assert srv.dialogue_store.get("sess-1") is None


def test_followup_retry_cap_does_not_loop(dlg_client) -> None:
    _seed_two_boxes(dlg_client)
    dlg_client.post("/command", json={"text": "go to the box", "session_id": "sess-1"})
    from src.runtime import websocket_server as srv

    # Replies that carry no usable discriminator keep grounding ambiguous.
    last = None
    for _ in range(5):
        last = dlg_client.post("/command", json={"text": "hmm", "session_id": "sess-1"}).json()
    # After the retry cap, the dialogue must be resolved (no infinite loop).
    assert srv.dialogue_store.get("sess-1") is None
    assert last is not None
