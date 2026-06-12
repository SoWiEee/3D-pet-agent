"""§14.6.4 — LLM-assisted grounding (llm_pick_target)."""

from __future__ import annotations

import json
from typing import Any

from src.planning.llm_grounding import LLMTargetPick, llm_pick_target
from src.spatial.semantic_map import SemanticMap
from tests.factories import make_object


class _FakePickClient:
    """Returns a fixed JSON pick from .chat, or raises."""

    def __init__(self, payload: dict[str, Any] | None, *, raises: bool = False) -> None:
        self._payload = payload
        self._raises = raises
        self.last_user: str | None = None

    def chat(self, **kwargs: Any) -> Any:
        if self._raises:
            raise RuntimeError("boom")
        self.last_user = kwargs["messages"][1]["content"]
        return {"message": {"content": json.dumps(self._payload)}}


def _two_box_map() -> SemanticMap:
    m = SemanticMap()
    m.update(
        [
            make_object(
                object_id="box_left",
                class_label="box",
                attributes=["red"],
                center_3d_world=(-0.8, 0.0, -1.5),
            ),
            make_object(
                object_id="box_right",
                class_label="box",
                attributes=["blue"],
                center_3d_world=(0.9, 0.0, -1.6),
            ),
        ],
        frame_id=0,
    )
    return m


_CANDIDATES = [("box_left", 0.61), ("box_right", 0.58)]


def test_pick_returns_chosen_candidate() -> None:
    client = _FakePickClient({"object_id": "box_left", "justification": "the red box"})
    out = llm_pick_target("go to the red box", None, _CANDIDATES, _two_box_map(), client=client)
    assert out == ("box_left", "the red box")


def test_pick_passes_candidate_context_to_model() -> None:
    client = _FakePickClient({"object_id": "box_right", "justification": "blue one"})
    llm_pick_target("the blue box", None, _CANDIDATES, _two_box_map(), client=client)
    assert client.last_user is not None
    assert "box_left" in client.last_user and "box_right" in client.last_user
    assert "red" in client.last_user and "blue" in client.last_user


def test_hallucinated_id_is_rejected() -> None:
    client = _FakePickClient({"object_id": "box_nope", "justification": "made up"})
    out = llm_pick_target("the box", None, _CANDIDATES, _two_box_map(), client=client)
    assert out is None


def test_client_failure_returns_none() -> None:
    client = _FakePickClient(None, raises=True)
    out = llm_pick_target("the box", None, _CANDIDATES, _two_box_map(), client=client)
    assert out is None


def test_schema_invalid_returns_none() -> None:
    client = _FakePickClient({"object_id": "box_left"})  # missing justification
    out = llm_pick_target("the box", None, _CANDIDATES, _two_box_map(), client=client)
    assert out is None


def test_empty_candidates_returns_none() -> None:
    client = _FakePickClient({"object_id": "x", "justification": "y"})
    assert llm_pick_target("the box", None, [], _two_box_map(), client=client) is None


def test_target_pick_schema_roundtrips() -> None:
    pick = LLMTargetPick(object_id="a", justification="b")
    assert pick.object_id == "a" and pick.justification == "b"
