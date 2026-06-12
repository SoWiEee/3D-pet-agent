"""§14.6.4 Task 1 — local Ollama command-parser backend tests.

The hermetic tests inject a fake Ollama client via the ``client_factory``
seam on :class:`OllamaCommandParser`, so they run without the ``ollama``
package talking to a server and without a live model. They assert the
structured-output contract: a JSON ``CommandIntent`` string on the chat
response → a validated :class:`CommandIntent`, with ``raw_text`` forced to
the original utterance, and every failure path → ``None`` (rule fallback).

One opt-in ``@pytest.mark.live`` smoke test exercises the real
``qwen2.5-coder:7b`` end-to-end; it is deselected by the default suite
(``addopts = -m "not live"``) and self-skips when Ollama is unreachable.
"""

from __future__ import annotations

import json
import socket
from typing import Any
from urllib.parse import urlparse

import pytest

from src.language.llm_parser import (
    LLMCommandParser,
    OllamaCommandParser,
    make_llm_parser,
)
from src.language.schema import CommandIntent

# ── fake Ollama response shapes ────────────────────────────────────────────
# The real client returns an object where both ``resp["message"]["content"]``
# (mapping form) and ``resp.message.content`` (attribute form) yield the JSON
# string. We exercise BOTH shapes.


class _AttrMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _AttrResponse:
    """Attribute-style response: resp.message.content."""

    def __init__(self, content: str) -> None:
        self.message = _AttrMessage(content)


def _dict_response(content: str) -> dict[str, Any]:
    """Mapping-style response: resp["message"]["content"]."""
    return {"message": {"content": content}}


class _FakeChat:
    def __init__(self, response: Any | Exception) -> None:
        self._response = response
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


class _FakeClient:
    def __init__(self, response: Any | Exception) -> None:
        self.chat = _FakeChat(response)


def _intent_json(**overrides: Any) -> str:
    data: dict[str, Any] = {
        "raw_text": "ignored — will be overwritten",
        "intent_type": "move_to",
        "target": {"class_label": "cup", "attributes": ["red"], "object_id": None},
        "spatial_relation": None,
        "constraints": [],
        "fallback": "ask_clarification",
        "confidence": 0.9,
    }
    data.update(overrides)
    return json.dumps(data)


# ── OllamaCommandParser — happy path (both response shapes) ─────────────────


def test_ollama_parser_returns_intent_on_dict_response() -> None:
    client = _FakeClient(_dict_response(_intent_json()))
    parser = OllamaCommandParser(client_factory=lambda: client)

    intent = parser.parse("go to the red cup")

    assert isinstance(intent, CommandIntent)
    assert intent.intent_type == "move_to"
    assert intent.target is not None
    assert intent.target.class_label == "cup"
    # raw_text is forced to the original utterance, never trusted from the model.
    assert intent.raw_text == "go to the red cup"


def test_ollama_parser_returns_intent_on_attr_response() -> None:
    client = _FakeClient(_AttrResponse(_intent_json()))
    parser = OllamaCommandParser(client_factory=lambda: client)

    intent = parser.parse("go to the red cup")

    assert isinstance(intent, CommandIntent)
    assert intent.intent_type == "move_to"
    assert intent.raw_text == "go to the red cup"


def test_ollama_parser_passes_schema_and_options_to_chat() -> None:
    client = _FakeClient(_dict_response(_intent_json()))
    parser = OllamaCommandParser(model="qwen2.5-coder:7b", client_factory=lambda: client)

    parser.parse("go to the red cup")

    (kwargs,) = client.chat.calls
    assert kwargs["model"] == "qwen2.5-coder:7b"
    assert kwargs["format"] == CommandIntent.model_json_schema()
    assert kwargs["options"]["temperature"] == 0.0
    roles = [m["role"] for m in kwargs["messages"]]
    assert roles == ["system", "user"]
    assert kwargs["messages"][1]["content"] == "go to the red cup"


# ── OllamaCommandParser — failure paths all → None ─────────────────────────


def test_ollama_parser_returns_none_on_malformed_json() -> None:
    client = _FakeClient(_dict_response("this is not json {"))
    parser = OllamaCommandParser(client_factory=lambda: client)
    assert parser.parse("go to the cup") is None


def test_ollama_parser_returns_none_on_schema_invalid_content() -> None:
    # Bad intent_type — not in the IntentType literal.
    bad = _dict_response(_intent_json(intent_type="teleport"))
    parser = OllamaCommandParser(client_factory=lambda: _FakeClient(bad))
    assert parser.parse("go to the cup") is None


def test_ollama_parser_returns_none_when_client_factory_raises() -> None:
    def boom() -> Any:
        raise RuntimeError("ollama package not installed")

    parser = OllamaCommandParser(client_factory=boom)
    assert parser.parse("anything") is None


def test_ollama_parser_returns_none_when_chat_raises() -> None:
    parser = OllamaCommandParser(client_factory=lambda: _FakeClient(ConnectionError("host down")))
    assert parser.parse("go to the cup") is None


def test_ollama_parser_returns_none_on_empty_content() -> None:
    parser = OllamaCommandParser(client_factory=lambda: _FakeClient(_dict_response("")))
    assert parser.parse("go to the cup") is None


def test_ollama_parser_caches_client_across_calls() -> None:
    calls = 0

    def factory() -> _FakeClient:
        nonlocal calls
        calls += 1
        return _FakeClient(_dict_response(_intent_json(intent_type="stop", target=None)))

    parser = OllamaCommandParser(client_factory=factory)
    parser.parse("a")
    parser.parse("b")
    parser.parse("c")
    assert calls == 1


# ── make_llm_parser factory ────────────────────────────────────────────────


def test_make_llm_parser_returns_ollama_when_backend_ollama(monkeypatch) -> None:
    monkeypatch.setenv("PET_AGENT_LLM_BACKEND", "ollama")
    assert isinstance(make_llm_parser(), OllamaCommandParser)


def test_make_llm_parser_defaults_to_anthropic_when_unset(monkeypatch) -> None:
    monkeypatch.delenv("PET_AGENT_LLM_BACKEND", raising=False)
    assert isinstance(make_llm_parser(), LLMCommandParser)


def test_make_llm_parser_returns_anthropic_for_explicit_anthropic(monkeypatch) -> None:
    monkeypatch.setenv("PET_AGENT_LLM_BACKEND", "anthropic")
    assert isinstance(make_llm_parser(), LLMCommandParser)


def test_make_llm_parser_returns_anthropic_for_unknown_backend(monkeypatch) -> None:
    monkeypatch.setenv("PET_AGENT_LLM_BACKEND", "gpt5")
    assert isinstance(make_llm_parser(), LLMCommandParser)


# ── opt-in live smoke test (real qwen2.5-coder:7b) ─────────────────────────


def _ollama_reachable(host: str) -> bool:
    parsed = urlparse(host)
    hostname = parsed.hostname or "localhost"
    port = parsed.port or 11434
    try:
        with socket.create_connection((hostname, port), timeout=1.0):
            return True
    except OSError:
        return False


def _model_serves(host: str, model: str) -> bool:
    """True only if the model actually loads + generates. A busy GPU
    (``cudaMalloc failed: device busy``) or a missing model returns False so the
    live test SKIPS on infrastructure trouble rather than reporting a spurious
    parser failure."""
    try:
        import ollama  # type: ignore[import-not-found]

        ollama.Client(host=host).chat(
            model=model,
            messages=[{"role": "user", "content": "ok"}],
            options={"num_predict": 1},
        )
        return True
    except Exception:  # noqa: BLE001 — any infra error → skip
        return False


@pytest.mark.live
def test_ollama_parser_live_end_to_end() -> None:
    pytest.importorskip("ollama")
    host = "http://localhost:11434"
    if not _ollama_reachable(host):
        pytest.skip("Ollama not reachable on localhost:11434")
    if not _model_serves(host, "qwen2.5-coder:7b"):
        pytest.skip("Ollama model could not load (GPU busy / model missing)")

    parser = OllamaCommandParser(model="qwen2.5-coder:7b", host=host)
    intent = parser.parse("go to the red cup")

    # A 7B local model can miss edge phrasings; we only require a valid intent
    # for a clear, canonical utterance here. None would indicate a hard failure.
    assert intent is not None, "live Ollama returned no valid CommandIntent"
    assert isinstance(intent, CommandIntent)
    assert intent.raw_text == "go to the red cup"
