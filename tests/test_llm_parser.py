"""Phase A2 — LLM command parser tests.

The tests mock the Anthropic client (via the ``client_factory`` injection
point on :class:`LLMCommandParser`) so they pass without an API key and
without ``anthropic`` being installed. The integration with the top-level
:func:`parse_command` is covered separately via monkeypatching.
"""

from __future__ import annotations

from typing import Any

from src.language.command_parser import parse_command
from src.language.llm_parser import LLMCommandParser
from src.language.schema import CommandIntent

# ── fake Anthropic response shape ──────────────────────────────────────────


class _FakeBlock:
    def __init__(self, type_: str, name: str | None = None, input_: Any = None) -> None:
        self.type = type_
        self.name = name
        self.input = input_


class _FakeResponse:
    def __init__(self, content: list[_FakeBlock]) -> None:
        self.content = content


class _FakeMessages:
    def __init__(self, response: _FakeResponse | Exception) -> None:
        self._response = response
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> _FakeResponse:
        self.calls.append(kwargs)
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


class _FakeClient:
    def __init__(self, response: _FakeResponse | Exception) -> None:
        self.messages = _FakeMessages(response)


def _ok_response(intent_kwargs: dict[str, Any]) -> _FakeResponse:
    return _FakeResponse([
        _FakeBlock("tool_use", name="emit_command_intent", input_=intent_kwargs),
    ])


# ── LLMCommandParser ───────────────────────────────────────────────────────


def test_llm_parser_returns_intent_on_valid_tool_use() -> None:
    client = _FakeClient(
        _ok_response({
            "raw_text": "ignored — will be overwritten",
            "intent_type": "move_to",
            "target": {"class_label": "cup", "attributes": ["red"], "object_id": None},
            "spatial_relation": None,
            "constraints": [],
            "fallback": "ask_clarification",
            "confidence": 0.9,
        })
    )
    parser = LLMCommandParser(client_factory=lambda: client)

    intent = parser.parse("walk over to the red cup")

    assert isinstance(intent, CommandIntent)
    assert intent.intent_type == "move_to"
    assert intent.target is not None
    assert intent.target.class_label == "cup"
    # raw_text must be forced to the original utterance, never trusted from LLM.
    assert intent.raw_text == "walk over to the red cup"


def test_llm_parser_falls_back_when_client_factory_raises() -> None:
    def boom() -> Any:
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    parser = LLMCommandParser(client_factory=boom)
    assert parser.parse("anything") is None


def test_llm_parser_falls_back_when_api_call_raises() -> None:
    parser = LLMCommandParser(
        client_factory=lambda: _FakeClient(RuntimeError("rate limited"))
    )
    assert parser.parse("go to the cup") is None


def test_llm_parser_falls_back_when_no_tool_use_in_response() -> None:
    empty = _FakeResponse([_FakeBlock("text", name=None, input_=None)])
    parser = LLMCommandParser(client_factory=lambda: _FakeClient(empty))
    assert parser.parse("go to the cup") is None


def test_llm_parser_falls_back_when_tool_args_fail_schema_validation() -> None:
    # Missing required intent_type field
    bad = _ok_response({"raw_text": "x", "target": {"class_label": "cup"}})
    parser = LLMCommandParser(client_factory=lambda: _FakeClient(bad))
    assert parser.parse("go to the cup") is None


def test_llm_parser_accepts_json_string_input_in_tool_use() -> None:
    # Some SDKs deliver tool input as a JSON-encoded string instead of dict.
    block = _FakeBlock(
        "tool_use",
        name="emit_command_intent",
        input_='{"raw_text": "x", "intent_type": "stop", "confidence": 0.5}',
    )
    parser = LLMCommandParser(
        client_factory=lambda: _FakeClient(_FakeResponse([block]))
    )
    intent = parser.parse("halt")
    assert intent is not None
    assert intent.intent_type == "stop"


def test_llm_parser_caches_client_across_calls() -> None:
    calls = 0

    def factory() -> _FakeClient:
        nonlocal calls
        calls += 1
        return _FakeClient(_ok_response({"raw_text": "x", "intent_type": "stop"}))

    parser = LLMCommandParser(client_factory=factory)
    parser.parse("a")
    parser.parse("b")
    parser.parse("c")
    assert calls == 1


# ── parse_command integration ──────────────────────────────────────────────


def test_parse_command_uses_rule_when_llm_disabled(monkeypatch) -> None:
    monkeypatch.delenv("PET_AGENT_LLM_PARSER", raising=False)
    intent = parse_command("go to the cup")
    assert intent is not None
    assert intent.intent_type == "move_to"


def test_parse_command_calls_llm_when_enabled(monkeypatch) -> None:
    monkeypatch.setenv("PET_AGENT_LLM_PARSER", "on")

    fake = _FakeClient(
        _ok_response({
            "raw_text": "",
            "intent_type": "hide",
            "target": {"class_label": "keyboard"},
            "confidence": 0.8,
        })
    )
    from src.language import command_parser as cp

    # Force a fresh cached parser using our fake client.
    cp._LLM_PARSER = LLMCommandParser(client_factory=lambda: fake)
    try:
        intent = parse_command("would you tuck yourself behind the keyboard please?")
    finally:
        cp._LLM_PARSER = None

    assert intent is not None
    assert intent.intent_type == "hide"
    assert intent.target is not None
    assert intent.target.class_label == "keyboard"


def test_parse_command_falls_back_to_rule_when_llm_returns_none(monkeypatch) -> None:
    monkeypatch.setenv("PET_AGENT_LLM_PARSER", "on")
    from src.language import command_parser as cp

    cp._LLM_PARSER = LLMCommandParser(
        client_factory=lambda: _FakeClient(RuntimeError("upstream down"))
    )
    try:
        intent = parse_command("go to the cup")
    finally:
        cp._LLM_PARSER = None

    assert intent is not None
    # Rule parser handled it.
    assert intent.intent_type == "move_to"
    assert intent.target is not None
    assert intent.target.class_label == "cup"
