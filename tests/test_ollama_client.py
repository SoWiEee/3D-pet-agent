"""§14.6.4 — shared Ollama call helpers (chat_json / chat_text / get_client)."""

from __future__ import annotations

from typing import Any

from src.language.ollama_client import chat_json, chat_text, get_client

_SCHEMA = {"type": "object", "properties": {"object_id": {"type": "string"}}}


class _FakeChatClient:
    """Mimics ollama.Client.chat returning a response with .message.content."""

    def __init__(self, content: str | None, *, raises: bool = False) -> None:
        self._content = content
        self._raises = raises

    def chat(self, **_kwargs: Any) -> Any:
        if self._raises:
            raise RuntimeError("boom")
        return {"message": {"content": self._content}}


def test_get_client_uses_injected_factory() -> None:
    sentinel = object()
    assert get_client("http://x", client_factory=lambda: sentinel) is sentinel


def test_get_client_returns_none_when_factory_raises() -> None:
    def boom() -> Any:
        raise RuntimeError("no ollama")

    assert get_client("http://x", client_factory=boom) is None


def test_chat_json_happy_path() -> None:
    client = _FakeChatClient('{"object_id": "track_001", "justification": "the red one"}')
    out = chat_json(client, model="m", system="s", user="u", schema=_SCHEMA)
    assert out == {"object_id": "track_001", "justification": "the red one"}


def test_chat_json_malformed_returns_none() -> None:
    client = _FakeChatClient("not json {")
    assert chat_json(client, model="m", system="s", user="u", schema=_SCHEMA) is None


def test_chat_json_non_object_returns_none() -> None:
    client = _FakeChatClient("[1, 2, 3]")
    assert chat_json(client, model="m", system="s", user="u", schema=_SCHEMA) is None


def test_chat_json_call_raises_returns_none() -> None:
    client = _FakeChatClient(None, raises=True)
    assert chat_json(client, model="m", system="s", user="u", schema=_SCHEMA) is None


def test_chat_text_happy_path() -> None:
    client = _FakeChatClient("  Which box did you mean?  ")
    assert chat_text(client, model="m", system="s", user="u") == "Which box did you mean?"


def test_chat_text_empty_returns_none() -> None:
    client = _FakeChatClient("   ")
    assert chat_text(client, model="m", system="s", user="u") is None


def test_chat_text_call_raises_returns_none() -> None:
    client = _FakeChatClient(None, raises=True)
    assert chat_text(client, model="m", system="s", user="u") is None
