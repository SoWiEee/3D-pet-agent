"""Shared local-Ollama call helpers (spec §14.6.4).

Small, defensive wrappers around the official ``ollama`` Python client used by
the LLM-assisted grounding (``planning/llm_grounding.py``) and the generative
discriminating-question path (``dialogue.py``). Every function swallows all
errors and returns ``None`` so the caller always has a deterministic fallback —
a missing package, an unreachable host, a timeout, or malformed output must
never break ``/command``.

Task 1's :class:`~src.language.llm_parser.OllamaCommandParser` keeps its own
client bootstrap; this module is for the new call sites. The slight duplication
of ``ollama.Client(host=...)`` is intentional — it isolates the parser from
these helpers.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any

from .llm_parser import _extract_content

log = logging.getLogger("pet_agent.ollama_client")

DEFAULT_TIMEOUT_S = 6.0


def get_client(host: str, *, client_factory: Callable[[], Any] | None = None) -> Any | None:
    """Build an Ollama client for ``host``. Returns ``None`` on any failure
    (package missing, bad host) so callers fall back deterministically."""
    try:
        if client_factory is not None:
            return client_factory()
        import ollama  # type: ignore[import-not-found]

        return ollama.Client(host=host)
    except Exception as e:  # noqa: BLE001 — never let client bootstrap escape
        log.warning("Ollama client unavailable (%s); caller will fall back", e)
        return None


def chat_json(
    client: Any,
    *,
    model: str,
    system: str,
    user: str,
    schema: dict[str, Any],
    timeout_s: float = DEFAULT_TIMEOUT_S,  # noqa: ARG001 — client owns its timeout
) -> dict[str, Any] | None:
    """Structured chat: returns a JSON object validated by ``schema`` as a dict,
    or ``None`` on any failure. Never raises."""
    try:
        response = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            format=schema,
            options={"temperature": 0.0},
        )
    except Exception as e:  # noqa: BLE001
        log.warning("Ollama chat_json call failed (%s)", e)
        return None
    content = _extract_content(response)
    if not content:
        return None
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return None
    return data if isinstance(data, dict) else None


def chat_text(
    client: Any,
    *,
    model: str,
    system: str,
    user: str,
    timeout_s: float = DEFAULT_TIMEOUT_S,  # noqa: ARG001 — client owns its timeout
) -> str | None:
    """Freeform chat: returns the trimmed assistant text, or ``None`` on any
    failure. Never raises."""
    try:
        response = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            options={"temperature": 0.2},
        )
    except Exception as e:  # noqa: BLE001
        log.warning("Ollama chat_text call failed (%s)", e)
        return None
    content = _extract_content(response)
    if not isinstance(content, str):
        return None
    text = content.strip()
    return text or None
