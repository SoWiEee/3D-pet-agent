"""Phase A2 — LLM command parser adapter.

docs/review.md A2: ``PET_AGENT_LLM_PARSER=on`` env var was reserved but
never wired. The rule parser handles ~10 canonical shapes; free-form
phrasing ("可以走去左邊那個盒子那邊嗎") falls back to ``unparseable``.

This module wires a real LLM that emits a JSON object validating against
:class:`CommandIntent`. The adapter is deliberately defensive:

- The ``anthropic`` SDK is imported **lazily** inside the parser, so
  installations without it work fine (parse just falls back to rules).
- Missing ``ANTHROPIC_API_KEY`` → return ``None`` immediately.
- API call wrapped in a wall-clock timeout (no hung demos).
- All exceptions → log + return ``None``. The rule parser then runs.
- The LLM response is validated through Pydantic before being returned;
  schema failures → return ``None``.

A ``client_factory`` injection point lets tests substitute a fake client
without monkey-patching ``anthropic`` module internals.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Callable
from typing import Any

from .schema import CommandIntent

log = logging.getLogger("pet_agent.llm_parser")

DEFAULT_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_TIMEOUT_S = 6.0
DEFAULT_MAX_TOKENS = 512

# Tool name the LLM is forced to call. We emit exactly one tool that
# accepts a CommandIntent so the response is structured by construction.
_TOOL_NAME = "emit_command_intent"

SYSTEM_PROMPT = (
    "You are a strict command parser for a 3D pet agent. The user speaks "
    "(in English or Traditional Chinese) about what the pet should do. "
    "Convert each utterance into a single structured CommandIntent by "
    "calling the emit_command_intent tool exactly once. Do not produce "
    "free-form text. If the utterance is gibberish or not actionable, "
    "still call the tool with intent_type='stop' and confidence=0.0."
)

# JSON-output variant for backends with native structured output (Ollama):
# there is no tool to call, so the model emits the CommandIntent JSON directly.
SYSTEM_PROMPT_JSON = (
    "You are a strict command parser for a 3D pet agent. The user speaks "
    "(in English or Traditional Chinese) about what the pet should do. "
    "Convert each utterance into a single CommandIntent JSON object that "
    "conforms to the provided schema. Respond with the JSON object only, no "
    "prose. If the utterance is gibberish or not actionable, set "
    "intent_type='stop' and confidence=0.0."
)

# Ollama backend defaults; overridable via PET_AGENT_OLLAMA_MODEL / _HOST.
DEFAULT_OLLAMA_MODEL = "qwen2.5-coder:7b"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"


def _command_intent_tool_schema() -> dict[str, Any]:
    """Build an Anthropic tool definition from the Pydantic CommandIntent
    schema so the LLM is forced to output a validating JSON object."""
    schema = CommandIntent.model_json_schema()
    return {
        "name": _TOOL_NAME,
        "description": (
            "Emit the structured CommandIntent for the user utterance. "
            "Always call this exactly once."
        ),
        "input_schema": schema,
    }


class LLMCommandParser:
    """LLM-backed parser. Stateless; safe to share across calls."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        client_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.model = model
        self.timeout_s = timeout_s
        self.max_tokens = max_tokens
        self._client_factory = client_factory or self._default_client_factory
        self._client: Any | None = None

    # ── client bootstrap ──────────────────────────────────────────────────
    @staticmethod
    def _default_client_factory() -> Any:
        """Lazily import anthropic. Raises ImportError if the SDK isn't
        installed — the caller catches it and falls back to rules."""
        import anthropic  # type: ignore[import-not-found]

        api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get(
            "PET_AGENT_ANTHROPIC_API_KEY"
        )
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        return anthropic.Anthropic(api_key=api_key, timeout=DEFAULT_TIMEOUT_S)

    def _client_or_none(self) -> Any | None:
        if self._client is not None:
            return self._client
        try:
            self._client = self._client_factory()
        except Exception as e:  # noqa: BLE001
            log.warning("LLM client unavailable (%s); rule parser will run", e)
            return None
        return self._client

    # ── public API ────────────────────────────────────────────────────────
    def parse(self, text: str) -> CommandIntent | None:
        client = self._client_or_none()
        if client is None:
            return None
        t0 = time.perf_counter()
        try:
            response = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=SYSTEM_PROMPT,
                tools=[_command_intent_tool_schema()],
                tool_choice={"type": "tool", "name": _TOOL_NAME},
                messages=[{"role": "user", "content": text}],
            )
        except Exception as e:  # noqa: BLE001
            log.warning("LLM call failed (%s); falling back to rules", e)
            return None
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        log.info("LLM parser: %.0fms model=%s", elapsed_ms, self.model)

        raw_args = _extract_tool_args(response)
        if raw_args is None:
            log.warning("LLM response missing tool_use; falling back to rules")
            return None
        # Force the original text in case the LLM rewrote it.
        raw_args["raw_text"] = text
        try:
            return CommandIntent(**raw_args)
        except Exception as e:  # noqa: BLE001 — schema mismatch is silent fallback
            log.warning("LLM output failed schema validation (%s); falling back", e)
            return None


def _extract_tool_args(response: Any) -> dict[str, Any] | None:
    """Pull the first tool_use block out of an Anthropic Messages response.

    Works against the official SDK response shape (response.content is a
    list of content blocks; each block has a ``type``). Also accepts a
    fake response that exposes the same shape — used by tests.
    """
    content = getattr(response, "content", None)
    if content is None and isinstance(response, dict):
        content = response.get("content")
    if not content:
        return None
    for block in content:
        btype = getattr(block, "type", None) or (
            block.get("type") if isinstance(block, dict) else None
        )
        if btype != "tool_use":
            continue
        name = getattr(block, "name", None) or (
            block.get("name") if isinstance(block, dict) else None
        )
        if name != _TOOL_NAME:
            continue
        args = getattr(block, "input", None)
        if args is None and isinstance(block, dict):
            args = block.get("input")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                return None
        if isinstance(args, dict):
            return args
    return None


# ── Ollama backend (local structured-output model) ─────────────────────────


def _settings_ollama_defaults() -> tuple[str, str]:
    """Read the Ollama model/host from pydantic ``Settings`` (env-backed).

    Defensive: if config import or instantiation fails for any reason, fall
    back to the module-level defaults so the parser is always constructible.
    """
    try:
        from ..config import Settings

        s = Settings()
        return s.ollama_model, s.ollama_host
    except Exception as e:  # noqa: BLE001 — config must never block construction
        log.warning("Ollama settings load failed (%s); using defaults", e)
        return DEFAULT_OLLAMA_MODEL, DEFAULT_OLLAMA_HOST


class OllamaCommandParser:
    """Local Ollama-backed parser using structured output (``format=schema``).

    Mirrors :class:`LLMCommandParser`'s contract exactly: stateless, lazy SDK
    import, ``client_factory`` injection for tests, ``parse → CommandIntent |
    None`` with every exception swallowed so the rule parser can take over.
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        host: str | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        client_factory: Callable[[], Any] | None = None,
    ) -> None:
        if model is None or host is None:
            cfg_model, cfg_host = _settings_ollama_defaults()
            model = model or cfg_model
            host = host or cfg_host
        self.model = model
        self.host = host
        self.timeout_s = timeout_s
        self._client_factory = client_factory or self._default_client_factory
        self._client: Any | None = None

    # ── client bootstrap ──────────────────────────────────────────────────
    def _default_client_factory(self) -> Any:
        """Lazily import ollama. Raises ImportError if the package isn't
        installed — the caller catches it and falls back to rules."""
        import ollama  # type: ignore[import-not-found]

        return ollama.Client(host=self.host)

    def _client_or_none(self) -> Any | None:
        if self._client is not None:
            return self._client
        try:
            self._client = self._client_factory()
        except Exception as e:  # noqa: BLE001
            log.warning("Ollama client unavailable (%s); rule parser will run", e)
            return None
        return self._client

    # ── public API ────────────────────────────────────────────────────────
    def parse(self, text: str) -> CommandIntent | None:
        client = self._client_or_none()
        if client is None:
            return None
        t0 = time.perf_counter()
        try:
            response = client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_JSON},
                    {"role": "user", "content": text},
                ],
                format=CommandIntent.model_json_schema(),
                options={"temperature": 0.0},
            )
        except Exception as e:  # noqa: BLE001
            log.warning("Ollama call failed (%s); falling back to rules", e)
            return None
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        log.info("Ollama parser: %.0fms model=%s", elapsed_ms, self.model)

        content = _extract_content(response)
        if not content:
            log.warning("Ollama response had no content; falling back to rules")
            return None
        try:
            data = json.loads(content)
        except (json.JSONDecodeError, TypeError) as e:
            log.warning("Ollama output was not valid JSON (%s); falling back", e)
            return None
        if not isinstance(data, dict):
            log.warning("Ollama output was not a JSON object; falling back")
            return None
        # Force the original text in case the model rewrote or dropped it.
        data["raw_text"] = text
        try:
            return CommandIntent(**data)
        except Exception as e:  # noqa: BLE001 — schema mismatch is silent fallback
            log.warning("Ollama output failed schema validation (%s); falling back", e)
            return None


def _extract_content(response: Any) -> str | None:
    """Pull the message content string out of an Ollama chat response.

    Handles both the mapping form (``resp["message"]["content"]``) and the
    attribute form (``resp.message.content``) the official client exposes.
    """
    message = getattr(response, "message", None)
    if message is None and isinstance(response, dict):
        message = response.get("message")
    if message is None:
        return None
    content = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        content = message.get("content")
    if isinstance(content, str):
        return content
    return None


def make_llm_parser() -> LLMCommandParser | OllamaCommandParser:
    """Select the LLM parser backend from ``PET_AGENT_LLM_BACKEND``.

    ``ollama`` → local :class:`OllamaCommandParser`; anything else (including
    unset, ``anthropic``, or an unknown value) → the Anthropic
    :class:`LLMCommandParser`. Construction never raises: a missing package or
    unreachable host surfaces lazily in ``parse`` → ``None`` → rule fallback.
    """
    backend = (os.environ.get("PET_AGENT_LLM_BACKEND") or "anthropic").strip().lower()
    if backend == "ollama":
        return OllamaCommandParser()
    return LLMCommandParser()
