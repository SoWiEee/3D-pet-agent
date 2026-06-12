"""Task 2 (§14.6.4) — per-session multi-turn clarification state.

When grounding is ambiguous the server asks a *discriminating* question and
remembers the pending turn keyed by ``session_id``. The user's reply (same
``session_id``) is folded into the prior intent and re-grounded.

Everything here has a deterministic fallback so the demo never blocks:

- ``discriminating_question`` builds a templated question from the candidate
  objects' class + distinguishing attribute (or a coarse left/right hint when
  attributes don't differ). An optional ``llm_parser`` may produce a nicer
  question, but any LLM failure silently falls back to the template — the call
  never raises.
- ``merge_followup`` re-parses the reply and folds a single discriminator
  (an attribute like "red"/"left", or a bare lexical cue) into the prior
  intent's ``target.attributes``, building a NEW ``CommandIntent`` (immutable).

The store is bounded (default 64 sessions, oldest-evicted) so it can't grow
without limit on a long-lived server.
"""

from __future__ import annotations

import json
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass

from ..spatial.semantic_map import SemanticMap
from .command_parser import parse_command
from .schema import CommandIntent, TargetSpec

log = logging.getLogger("pet_agent.dialogue")

DEFAULT_CAPACITY = 64

# Lexical discriminators a bare follow-up reply may carry ("left", "the red
# one"). Kept rule-simple; the rule parser already extracts colour/size
# attributes, this covers positional cues it doesn't treat as attributes.
_LEXICAL_DISCRIMINATORS: tuple[str, ...] = (
    "left",
    "right",
    "front",
    "back",
    "near",
    "far",
    "first",
    "second",
    "third",
)


@dataclass(frozen=True)
class PendingClarification:
    """One outstanding clarification turn awaiting the user's reply."""

    intent: CommandIntent
    candidates: list[tuple[str, float]]
    question: str
    created_at: float
    retries: int = 0


class DialogueStore:
    """Bounded, insertion-ordered map of ``session_id → PendingClarification``.

    Single-process async server; calls are synchronous, so no locking. Oldest
    entries are evicted once ``capacity`` is exceeded (FIFO eviction).
    """

    def __init__(self, *, capacity: int = DEFAULT_CAPACITY) -> None:
        self._capacity = max(1, capacity)
        self._pending: OrderedDict[str, PendingClarification] = OrderedDict()

    def __len__(self) -> int:
        return len(self._pending)

    def open_clarification(
        self,
        session_id: str,
        intent: CommandIntent,
        candidates: list[tuple[str, float]],
        question: str,
        *,
        retries: int = 0,
    ) -> PendingClarification:
        pending = PendingClarification(
            intent=intent,
            candidates=list(candidates),
            question=question,
            created_at=time.time(),
            retries=retries,
        )
        # Re-insert at the end (most-recent) and evict the oldest if over cap.
        self._pending.pop(session_id, None)
        self._pending[session_id] = pending
        while len(self._pending) > self._capacity:
            self._pending.popitem(last=False)
        return pending

    def get(self, session_id: str) -> PendingClarification | None:
        return self._pending.get(session_id)

    def resolve(self, session_id: str) -> None:
        self._pending.pop(session_id, None)


# ── discriminating question ─────────────────────────────────────────────────


def _coarse_side(x: float) -> str:
    """Left/right hint from a world-frame x coordinate (graphics-world: +x is
    screen-right). Centre-ish objects report 'middle'."""
    if x < -0.3:
        return "left"
    if x > 0.3:
        return "right"
    return "middle"


def _describe_candidate(track_id: str, semantic_map: SemanticMap, *, use_attr: bool) -> str:
    """Short human phrase for one candidate object."""
    obj = semantic_map.get(track_id)
    if obj is None:
        return track_id
    label = obj.class_label or "object"
    if use_attr and obj.attributes:
        return f"the {obj.attributes[0]} {label}"
    side = _coarse_side(obj.center_3d_world[0])
    return f"the {label} on the {side}" if side != "middle" else f"the {label} ({track_id})"


def _templated_question(candidates: list[tuple[str, float]], semantic_map: SemanticMap) -> str:
    """Deterministic fallback question naming each candidate's distinguishing
    feature: its first attribute if attributes differ, else a left/right hint."""
    ids = [tid for tid, _ in candidates]
    objs = [semantic_map.get(tid) for tid in ids]
    label = next((o.class_label for o in objs if o is not None and o.class_label), "object")

    attr_sets = [tuple(o.attributes) for o in objs if o is not None]
    attrs_differ = len({a for a in attr_sets}) > 1 and all(a for a in attr_sets)

    phrases = [_describe_candidate(tid, semantic_map, use_attr=attrs_differ) for tid in ids]
    if len(phrases) == 2:
        return f"Which {label} — {phrases[0]} or {phrases[1]}?"
    joined = ", ".join(phrases[:-1]) + f", or {phrases[-1]}"
    return f"Which {label} do you mean — {joined}?"


_QUESTION_SYSTEM_PROMPT = (
    "You help a 3D pet agent disambiguate a command. Several objects match what "
    "the user said. Given the candidates (class, attributes, position), write ONE "
    "short, natural question (max ~15 words) that asks the user which one they "
    "mean, referring to each candidate's most distinguishing feature. Output only "
    "the question text."
)


def discriminating_question(
    candidates: list[tuple[str, float]],
    semantic_map: SemanticMap,
    *,
    gen_client: object | None = None,
    model: str | None = None,
) -> str:
    """Ask the user which candidate they meant.

    The templated fallback is the default and is fully deterministic. When a
    generative ``gen_client`` (an Ollama text client) is supplied and yields
    usable prose, that LLM-authored question is used instead; any failure —
    unreachable host, empty output — falls back to the template. This function
    NEVER raises, so the default (hermetic) path always returns the template.
    """
    fallback = _templated_question(candidates, semantic_map)
    if gen_client is None:
        return fallback
    try:
        question = _llm_question(candidates, semantic_map, gen_client, model)
    except Exception as e:  # noqa: BLE001 — LLM path must never break clarification
        log.warning("LLM discriminating-question failed (%s); using template", e)
        return fallback
    return question or fallback


def _llm_question(
    candidates: list[tuple[str, float]],
    semantic_map: SemanticMap,
    gen_client: object,
    model: str | None,
) -> str | None:
    """LLM-authored disambiguating question via the local Ollama model. Returns
    ``None`` on any failure so the caller uses the deterministic template."""
    from .ollama_client import chat_text

    if model is None:
        try:
            from ..config import Settings

            model = Settings().ollama_model
        except Exception:  # noqa: BLE001 — config must not block the question
            return None

    objs: list[dict[str, object]] = []
    for tid, _ in candidates:
        obj = semantic_map.get(tid)
        if obj is None:
            continue
        objs.append(
            {
                "object_id": tid,
                "class": obj.class_label,
                "attributes": list(obj.attributes),
                "position": [round(v, 2) for v in obj.center_3d_world],
            }
        )
    if not objs:
        return None
    user = f"Candidates: {json.dumps(objs, ensure_ascii=False)}"
    return chat_text(gen_client, model=model, system=_QUESTION_SYSTEM_PROMPT, user=user)


# ── follow-up merge ─────────────────────────────────────────────────────────


def _extract_discriminator(reply_text: str) -> list[str]:
    """Pull discriminators out of a bare follow-up reply.

    Combines the rule parser's attribute extraction (colour/size) with a small
    lexical cue list (positional words the parser doesn't treat as attributes).
    """
    found: list[str] = []
    parsed = parse_command(reply_text)
    if parsed is not None and parsed.target is not None:
        found.extend(parsed.target.attributes)

    lowered = reply_text.lower()
    for cue in _LEXICAL_DISCRIMINATORS:
        if cue in lowered and cue not in found:
            found.append(cue)
    return found


def merge_followup(prior_intent: CommandIntent, reply_text: str) -> CommandIntent:
    """Fold the reply's discriminator into ``prior_intent``, returning a NEW
    ``CommandIntent`` (immutability — never mutate the prior turn).

    Keeps ``intent_type`` from the prior turn. If the reply carries no usable
    discriminator, returns the prior intent unchanged so re-grounding just
    retries the original utterance.
    """
    discriminators = _extract_discriminator(reply_text)
    if not discriminators:
        return prior_intent

    prior_target = prior_intent.target or TargetSpec()
    merged_attrs = list(prior_target.attributes)
    for d in discriminators:
        if d not in merged_attrs:
            merged_attrs.append(d)

    new_target = prior_target.model_copy(update={"attributes": merged_attrs})
    return prior_intent.model_copy(update={"target": new_target, "raw_text": reply_text})
