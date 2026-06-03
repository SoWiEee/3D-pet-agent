"""Rule-based command parser (spec §9.1).

Handles a curated list of canonical command shapes used in the live demo and
the Phase 10 evaluation. Order matters — patterns are tried top-down, so the
most specific shapes come first.

LLM mode is gated by ``PET_AGENT_LLM_PARSER=on``. The expectation is the LLM
emits a JSON object that validates against ``CommandIntent``; on any failure
(network, schema, timeout) we silently fall back to the rule parser. The LLM
adapter itself is deferred — Phase 6 ships the rule path and the seam.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass

from .schema import (
    CommandIntent,
    ConstraintSpec,
    RelationSpec,
    RelationType,
    TargetSpec,
)

log = logging.getLogger("pet_agent.command_parser")

# Open-vocab labels seen in `configs/prompts.txt`. We do not gate the parser
# on this list — unknown labels still parse — but having a short bias list
# avoids accidentally treating common verbs ("hide", "stop") as object names.
_DEFAULT_ATTRIBUTES: tuple[str, ...] = (
    "red",
    "blue",
    "green",
    "yellow",
    "black",
    "white",
    "small",
    "large",
    "big",
    "tiny",
    "tall",
    "short",
)

_RELATION_WORDS: dict[str, RelationType] = {
    "left of": "left_of",
    "to the left of": "left_of",
    "right of": "right_of",
    "to the right of": "right_of",
    "in front of": "in_front_of",
    "behind": "behind",
    "above": "above",
    "on top of": "above",
    "below": "below",
    "under": "below",
    "underneath": "below",
    "near": "near",
    "next to": "near",
    "by": "near",
    "far from": "far_from",
    "away from": "far_from",
    "on": "on_surface",
    "between": "between",
}

# Order: longest phrases first so "to the left of" matches before "left of".
_RELATION_ORDER = sorted(_RELATION_WORDS, key=len, reverse=True)


@dataclass
class _ParseConfig:
    attributes: tuple[str, ...] = _DEFAULT_ATTRIBUTES


def _strip_articles(s: str) -> str:
    return re.sub(r"\b(?:the|a|an)\b", "", s, flags=re.IGNORECASE).strip()


def _split_target(phrase: str, cfg: _ParseConfig) -> TargetSpec:
    """Extract attributes + class_label from a noun phrase like "red cup"."""
    phrase = _strip_articles(phrase.lower()).strip()
    if not phrase:
        return TargetSpec()
    words = phrase.split()
    attrs: list[str] = []
    while words and words[0] in cfg.attributes:
        attrs.append(words.pop(0))
    label = " ".join(words).strip() or None
    return TargetSpec(class_label=label, attributes=attrs)


def _extract_relation(text: str, cfg: _ParseConfig) -> tuple[RelationSpec | None, str]:
    """Pull a leading "<rel> <anchor>" off the text; return (relation, rest)."""
    lowered = text.lower()
    for word in _RELATION_ORDER:
        pattern = rf"\b{re.escape(word)}\b\s+(.+)$"
        m = re.search(pattern, lowered)
        if not m:
            continue
        anchor_phrase = m.group(1).strip()
        # Strip trailing punctuation.
        anchor_phrase = re.split(r"\bbut\b|\bwhile\b|\band\b|,", anchor_phrase)[0].strip()
        anchor = _split_target(anchor_phrase, cfg)
        rest = lowered[: m.start()].strip()
        return RelationSpec(type=_RELATION_WORDS[word], anchor=anchor), rest
    return None, text


def _extract_constraints(text: str, cfg: _ParseConfig) -> tuple[list[ConstraintSpec], str]:
    """Find "but avoid …", "while staying away from …", etc."""
    out: list[ConstraintSpec] = []
    # Avoid clauses: "but avoid the mouse", "while avoiding the cup"
    avoid_pat = re.compile(
        r"\b(?:but\s+avoid|avoid|avoiding|stay\s+away\s+from)\s+(.+?)(?:\.|$|,)",
        re.IGNORECASE,
    )
    for m in avoid_pat.finditer(text):
        target = _split_target(m.group(1), cfg)
        if target.class_label:
            out.append(ConstraintSpec(type="avoid", object=target, min_distance=0.25))
    cleaned = avoid_pat.sub("", text).strip()
    return out, cleaned


def _is_self_word(s: str) -> bool:
    return s.strip().lower() in {"me", "user", "you"}


# Each rule returns a CommandIntent or None. They're tried in order.


def _rule_stop(text: str, _: _ParseConfig) -> CommandIntent | None:
    if re.match(r"^\s*(?:stop|halt|wait|stay|freeze)\b", text, re.IGNORECASE):
        return CommandIntent(raw_text=text, intent_type="stop", fallback="noop")
    return None


def _rule_explore(text: str, _: _ParseConfig) -> CommandIntent | None:
    if re.match(r"^\s*(?:explore|wander|look\s+around)\b", text, re.IGNORECASE):
        return CommandIntent(raw_text=text, intent_type="explore", fallback="noop")
    return None


def _rule_follow(text: str, cfg: _ParseConfig) -> CommandIntent | None:
    m = re.match(r"^\s*(?:follow|come\s+with)\s+(.+?)\s*$", text, re.IGNORECASE)
    if not m:
        return None
    rest = m.group(1)
    target = TargetSpec(class_label="user") if _is_self_word(rest) else _split_target(rest, cfg)
    return CommandIntent(raw_text=text, intent_type="follow", target=target)


def _rule_hide(text: str, cfg: _ParseConfig) -> CommandIntent | None:
    if not re.search(r"\bhide\b", text, re.IGNORECASE):
        return None
    rest = re.sub(r"^.*?\bhide\b\s*", "", text, count=1, flags=re.IGNORECASE).strip()
    constraints, rest = _extract_constraints(rest, cfg)
    relation, rest = _extract_relation(rest, cfg)
    if relation is None and rest:
        # "hide the cup" doesn't really make sense; assume the rest is the
        # anchor with an implicit "behind".
        anchor = _split_target(rest, cfg)
        relation = RelationSpec(type="behind", anchor=anchor)
    return CommandIntent(
        raw_text=text,
        intent_type="hide",
        target=relation.anchor if isinstance(relation.anchor, TargetSpec) else None,
        spatial_relation=relation,
        constraints=constraints,
    )


def _rule_look_at(text: str, cfg: _ParseConfig) -> CommandIntent | None:
    m = re.match(r"^\s*(?:look\s+at|stare\s+at|watch)\s+(.+?)\s*$", text, re.IGNORECASE)
    if not m:
        return None
    target = _split_target(m.group(1), cfg)
    return CommandIntent(raw_text=text, intent_type="look_at", target=target)


def _rule_inspect(text: str, cfg: _ParseConfig) -> CommandIntent | None:
    m = re.match(r"^\s*(?:inspect|examine|check)\s+(.+?)\s*$", text, re.IGNORECASE)
    if not m:
        return None
    target = _split_target(m.group(1), cfg)
    return CommandIntent(raw_text=text, intent_type="inspect", target=target)


def _rule_search(text: str, cfg: _ParseConfig) -> CommandIntent | None:
    m = re.match(r"^\s*(?:find|search\s+for|look\s+for)\s+(.+?)\s*$", text, re.IGNORECASE)
    if not m:
        return None
    target = _split_target(m.group(1), cfg)
    return CommandIntent(raw_text=text, intent_type="search", target=target)


def _rule_report(text: str, _: _ParseConfig) -> CommandIntent | None:
    if re.match(
        r"^\s*(?:what\s+do\s+you\s+see|describe|report|tell\s+me\s+what)\b",
        text,
        re.IGNORECASE,
    ):
        return CommandIntent(raw_text=text, intent_type="report", fallback="noop")
    return None


def _rule_pick_up(text: str, cfg: _ParseConfig) -> CommandIntent | None:
    """Mobile-manipulator pick: "pick up X", "pick X up", "grab/grasp X"
    (spec §14.5 Stage E). Grounds + navigates + emits a grasp sequence."""
    m = re.match(
        r"^\s*(?:pick\s+up|grab|grasp|fetch|pick)\s+(.+?)(?:\s+up)?\s*$",
        text,
        re.IGNORECASE,
    )
    if not m:
        return None
    rest = m.group(1).strip()
    constraints, rest = _extract_constraints(rest, cfg)
    relation, rest = _extract_relation(rest, cfg)
    target = _split_target(rest, cfg) if rest else None
    if not target or not target.class_label:
        return None
    return CommandIntent(
        raw_text=text,
        intent_type="pick_up",
        target=target,
        spatial_relation=relation,
        constraints=constraints,
    )


def _rule_move_to(text: str, cfg: _ParseConfig) -> CommandIntent | None:
    """Catch-all for "go to X", "approach X", with optional relation + avoid."""
    m = re.match(
        r"^\s*(?:go\s+(?:to|toward|towards)|move\s+(?:to|toward|towards)|"
        r"walk\s+to|head\s+to|approach|come\s+(?:to|here)|come)\s+(.+?)\s*$",
        text,
        re.IGNORECASE,
    )
    if not m:
        return None
    rest = m.group(1).strip()
    constraints, rest = _extract_constraints(rest, cfg)
    relation, rest = _extract_relation(rest, cfg)
    target = _split_target(rest, cfg) if rest else None
    # If only a relation was given (no leftover noun phrase), treat the anchor
    # as the target.
    if (
        (target is None or target.class_label is None)
        and relation
        and isinstance(relation.anchor, TargetSpec)
    ):
        target = relation.anchor
    return CommandIntent(
        raw_text=text,
        intent_type="move_to",
        target=target,
        spatial_relation=relation,
        constraints=constraints,
    )


# Bare object name → implicit move_to. Lowest priority.
def _rule_bare_target(text: str, cfg: _ParseConfig) -> CommandIntent | None:
    stripped = _strip_articles(text).strip()
    if not stripped or len(stripped.split()) > 3:
        return None
    target = _split_target(stripped, cfg)
    if not target.class_label:
        return None
    return CommandIntent(
        raw_text=text,
        intent_type="move_to",
        target=target,
        confidence=0.6,
    )


_RULES: tuple = (
    _rule_stop,
    _rule_explore,
    _rule_report,
    _rule_follow,
    _rule_hide,
    _rule_look_at,
    _rule_inspect,
    _rule_search,
    _rule_pick_up,
    _rule_move_to,
    _rule_bare_target,
)


class RuleCommandParser:
    """Rule-based parser. Stateless; instantiate once and call :meth:`parse`."""

    def __init__(self, *, attributes: Iterable[str] | None = None) -> None:
        self.cfg = _ParseConfig(attributes=tuple(attributes) if attributes else _DEFAULT_ATTRIBUTES)

    def parse(self, text: str) -> CommandIntent | None:
        if not text or not text.strip():
            return None
        for rule in _RULES:
            intent = rule(text, self.cfg)
            if intent is not None:
                log.debug("rule %s matched %r → %s", rule.__name__, text, intent.intent_type)
                return intent
        return None


# Module-level convenience used by the server endpoint.
_DEFAULT_PARSER = RuleCommandParser()

# Lazily built so importing this module never instantiates the LLM client.
_LLM_PARSER = None


def _get_llm_parser():  # type: ignore[no-untyped-def]
    """Cache one :class:`LLMCommandParser` instance — heavy SDK import lives
    inside it, so we only pay the cost when LLM mode is actually used."""
    global _LLM_PARSER
    if _LLM_PARSER is None:
        from .llm_parser import LLMCommandParser

        _LLM_PARSER = LLMCommandParser()
    return _LLM_PARSER


def _llm_parse(text: str) -> CommandIntent | None:
    """Try the LLM adapter. Returns None on any failure so the rule parser
    runs. Network / SDK errors are swallowed by :class:`LLMCommandParser`
    itself — this wrapper only protects against import-time errors."""
    try:
        parser = _get_llm_parser()
    except Exception as e:  # noqa: BLE001
        log.warning("LLM parser bootstrap failed (%s); rule parser will run", e)
        return None
    return parser.parse(text)


def parse_command(text: str) -> CommandIntent | None:
    """Parse ``text`` into a :class:`CommandIntent` or ``None``.

    Honours ``PET_AGENT_LLM_PARSER=on`` to consult an LLM adapter first; on
    any LLM failure the rule parser runs.
    """
    use_llm = os.environ.get("PET_AGENT_LLM_PARSER", "").strip().lower() == "on"
    if use_llm:
        try:
            llm_intent = _llm_parse(text)
        except Exception as e:  # noqa: BLE001 — adapter robustness over crash
            log.warning("LLM parser raised %r; falling back to rules", e)
            llm_intent = None
        if llm_intent is not None:
            return llm_intent
    return _DEFAULT_PARSER.parse(text)
