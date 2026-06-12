"""§14.6.4 — LLM-assisted grounding.

When the heuristic :class:`~src.planning.grounding_resolver.GroundingResolver`
is low-confidence (``clarification`` / ``no_match`` with candidates), the scene
graph + candidate objects + the utterance go to the local Ollama model, which
picks the target ``object_id`` and writes a justification. The justification is
threaded into ``NavigationGoal.explanation`` so grounding stays explainable
(spec §3 — every NavigationGoal carries an explanation).

Gated by ``PET_AGENT_LLM_GROUNDING=on`` at the call site. This module never
raises: a missing package, unreachable host, malformed output, or a
**hallucinated** object id (one not among the candidates) all yield ``None`` so
the caller keeps the heuristic result.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import BaseModel

from ..spatial.scene_graph import SceneGraph
from ..spatial.semantic_map import SemanticMap

log = logging.getLogger("pet_agent.llm_grounding")

_SYSTEM_PROMPT = (
    "You are the grounding module of a 3D pet agent. Given a user utterance and "
    "a small set of candidate objects (with their class, attributes, and 3D "
    "position) plus spatial relations between them, pick the single object the "
    "user most likely means. Respond by emitting object_id (exactly one of the "
    "provided candidate ids) and a one-sentence justification. Never invent an "
    "id that is not in the candidate list."
)


class LLMTargetPick(BaseModel):
    """Structured output schema for the grounding pick."""

    object_id: str
    justification: str


def _candidate_context(
    candidates: list[tuple[str, float]],
    scene_graph: SceneGraph | None,
    semantic_map: SemanticMap,
) -> dict[str, Any]:
    """Compact JSON context: candidate objects + the relations among them."""
    ids = [cid for cid, _ in candidates]
    id_set = set(ids)
    objects: list[dict[str, Any]] = []
    for cid, score in candidates:
        obj = semantic_map.get(cid)
        if obj is None:
            continue
        objects.append(
            {
                "object_id": cid,
                "class": obj.class_label,
                "attributes": list(obj.attributes),
                "position": [round(v, 3) for v in obj.center_3d_world],
                "heuristic_score": round(score, 3),
            }
        )
    relations: list[dict[str, Any]] = []
    if scene_graph is not None:
        for edge in scene_graph.relations:
            if edge.subject in id_set and edge.object in id_set:
                relations.append(
                    {
                        "subject": edge.subject,
                        "relation": edge.relation,
                        "object": edge.object,
                    }
                )
    return {"candidate_objects": objects, "relations": relations}


def llm_pick_target(
    utterance: str,
    scene_graph: SceneGraph | None,
    candidates: list[tuple[str, float]],
    semantic_map: SemanticMap,
    *,
    client: Any | None = None,
    model: str | None = None,
    host: str | None = None,
) -> tuple[str, str] | None:
    """Ask the local model to pick a target from the candidates.

    Returns ``(object_id, justification)`` with ``object_id`` guaranteed to be
    one of the candidate ids, or ``None`` on any failure / hallucination.
    """
    if not candidates:
        return None
    candidate_ids = {cid for cid, _ in candidates}

    try:
        from ..config import Settings
        from ..language.ollama_client import chat_json, get_client

        if model is None or host is None:
            s = Settings()
            model = model or s.ollama_model
            host = host or s.ollama_host
        if client is None:
            client = get_client(host)
        if client is None:
            return None

        context = _candidate_context(candidates, scene_graph, semantic_map)
        user = (
            f"Utterance: {utterance!r}\n"
            f"Context: {json.dumps(context, ensure_ascii=False)}\n"
            f"Candidate ids: {sorted(candidate_ids)}"
        )
        data = chat_json(
            client,
            model=model,
            system=_SYSTEM_PROMPT,
            user=user,
            schema=LLMTargetPick.model_json_schema(),
        )
    except Exception as e:  # noqa: BLE001 — grounding must never break /command
        log.warning("LLM grounding failed (%s); keeping heuristic result", e)
        return None

    if data is None:
        return None
    try:
        pick = LLMTargetPick(**data)
    except Exception as e:  # noqa: BLE001 — schema mismatch → heuristic fallback
        log.warning("LLM grounding output failed schema validation (%s)", e)
        return None
    if pick.object_id not in candidate_ids:
        log.warning(
            "LLM grounding hallucinated id %r not in candidates; discarding",
            pick.object_id,
        )
        return None
    return pick.object_id, pick.justification
