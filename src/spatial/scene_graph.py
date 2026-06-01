"""Per-frame scene graph over the SemanticMap (spec §3.3 + §8).

A ``SceneGraph`` is a thin layer above ``SemanticMap``: for every observed
object pair (and selected triples), score the base spatial relations from
:class:`RelationScorer`, keep the edges that clear ``min_relation_score``, and
publish them as JSON. The frontend uses this for the relation-highlight debug
overlay; Phase 6's grounding resolver consumes it for ``CommandIntent``
constraints.

Performance: scoring is O(n²) over the map (and O(n³) for ``between`` if
enabled). Phase 5 caps ``between`` to the top-``between_top_k`` nearest pairs
to stay cheap on dense desks.
"""

from __future__ import annotations

import itertools
import logging
import time
from typing import Any, Literal

from pydantic import BaseModel, Field

from .object_lifter import ObjectState3D
from .relation_scorer import (
    BASE_PAIR_RELATIONS,
    TRIADIC_RELATIONS,
    RelationConfig,
    RelationScorer,
)
from .semantic_map import SemanticMap

log = logging.getLogger("pet_agent.scene_graph")

RelationLabel = Literal[
    "left_of",
    "right_of",
    "in_front_of",
    "behind",
    "above",
    "below",
    "near",
    "far_from",
    "on_surface",
    "occluding",
    "between",
]


class RelationEdge(BaseModel):
    """Spec §3.3 — a single scored edge between named objects."""

    subject: str
    relation: RelationLabel
    object: str
    score: float
    object_2: str | None = None  # for triadic relations (e.g. between)
    evidence: dict[str, Any] = Field(default_factory=dict)


class SceneGraph(BaseModel):
    """A frame-level snapshot of spatial relations."""

    timestamp: float = Field(default_factory=time.time)
    frame_id: int = 0
    coordinate_frame: str = "world"
    objects: list[str] = Field(default_factory=list)
    relations: list[RelationEdge] = Field(default_factory=list)

    def edges_for(self, object_id: str) -> list[RelationEdge]:
        return [r for r in self.relations if object_id in (r.subject, r.object, r.object_2)]

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class SceneGraphBuilder:
    """Iterate the SemanticMap each tick and emit a SceneGraph."""

    def __init__(
        self,
        scorer: RelationScorer | None = None,
        *,
        min_relation_score: float = 0.3,
        between_top_k: int = 6,
        include_between: bool = True,
    ) -> None:
        self.scorer = scorer or RelationScorer(RelationConfig())
        self.min_relation_score = min_relation_score
        self.between_top_k = between_top_k
        self.include_between = include_between

    def build(self, semantic_map: SemanticMap, *, frame_id: int | None = None) -> SceneGraph:
        objects = semantic_map.values()
        fid = frame_id if frame_id is not None else semantic_map.last_frame_id
        graph = SceneGraph(
            frame_id=fid,
            coordinate_frame=semantic_map.coordinate_frame,
            objects=[o.object_id for o in objects],
        )

        if len(objects) < 2:
            return graph

        # Pairwise relations — ordered pairs (subject, object) so asymmetric
        # relations like ``left_of`` are emitted in both directions where they
        # apply.
        for a, b in itertools.permutations(objects, 2):
            for rel in BASE_PAIR_RELATIONS:
                score = self._score_pair(rel, a, b)
                if score >= self.min_relation_score:
                    graph.relations.append(
                        RelationEdge(
                            subject=a.object_id,
                            relation=rel,  # type: ignore[arg-type]
                            object=b.object_id,
                            score=float(score),
                            evidence=_pair_evidence(a, b),
                        )
                    )

        if self.include_between and len(objects) >= 3:
            for a, b, c in self._between_candidates(objects):
                score = self.scorer.between(a, b, c)
                if score >= self.min_relation_score:
                    graph.relations.append(
                        RelationEdge(
                            subject=a.object_id,
                            relation="between",
                            object=b.object_id,
                            object_2=c.object_id,
                            score=float(score),
                            evidence=_triad_evidence(a, b, c),
                        )
                    )

        # Stable order: higher score first, then by (subject, relation, object).
        graph.relations.sort(
            key=lambda r: (-r.score, r.subject, r.relation, r.object, r.object_2 or "")
        )
        log.debug(
            "frame %d: scene graph has %d objects, %d edges",
            fid,
            len(graph.objects),
            len(graph.relations),
        )
        return graph

    # ── helpers ─────────────────────────────────────────────────────────────
    def _score_pair(self, rel: str, a: ObjectState3D, b: ObjectState3D) -> float:
        fn = getattr(self.scorer, rel)
        return float(fn(a, b))

    def _between_candidates(
        self, objects: list[ObjectState3D]
    ) -> list[tuple[ObjectState3D, ObjectState3D, ObjectState3D]]:
        """Return the top-K closest (B, C) pairs and pair each with every A
        that is *not* B or C. Avoids O(n³) explosion on dense scenes."""
        pairs: list[tuple[float, ObjectState3D, ObjectState3D]] = []
        for b, c in itertools.combinations(objects, 2):
            d = RelationScorer._distance(b, c)
            pairs.append((d, b, c))
        pairs.sort(key=lambda x: x[0])
        chosen = pairs[: max(1, self.between_top_k)]
        out: list[tuple[ObjectState3D, ObjectState3D, ObjectState3D]] = []
        ids = [o.object_id for o in objects]
        for _, b, c in chosen:
            for a in objects:
                if a.object_id in (b.object_id, c.object_id):
                    continue
                out.append((a, b, c))
            _ = ids
        return out


def _pair_evidence(a: ObjectState3D, b: ObjectState3D) -> dict[str, Any]:
    return {
        "subject_center_3d": list(a.center_3d_world),
        "object_center_3d": list(b.center_3d_world),
    }


def _triad_evidence(a: ObjectState3D, b: ObjectState3D, c: ObjectState3D) -> dict[str, Any]:
    return {
        "subject_center_3d": list(a.center_3d_world),
        "object_center_3d": list(b.center_3d_world),
        "object_2_center_3d": list(c.center_3d_world),
    }


__all__ = [
    "RelationEdge",
    "SceneGraph",
    "SceneGraphBuilder",
    "BASE_PAIR_RELATIONS",
    "TRIADIC_RELATIONS",
]
