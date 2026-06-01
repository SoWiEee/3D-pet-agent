"""Ground a ``CommandIntent`` against the live SemanticMap + SceneGraph.

Spec §9.1 scoring::

    score = 0.35·semantic_match
          + 0.20·attribute_match
          + 0.25·relation_match
          + 0.10·visibility_score
          + 0.10·navigation_feasibility

The resolver returns a :class:`GroundingResult` discriminated union::

    success          → NavigationGoal carries a target position + explanation.
    clarification    → ambiguity margin too tight; explanation lists the
                       contenders so the renderer can ask the user.
    no_match         → no candidate cleared ``min_final_score``.
    empty_map        → nothing observed yet.

Phase 7's planner consumes ``NavigationGoal``. Phase 8's controller consumes
the planner's path. The resolver itself is pure and stateless — call it once
per user utterance.
"""

from __future__ import annotations

import logging
import math
import time
import uuid
from dataclasses import dataclass
from typing import Literal

from ..language.schema import CommandIntent, RelationSpec, TargetSpec
from ..spatial.object_lifter import ObjectState3D
from ..spatial.relation_scorer import RelationScorer
from ..spatial.scene_graph import SceneGraph
from ..spatial.semantic_map import SemanticMap
from .schema import NavigationConstraint, NavigationGoal

log = logging.getLogger("pet_agent.grounding")


@dataclass
class GroundingResult:
    """Discriminated result of one grounding call."""

    status: Literal["success", "clarification", "no_match", "empty_map"]
    goal: NavigationGoal | None = None
    candidates: list[tuple[str, float]] | None = None  # for clarification + debugging
    explanation: str = ""


# Weights from spec §9.1.
_W_SEMANTIC = 0.35
_W_ATTRIBUTE = 0.20
_W_RELATION = 0.25
_W_VISIBILITY = 0.10
_W_FEASIBILITY = 0.10


class GroundingResolver:
    """Stateless resolver over a SemanticMap + SceneGraph snapshot."""

    def __init__(
        self,
        scorer: RelationScorer | None = None,
        *,
        min_final_score: float = 0.65,
        ambiguity_margin: float = 0.12,
        approach_distance: float = 0.4,
    ) -> None:
        self.scorer = scorer or RelationScorer()
        self.min_final_score = min_final_score
        self.ambiguity_margin = ambiguity_margin
        self.approach_distance = approach_distance

    # ── public entrypoint ───────────────────────────────────────────────────
    def resolve(
        self,
        intent: CommandIntent,
        semantic_map: SemanticMap,
        scene_graph: SceneGraph | None = None,
    ) -> GroundingResult:
        objects = semantic_map.values()
        if not objects:
            return GroundingResult(
                status="empty_map",
                explanation="No objects in the map yet — wait for perception to populate.",
            )

        # Some intents don't need a target (stop, explore, report).
        if intent.intent_type in {"stop", "explore", "report"}:
            return self._goal_without_target(intent)

        # Score candidates against the named target. Hard-gate at semantic=0
        # when the intent named a class — the weighted sum can otherwise let a
        # totally wrong class clear the threshold via attribute/visibility
        # neutrals.
        has_class = intent.target is not None and intent.target.class_label is not None
        scored: list[tuple[ObjectState3D, float, dict[str, float]]] = []
        for obj in objects:
            breakdown = self._score(obj, intent, semantic_map, scene_graph)
            if has_class and breakdown["semantic"] <= 0.0:
                continue
            total = (
                _W_SEMANTIC * breakdown["semantic"]
                + _W_ATTRIBUTE * breakdown["attribute"]
                + _W_RELATION * breakdown["relation"]
                + _W_VISIBILITY * breakdown["visibility"]
                + _W_FEASIBILITY * breakdown["feasibility"]
            )
            scored.append((obj, total, breakdown))

        if not scored:
            return GroundingResult(
                status="no_match",
                explanation=(
                    f"No object matched the requested class {intent.target.class_label!r}."
                    if intent.target and intent.target.class_label
                    else "No candidate matched."
                ),
            )

        scored.sort(key=lambda x: -x[1])
        top, top_score, _ = scored[0]
        log.debug(
            "intent=%s target=%s → top=%s score=%.3f",
            intent.intent_type,
            intent.target.class_label if intent.target else None,
            top.object_id,
            top_score,
        )

        if top_score < self.min_final_score:
            return GroundingResult(
                status="no_match",
                candidates=[(o.object_id, s) for o, s, _ in scored[:3]],
                explanation=(
                    f"No candidate scored above {self.min_final_score:.2f} "
                    f"(best: {top.object_id}={top_score:.2f})."
                ),
            )

        if len(scored) > 1:
            second_score = scored[1][1]
            if top_score - second_score < self.ambiguity_margin:
                contenders = ", ".join(f"{o.object_id} ({s:.2f})" for o, s, _ in scored[:3])
                return GroundingResult(
                    status="clarification",
                    candidates=[(o.object_id, s) for o, s, _ in scored[:3]],
                    explanation=(
                        f"Ambiguous between top candidates: {contenders}. Which one did you mean?"
                    ),
                )

        # Resolve to a 3D target pose.
        anchor = top
        relation_for_pose = intent.spatial_relation
        target_xyz, explanation_parts = self._resolve_pose(
            anchor, intent, relation_for_pose, objects
        )
        constraints = _project_constraints(intent, objects)

        goal = NavigationGoal(
            goal_id=f"goal_{uuid.uuid4().hex[:8]}",
            goal_type="pose",
            target_object_id=anchor.object_id,
            target_position_world=tuple(target_xyz),
            target_orientation_hint=(
                "face_object" if intent.intent_type in {"look_at", "inspect"} else None
            ),
            constraints=constraints,
            source_command=intent.raw_text,
            explanation=_compose_explanation(intent, anchor, explanation_parts, constraints),
            score=top_score,
            timestamp=time.time(),
        )
        return GroundingResult(status="success", goal=goal, explanation=goal.explanation)

    # ── scoring ─────────────────────────────────────────────────────────────
    def _score(
        self,
        obj: ObjectState3D,
        intent: CommandIntent,
        semantic_map: SemanticMap,
        scene_graph: SceneGraph | None,
    ) -> dict[str, float]:
        semantic = _semantic_match(obj, intent.target)
        attribute = _attribute_match(obj, intent.target)
        relation = self._relation_match(obj, intent.spatial_relation, semantic_map, scene_graph)
        visibility = _visibility(obj)
        feasibility = _navigation_feasibility(obj)
        return {
            "semantic": semantic,
            "attribute": attribute,
            "relation": relation,
            "visibility": visibility,
            "feasibility": feasibility,
        }

    def _relation_match(
        self,
        obj: ObjectState3D,
        rel: RelationSpec | None,
        semantic_map: SemanticMap,
        scene_graph: SceneGraph | None,
    ) -> float:
        if rel is None:
            return 1.0  # neutral — no relation constraint specified
        # Anchor resolution: pick the best matching ObjectState3D for the anchor target.
        anchor_target = rel.anchor if isinstance(rel.anchor, TargetSpec) else None
        if anchor_target is None or anchor_target.class_label is None:
            return 0.5  # vague anchor → mild penalty
        candidates = [
            o
            for o in semantic_map.values()
            if _semantic_match(o, anchor_target) > 0.5 and o.object_id != obj.object_id
        ]
        if not candidates:
            return 0.0
        # ``obj`` *is* the spatial subject; the anchor is the relation object.
        score_fn = getattr(self.scorer, rel.type, None)
        if score_fn is None:
            return 0.0
        # If a scene_graph snapshot was passed, prefer its precomputed score
        # for stability with what the UI shows.
        if scene_graph is not None:
            for edge in scene_graph.relations:
                if (
                    edge.subject == obj.object_id
                    and edge.relation == rel.type
                    and any(c.object_id == edge.object for c in candidates)
                ):
                    return float(edge.score)
        return float(max(score_fn(obj, anchor) for anchor in candidates))

    # ── target pose resolution ──────────────────────────────────────────────
    def _resolve_pose(
        self,
        anchor: ObjectState3D,
        intent: CommandIntent,
        relation: RelationSpec | None,
        objects: list[ObjectState3D],
    ) -> tuple[tuple[float, float, float], list[str]]:
        """Pick a position near the anchor that respects intent + relation."""
        notes: list[str] = []
        x, y, z = anchor.center_3d_world

        if intent.intent_type == "look_at":
            return (x, y, z), ["facing the object"]

        if intent.intent_type == "follow":
            notes.append("trailing the object at safe distance")
            # Place the cat ``approach_distance`` toward the camera from the target.
            d = self.approach_distance
            return (x, 0.0, z + d), notes

        # hide / move_to with relation
        rel_type = relation.type if relation else None
        d = self.approach_distance
        if rel_type == "behind":
            target = (x, 0.0, z - d)
            notes.append("behind the anchor (further from camera)")
        elif rel_type == "in_front_of":
            target = (x, 0.0, z + d)
            notes.append("in front of the anchor (toward the camera)")
        elif rel_type == "left_of":
            target = (x - d, 0.0, z)
            notes.append("to the left")
        elif rel_type == "right_of":
            target = (x + d, 0.0, z)
            notes.append("to the right")
        elif rel_type == "near" or rel_type is None:
            # Standoff in the direction of the camera so the cat doesn't sit
            # *on* the object.
            target = (x, 0.0, z + d)
            notes.append(f"within {d:.2f} m of the object")
        else:
            target = (x, 0.0, z + d)
            notes.append(f"near anchor ({rel_type})")
        _ = objects  # constraint-aware positioning is Phase 7's job
        return target, notes

    # ── intents without a target ────────────────────────────────────────────
    def _goal_without_target(self, intent: CommandIntent) -> GroundingResult:
        kind = intent.intent_type
        if kind == "stop":
            goal = NavigationGoal(
                goal_id=f"goal_{uuid.uuid4().hex[:8]}",
                goal_type="pose",
                target_position_world=None,
                source_command=intent.raw_text,
                explanation="Stop and hold position.",
                score=1.0,
            )
            return GroundingResult(status="success", goal=goal, explanation=goal.explanation)
        if kind == "explore":
            goal = NavigationGoal(
                goal_id=f"goal_{uuid.uuid4().hex[:8]}",
                goal_type="region",
                source_command=intent.raw_text,
                explanation="Wander to gather new observations.",
                score=1.0,
            )
            return GroundingResult(status="success", goal=goal, explanation=goal.explanation)
        if kind == "report":
            goal = NavigationGoal(
                goal_id=f"goal_{uuid.uuid4().hex[:8]}",
                goal_type="viewpoint",
                source_command=intent.raw_text,
                explanation="Report what is currently visible.",
                score=1.0,
            )
            return GroundingResult(status="success", goal=goal, explanation=goal.explanation)
        return GroundingResult(
            status="no_match",
            explanation=f"Intent {kind} has no target and no fallback resolution.",
        )


# ── scoring helpers (pure) ─────────────────────────────────────────────────
def _semantic_match(obj: ObjectState3D, target: TargetSpec | None) -> float:
    if target is None:
        return 0.0
    if target.object_id and target.object_id == obj.object_id:
        return 1.0
    if target.class_label is None:
        return 0.0
    obj_label = obj.class_label.lower()
    want = target.class_label.lower()
    if want == obj_label:
        return 1.0
    if want in obj_label or obj_label in want:
        return 0.75
    return 0.0


def _attribute_match(obj: ObjectState3D, target: TargetSpec | None) -> float:
    if target is None or not target.attributes:
        return 1.0  # no attributes requested → neutral
    wanted = {a.lower() for a in target.attributes}
    have = {a.lower() for a in obj.attributes}
    if not have:
        return 0.5  # we don't know — give partial credit so the resolver
        # doesn't refuse to ground because the lifter hasn't populated colours.
    overlap = wanted & have
    return len(overlap) / len(wanted) if wanted else 1.0


def _visibility(obj: ObjectState3D) -> float:
    if obj.tracking_status == "tracked":
        return 1.0
    if obj.tracking_status == "occluded":
        return 0.7
    if obj.tracking_status == "stale":
        return 0.3
    return 0.05


def _navigation_feasibility(obj: ObjectState3D) -> float:
    """Cheap proxy until Phase 7 hooks the occupancy grid in.

    Penalise distant or extreme-y targets; reward objects sitting near the
    ground (where the cat can actually walk).
    """
    x, y, z = obj.center_3d_world
    dist = math.sqrt(x * x + y * y + z * z)
    reachable = math.exp(-((dist - 1.5) ** 2) / 8.0)  # peaks around 1.5 m
    grounded = math.exp(-((y) ** 2) / 0.5)  # near floor
    return float(0.6 * reachable + 0.4 * grounded)


def _project_constraints(
    intent: CommandIntent, objects: list[ObjectState3D]
) -> list[NavigationConstraint]:
    out: list[NavigationConstraint] = []
    for c in intent.constraints:
        if c.type == "avoid" and c.object and c.object.class_label:
            for o in objects:
                if _semantic_match(o, c.object) >= 0.75:
                    out.append(
                        NavigationConstraint(
                            type="avoid_object",
                            object_id=o.object_id,
                            min_distance=c.min_distance or 0.25,
                        )
                    )
        elif c.type == "stay_on_surface" and c.region_id:
            out.append(NavigationConstraint(type="stay_on_surface", region_id=c.region_id))
        elif c.type == "keep_distance" and c.min_distance is not None:
            out.append(NavigationConstraint(type="keep_distance", min_distance=c.min_distance))
    return out


def _compose_explanation(
    intent: CommandIntent,
    anchor: ObjectState3D,
    pose_notes: list[str],
    constraints: list[NavigationConstraint],
) -> str:
    bits: list[str] = []
    bits.append(f"{intent.intent_type}: {anchor.class_label} ({anchor.object_id})")
    if intent.spatial_relation:
        rel = intent.spatial_relation.type.replace("_", " ")
        bits.append(rel)
    if pose_notes:
        bits.append(pose_notes[0])
    if constraints:
        avoids = [c.object_id for c in constraints if c.type == "avoid_object"]
        if avoids:
            bits.append(f"avoiding {', '.join(avoids)}")
    return " — ".join(bits)
