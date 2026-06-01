"""Exploration planner (spec §12).

Takes a :class:`SemanticMap` + :class:`CoverageGrid` + cat pose and emits
the next :class:`ExplorationGoal`. The selected goal can be lifted into a
:class:`NavigationGoal` so the existing A* planner runs unchanged — no
new path-planning code needed for exploration.

Heuristic (spec §12.1)::

    score = 0.40·expected_new_area
          + 0.25·semantic_uncertainty
          + 0.20·object_search_relevance
          - 0.15·travel_cost

All four components live in [0, 1] so weights are comparable.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Literal

from ..planning.schema import NavigationGoal
from ..spatial.semantic_map import SemanticMap
from .coverage_grid import CoverageGrid

log = logging.getLogger("pet_agent.exploration")

GoalKind = Literal["inspect_unknown", "search_object", "verify_stale", "look_behind"]


@dataclass(frozen=True)
class ExplorationCandidate:
    """One scored viewpoint considered by the planner."""

    kind: GoalKind
    target_position_world: tuple[float, float, float]
    expected_new_area: float
    semantic_uncertainty: float
    object_search_relevance: float
    travel_cost: float
    score: float
    rationale: str
    related_object_id: str | None = None


@dataclass(frozen=True)
class ExplorationGoal:
    """The exploration planner's output. Mirrors :class:`NavigationGoal`
    enough that callers can convert directly via :meth:`to_navigation_goal`."""

    kind: GoalKind
    target_position_world: tuple[float, float, float]
    score: float
    explanation: str
    related_object_id: str | None = None
    source_command: str = "explore"

    def to_navigation_goal(self) -> NavigationGoal:
        return NavigationGoal(
            goal_id=f"explore_{self.kind}",
            goal_type="viewpoint",
            target_position_world=self.target_position_world,
            target_object_id=self.related_object_id,
            source_command=self.source_command,
            explanation=self.explanation,
            score=self.score,
        )


@dataclass(frozen=True)
class ExplorationPlannerConfig:
    """Weights + heuristic knobs. Defaults match spec §12.1."""

    w_new_area: float = 0.40
    w_semantic_uncertainty: float = 0.25
    w_object_search: float = 0.20
    w_travel_cost: float = 0.15
    max_candidates_per_kind: int = 4
    min_cluster_cells: int = 16
    cluster_area_cap_cells: float = 400.0
    travel_cost_cap_m: float = 4.0


class ExplorationPlanner:
    """Pure planner — no I/O, no mutation of inputs."""

    def __init__(self, cfg: ExplorationPlannerConfig | None = None) -> None:
        self.cfg = cfg or ExplorationPlannerConfig()

    def candidates(
        self,
        semantic_map: SemanticMap,
        coverage: CoverageGrid,
        cat_xz: tuple[float, float],
        *,
        target_class: str | None = None,
    ) -> list[ExplorationCandidate]:
        out: list[ExplorationCandidate] = []
        out.extend(self._inspect_unknown_candidates(coverage, cat_xz))
        out.extend(self._verify_stale_candidates(semantic_map, cat_xz))
        if target_class is not None:
            out.extend(self._search_object_candidates(semantic_map, coverage, cat_xz, target_class))
        out.extend(self._look_behind_candidates(semantic_map, cat_xz))
        out.sort(key=lambda c: -c.score)
        return out

    def next_goal(
        self,
        semantic_map: SemanticMap,
        coverage: CoverageGrid,
        cat_xz: tuple[float, float],
        *,
        target_class: str | None = None,
    ) -> ExplorationGoal | None:
        cands = self.candidates(semantic_map, coverage, cat_xz, target_class=target_class)
        if not cands:
            return None
        top = cands[0]
        return ExplorationGoal(
            kind=top.kind,
            target_position_world=top.target_position_world,
            score=top.score,
            explanation=top.rationale,
            related_object_id=top.related_object_id,
        )

    # ── candidate generators ───────────────────────────────────────────────
    def _inspect_unknown_candidates(
        self, coverage: CoverageGrid, cat_xz: tuple[float, float]
    ) -> list[ExplorationCandidate]:
        clusters = coverage.unknown_clusters(min_cluster_cells=self.cfg.min_cluster_cells)
        out: list[ExplorationCandidate] = []
        for cluster in clusters[: self.cfg.max_candidates_per_kind]:
            cx = float(cluster["centroid_x"])
            cz = float(cluster["centroid_z"])
            cells = float(cluster["cell_count"])
            travel = math.hypot(cx - cat_xz[0], cz - cat_xz[1])
            expected = min(1.0, cells / self.cfg.cluster_area_cap_cells)
            uncertainty = 1.0
            object_relevance = 0.0
            travel_norm = min(1.0, travel / max(self.cfg.travel_cost_cap_m, 1e-9))
            score = self._combine(expected, uncertainty, object_relevance, travel_norm)
            out.append(
                ExplorationCandidate(
                    kind="inspect_unknown",
                    target_position_world=(cx, 0.0, cz),
                    expected_new_area=expected,
                    semantic_uncertainty=uncertainty,
                    object_search_relevance=object_relevance,
                    travel_cost=travel_norm,
                    score=score,
                    rationale=(
                        f"Inspect unknown region (~{int(cells)} cells) at ({cx:.2f}, {cz:.2f})."
                    ),
                )
            )
        return out

    def _verify_stale_candidates(
        self, semantic_map: SemanticMap, cat_xz: tuple[float, float]
    ) -> list[ExplorationCandidate]:
        out: list[ExplorationCandidate] = []
        stale = [o for o in semantic_map.values() if o.tracking_status in ("stale", "occluded")]
        stale.sort(key=lambda o: o.confidence.overall)
        for obj in stale[: self.cfg.max_candidates_per_kind]:
            x, _, z = obj.center_3d_world
            travel = math.hypot(x - cat_xz[0], z - cat_xz[1])
            uncertainty = 1.0 - float(obj.confidence.overall)
            travel_norm = min(1.0, travel / max(self.cfg.travel_cost_cap_m, 1e-9))
            expected = 0.1
            relevance = 0.5
            score = self._combine(expected, uncertainty, relevance, travel_norm)
            out.append(
                ExplorationCandidate(
                    kind="verify_stale",
                    target_position_world=(x, 0.0, z),
                    expected_new_area=expected,
                    semantic_uncertainty=uncertainty,
                    object_search_relevance=relevance,
                    travel_cost=travel_norm,
                    score=score,
                    rationale=(
                        f"Re-verify {obj.class_label} (id={obj.object_id}, "
                        f"status={obj.tracking_status})."
                    ),
                    related_object_id=obj.object_id,
                )
            )
        return out

    def _search_object_candidates(
        self,
        semantic_map: SemanticMap,
        coverage: CoverageGrid,
        cat_xz: tuple[float, float],
        target_class: str,
    ) -> list[ExplorationCandidate]:
        present = [
            o for o in semantic_map.values() if o.class_label.lower() == target_class.lower()
        ]
        if present:
            return []
        clusters = coverage.unknown_clusters(min_cluster_cells=self.cfg.min_cluster_cells)
        out: list[ExplorationCandidate] = []
        for cluster in clusters[: self.cfg.max_candidates_per_kind]:
            cx = float(cluster["centroid_x"])
            cz = float(cluster["centroid_z"])
            cells = float(cluster["cell_count"])
            travel = math.hypot(cx - cat_xz[0], cz - cat_xz[1])
            expected = min(1.0, cells / self.cfg.cluster_area_cap_cells)
            uncertainty = 1.0
            relevance = 1.0
            travel_norm = min(1.0, travel / max(self.cfg.travel_cost_cap_m, 1e-9))
            score = self._combine(expected, uncertainty, relevance, travel_norm)
            out.append(
                ExplorationCandidate(
                    kind="search_object",
                    target_position_world=(cx, 0.0, cz),
                    expected_new_area=expected,
                    semantic_uncertainty=uncertainty,
                    object_search_relevance=relevance,
                    travel_cost=travel_norm,
                    score=score,
                    rationale=(
                        f"Search for '{target_class}' in unknown region near ({cx:.2f}, {cz:.2f})."
                    ),
                )
            )
        return out

    def _look_behind_candidates(
        self, semantic_map: SemanticMap, cat_xz: tuple[float, float]
    ) -> list[ExplorationCandidate]:
        out: list[ExplorationCandidate] = []
        standoff = 0.4
        for obj in semantic_map.values():
            if obj.tracking_status != "tracked":
                continue
            ox, _, oz = obj.center_3d_world
            dx = ox - cat_xz[0]
            dz = oz - cat_xz[1]
            d = math.hypot(dx, dz)
            if d < 1e-3:
                continue
            tx = ox + standoff * dx / d
            tz = oz + standoff * dz / d
            travel = math.hypot(tx - cat_xz[0], tz - cat_xz[1])
            expected = 0.5
            uncertainty = 0.3
            relevance = 0.2
            travel_norm = min(1.0, travel / max(self.cfg.travel_cost_cap_m, 1e-9))
            score = self._combine(expected, uncertainty, relevance, travel_norm)
            out.append(
                ExplorationCandidate(
                    kind="look_behind",
                    target_position_world=(tx, 0.0, tz),
                    expected_new_area=expected,
                    semantic_uncertainty=uncertainty,
                    object_search_relevance=relevance,
                    travel_cost=travel_norm,
                    score=score,
                    rationale=f"Peek behind {obj.class_label} (id={obj.object_id}).",
                    related_object_id=obj.object_id,
                )
            )
        out.sort(key=lambda c: -c.score)
        return out[: self.cfg.max_candidates_per_kind]

    def _combine(
        self,
        expected_new_area: float,
        semantic_uncertainty: float,
        object_search_relevance: float,
        travel_cost: float,
    ) -> float:
        return (
            self.cfg.w_new_area * expected_new_area
            + self.cfg.w_semantic_uncertainty * semantic_uncertainty
            + self.cfg.w_object_search * object_search_relevance
            - self.cfg.w_travel_cost * travel_cost
        )
